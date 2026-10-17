#!/usr/bin/env bash
# Runs ON the VM. The gcp/*.sh scripts on your Mac copy it to ~/skycap_node_setup.sh and call one phase at a time;
# every phase can be run again safely (after a reboot, a preemption, or a dropped connection). Adapted from
# skyrl-test/opd-gcp-spot-full-repro/gcp/node_setup.sh (no Ray, no secrets; a serve phase instead).
#   driver           NVIDIA driver 580 or newer (SkyRL's torch is a CUDA 13 build). The current image ships it,
#                    so this installs nothing there; on an older image it installs 580 + gIB and the nccl.h
#                    symlink (guide 2.1). Prints REBOOT_NEEDED when a reboot is due (after a driver install, or
#                    for the open-file limit on a fresh VM).
#   storage          RAID-0 the local NVMe SSDs at /mnt/local_storage (guide 3) and point ~/.cache at it (model
#                    weights, uv and HF caches); write ~/.skycap_env.
#   software         build tools, uv, SkyRL at $SKYRL_BRANCH with skyrl-test inside it at $KIT_BRANCH, the
#                    environments vLLM and skycap run in, and the kit's _tools.sh (Node, Codex, skycap-viewer).
#   software-bg      start `software` detached (10-20 minutes on a fresh node); software-status reports.
#   serve [--restart]  start vLLM and four skycap servers from the kit's 01_engine.sh / 02_skycap.sh, detached:
#                    token mode on :8080, and :8081 with --length-as-completed; text mode (forwarding to vLLM's
#                    own /v1/responses) on :8090, and :8091 with --length-as-completed. Wait until they answer,
#                    and start skycap-viewer (:8765) on the records.
#   unserve          stop them.
#   status           one screen: GPUs, disks, the servers, the records.
set -euo pipefail
APT="sudo DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 -y"
KIT="$HOME/SkyRL/skyrl-test/skycap-responses"
phase="${1:?phase: driver | storage | software | software-bg | software-status | serve | unserve | status}"

up() { curl -sf -m 5 "$1" >/dev/null 2>&1; }

case "$phase" in
driver)
  ver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
  major="${ver%%.*}"
  if [[ "$major" =~ ^[0-9]+$ && "$major" -ge 580 ]]; then
    echo "NVIDIA driver $ver: ok"
  else
    echo "NVIDIA driver is '${ver:-not loaded}': installing 580"
    $APT update
    $APT install nvidia-driver-580-server-open nccl-gib
    sudo ln -sf /usr/local/gib/include/nccl.h /usr/local/cuda/include/nccl.h
    echo REBOOT_NEEDED
  fi
  # The open-file limits the startup script writes only reach sessions opened after a reboot.
  nofile="$(ulimit -n)"
  if [[ "$nofile" != "unlimited" && "$nofile" -lt 65536 ]]; then echo "open-file limit is $nofile"; echo REBOOT_NEEDED; fi
  ;;

storage)
  if mountpoint -q /mnt/local_storage; then
    echo "/mnt/local_storage is already mounted"
  else
    command -v mdadm >/dev/null 2>&1 || $APT install mdadm
    md_dev() { lsblk -d -n -o NAME,TYPE | awk '$2 ~ /^raid/ {print "/dev/" $1; exit}'; }
    MD="$(md_dev)"
    # After a plain reboot the array still exists and only needs assembling; after a preemption the SSDs come
    # back blank and it is created again.
    if [[ -z "$MD" ]]; then sudo mdadm --assemble --scan >/dev/null 2>&1 || true; MD="$(md_dev)"; fi
    if [[ -z "$MD" ]]; then
      mapfile -t DRIVES < <(lsblk -d -n -o NAME,SIZE | awk '$1 ~ /^nvme/ && $2 == "375G" {print "/dev/" $1}' | grep -v -E "${SKYCAP_RAID_EXCLUDE:-^$}" || true)
      if [[ ${#DRIVES[@]} -eq 0 ]]; then
        # A machine type with no local SSD attached: keep the caches on the boot disk instead.
        echo "no 375G NVMe local SSDs (lsblk -d): caches stay on the boot disk"
        sudo mkdir -p /mnt/local_storage
      else
        echo "creating RAID-0 over ${#DRIVES[@]} local SSDs"
        sudo mdadm --create /dev/md0 --level=0 --raid-devices="${#DRIVES[@]}" --run "${DRIVES[@]}"
        MD=/dev/md0
      fi
    fi
    if [[ -n "$MD" ]]; then
      [[ "$(sudo blkid -o value -s TYPE "$MD" 2>/dev/null || true)" == "ext4" ]] || sudo mkfs.ext4 -F -q "$MD"
      sudo mkdir -p /mnt/local_storage
      sudo mount "$MD" /mnt/local_storage
    fi
    sudo chmod 777 /mnt/local_storage
  fi
  # Model weights and the uv/HF caches: off the boot disk where there are local SSDs (guide 3). Lost with a
  # preemption, and fetched again by the next start.
  mkdir -p /mnt/local_storage/.cache
  if [[ ! -L "$HOME/.cache" ]]; then
    if [[ -d "$HOME/.cache" ]]; then cp -a "$HOME/.cache/." /mnt/local_storage/.cache/ 2>/dev/null || true; fi
    rm -rf "$HOME/.cache"
    ln -s /mnt/local_storage/.cache "$HOME/.cache"
  fi
  cat > "$HOME/.skycap_env" <<'EOF'
# Written by gcp/node_setup.sh; sourced by the serve phase and by login shells.
export PATH="$HOME/.local/bin:$PATH"
# One node: plain sockets and no gIB plugin, which fails to initialise without the RDMA NICs (Charlie's guide,
# "For single-node training"). Only matters with --tensor-parallel-size > 1.
export NCCL_NET=Socket
export NCCL_NET_PLUGIN=none
EOF
  grep -qF '.skycap_env' "$HOME/.bashrc" 2>/dev/null || echo '[ -f ~/.skycap_env ] && . ~/.skycap_env' >> "$HOME/.bashrc"
  df -h /mnt/local_storage "$HOME" | sed 's/^/  /'
  ;;

software)
  rm -f "$HOME/.skycap_software_ok"
  export PATH="$HOME/.local/bin:$PATH"
  : "${SKYRL_REPO:?}" "${SKYRL_BRANCH:?}" "${KIT_REPO:?}" "${KIT_BRANCH:?}"
  [[ -d /mnt/local_storage ]] || { echo "run the storage phase first" >&2; exit 1; }
  # SkyRL install doc: build-essential, and libnuma for the FSDP extra vLLM comes with
  dpkg -s build-essential libnuma-dev >/dev/null 2>&1 || { $APT update; $APT install build-essential libnuma-dev; }
  command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
  checkout() {   # $1 = dir, $2 = repo, $3 = branch
    [[ -d "$1/.git" ]] || git clone "$2" "$1"
    git -C "$1" fetch origin "$3"
    git -C "$1" checkout "$3"
    git -C "$1" pull --ff-only origin "$3"
  }
  checkout "$HOME/SkyRL" "$SKYRL_REPO" "$SKYRL_BRANCH"
  checkout "$HOME/SkyRL/skyrl-test" "$KIT_REPO" "$KIT_BRANCH"   # the workspace rule: the kit is cloned inside SkyRL/
  # The environments 01_engine.sh and 02_skycap.sh run in, built once now.
  (cd "$HOME/SkyRL" && uv run --isolated --extra fsdp python -c "import torch, vllm; print('torch', torch.__version__, '| gpus', torch.cuda.device_count(), '| vllm', vllm.__version__)")
  (cd "$HOME/SkyRL/skycap" && uv run --isolated --extra tokens python -c "import renderers, skycap.dialects.openai_responses; print('skycap with the Responses dialect, renderers', renderers.__name__)")
  CODEX_VERSION="${CODEX_VERSION:-}" bash "$KIT/_tools.sh"
  echo "SkyRL    : $(git -C "$HOME/SkyRL" rev-parse --abbrev-ref HEAD) $(git -C "$HOME/SkyRL" rev-parse --short HEAD)"
  echo "kit      : $(git -C "$HOME/SkyRL/skyrl-test" rev-parse --abbrev-ref HEAD) $(git -C "$HOME/SkyRL/skyrl-test" rev-parse --short HEAD)"
  touch "$HOME/.skycap_software_ok"
  ;;

software-bg)
  if [[ -f "$HOME/.skycap_software.pid" ]] && kill -0 "$(cat "$HOME/.skycap_software.pid")" 2>/dev/null; then
    echo "the software phase is already running (pid $(cat "$HOME/.skycap_software.pid")); attaching"
  else
    nohup bash "$0" software > "$HOME/skycap_setup_software.log" 2>&1 < /dev/null &
    echo $! > "$HOME/.skycap_software.pid"
    echo "the software phase started (log: ~/skycap_setup_software.log on the VM)"
  fi
  ;;

software-status)
  if [[ -f "$HOME/.skycap_software.pid" ]] && kill -0 "$(cat "$HOME/.skycap_software.pid")" 2>/dev/null; then
    echo RUNNING; tail -n 1 "$HOME/skycap_setup_software.log" 2>/dev/null | cut -c1-160
  elif [[ -f "$HOME/.skycap_software_ok" ]]; then
    echo OK; tail -n 3 "$HOME/skycap_setup_software.log" 2>/dev/null
  else
    echo FAILED; tail -n 40 "$HOME/skycap_setup_software.log" 2>/dev/null
  fi
  ;;

serve)
  source "$HOME/.skycap_env"
  [[ -f "$HOME/.skycap_software_ok" ]] || { echo "the software phase has not finished: run 02_setup_node.sh" >&2; exit 1; }
  if [[ "${2:-}" == "--restart" ]]; then
    pkill -f 'skycap serve' || true; pkill -f 'vllm serve' || true; pkill -f 'skycap-viewer.js .*--port 8765' || true; sleep 10
  fi
  mkdir -p "$KIT/runs"
  cd "$HOME/SkyRL"
  engine_port="${SKYCAP_ENGINE_PORT:-8000}"
  if up "http://127.0.0.1:$engine_port/v1/models"; then
    echo "vLLM is already up on :$engine_port"
  else
    echo "starting vLLM (${SKYCAP_MODEL:-the kit default}${SKYCAP_ENGINE_ARGS:+, $SKYCAP_ENGINE_ARGS}); log: $KIT/runs/engine.log"
    # shellcheck disable=SC2086 # SKYCAP_ENGINE_ARGS is a list of flags
    nohup bash "$KIT/01_engine.sh" ${SKYCAP_ENGINE_ARGS:-} > "$KIT/runs/engine.log" 2>&1 < /dev/null &
    for i in $(seq 1 180); do
      up "http://127.0.0.1:$engine_port/v1/models" && break
      if ! pgrep -f 'vllm serve' >/dev/null; then echo "vLLM exited:" >&2; tail -n 30 "$KIT/runs/engine.log" >&2; exit 1; fi
      [[ $((i % 6)) == 0 ]] && echo "  waiting for vLLM ($((i / 6)) min): $(tail -n 1 "$KIT/runs/engine.log" | cut -c1-140)"
      sleep 10
    done
    up "http://127.0.0.1:$engine_port/v1/models" || { echo "vLLM did not come up in 30 minutes" >&2; exit 1; }
  fi
  for port in 8080 8081 8090 8091; do
    if up "http://127.0.0.1:$port/healthz"; then echo "skycap is already up on :$port"; continue; fi
    length=0; [[ $port == 8081 || $port == 8091 ]] && length=1
    mode=tokens; [[ $port == 809? ]] && mode=text
    SKYCAP_PORT=$port SKYCAP_MODE=$mode SKYCAP_LENGTH_AS_COMPLETED=$length \
      nohup bash "$KIT/02_skycap.sh" > "$KIT/runs/skycap-$port.log" 2>&1 < /dev/null &
  done
  for port in 8080 8081 8090 8091; do
    for _ in $(seq 1 60); do up "http://127.0.0.1:$port/healthz" && break; sleep 5; done
    if up "http://127.0.0.1:$port/healthz"; then
      echo "skycap :$port  $(curl -s "http://127.0.0.1:$port/healthz")"
    else
      echo "skycap :$port did not come up:" >&2; tail -n 20 "$KIT/runs/skycap-$port.log" >&2; exit 1
    fi
  done
  # skycap-viewer on the records, for gcp/tunnel.sh to forward.
  if up http://127.0.0.1:8765/; then
    echo "skycap-viewer is already up on :8765"
  elif [[ -f "$KIT/.tools/skycap-viewer/bin/skycap-viewer.js" ]]; then
    nohup bash "$KIT/05_view.sh" > "$KIT/runs/viewer.log" 2>&1 < /dev/null &
    echo "skycap-viewer started on :8765 (log: $KIT/runs/viewer.log)"
  fi
  ;;

unserve)
  pkill -f 'skycap serve' || true; pkill -f 'vllm serve' || true; pkill -f 'skycap-viewer.js .*--port 8765' || true
  echo "stopped skycap, vLLM and skycap-viewer (records already written stay in $KIT/records)"
  ;;

status)
  echo "gpus    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort | uniq -c | sed 's/^ *//' | tr '\n' ' ') driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
  echo "load    : $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ';')"
  echo "cache   : $(df -h --output=used,avail /mnt/local_storage 2>/dev/null | tail -1)   boot: $(df -h --output=used,avail "$HOME" | tail -1)"
  echo "vllm    : $(up "http://127.0.0.1:${SKYCAP_ENGINE_PORT:-8000}/v1/models" && echo up || echo down)"
  for port in 8080 8081 8090 8091; do
    echo "skycap  : :$port $(curl -s -m 5 "http://127.0.0.1:$port/healthz" || echo down)"
  done
  for dir in "$KIT"/records/*/; do
    [[ -d "$dir" ]] || { echo "records : none yet"; break; }
    echo "records : $dir $(find "$dir" -maxdepth 1 -type f | wc -l) files"
  done
  newest="$(ls -t "$KIT"/runs/*.jsonl 2>/dev/null | head -1 || true)"
  [[ -z "$newest" ]] || echo "drive   : $newest ($(wc -l < "$newest") trajectories)"
  ;;

*) echo "unknown phase: $phase" >&2; exit 2 ;;
esac
