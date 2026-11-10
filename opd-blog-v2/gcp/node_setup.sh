#!/usr/bin/env bash
# Runs ON the VM. gcp/02_setup_nodes.sh and status.sh on your Mac copy it to ~/opd_node_setup.sh and call one phase at
# a time; every phase can be run again safely (after a reboot, a preemption, or a dropped connection). Adapted from
# opd-full-experiments/teacher/node_setup.sh, with its fix for a half-stopped Ray cluster.
#   driver           Makes sure the NVIDIA driver is 580 or newer: SkyRL's torch is a CUDA 13 build, which a 570
#                    driver cannot run. The current image ships 580, so this installs nothing; on an older image
#                    it installs driver 580 + gIB and the nccl.h symlink (guide 2.1). Prints REBOOT_NEEDED when a
#                    reboot is due (after a driver install, or for the open-file limit on a fresh VM).
#   storage          RAID-0 the local NVMe SSDs at /mnt/local_storage (guide 3), point ~/.cache at it, create
#                    the store on the boot disk, write ~/.opd_cluster_env (single-node NCCL settings).
#   software         build tools, uv, SkyRL at $SKYRL_COMMIT with this kit inside it at $KIT_BRANCH, the base
#                    environment at ~/venvs/skyrl, and a warm uv cache for the runs' isolated environment.
#   software-bg      start `software` detached (it takes 10-20 minutes on a fresh node); software-status reports.
#   ray              start the single-node Ray cluster from the base environment, if it is not up.
#   stop-run <script>
#                    stop that run and Ray, and wait for the GPUs to free up. Its checkpoints and exports stay.
#   report <run> <script>
#                    one key=value per line, read by status.sh. <script> is the run's script under opd-blog-v2/.
set -euo pipefail
APT="sudo DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 -y"
STORE="$HOME/opd-store"
phase="${1:?phase: driver | storage | software | software-bg | software-status | ray | stop-run | report}"
# The pid file survives a preemption and a new boot can reuse the number, so check that it is still this script.
software_running() {
  local pid
  pid="$(cat "$HOME/.opd_software.pid" 2>/dev/null || true)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q 'software'
}
# `ray stop` kills GCS last and can exit before it gets there: on 2026-11-02 it left a lone gcs_server on both B200s.
ray_down() {
  "$HOME/venvs/skyrl/bin/ray" stop --force >/dev/null 2>&1 || true
  pkill -9 -x raylet || true
  pkill -9 -x gcs_server || true
}

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
      # OPD_RAID_EXCLUDE: a regex of drives to leave out, if one fails `mdadm --create` (guide: "nvme32n1 is not suitable")
      mapfile -t DRIVES < <(lsblk -d -n -o NAME,SIZE | awk '$1 ~ /^nvme/ && $2 == "375G" {print "/dev/" $1}' | grep -v -E "${OPD_RAID_EXCLUDE:-^$}")
      [[ ${#DRIVES[@]} -gt 0 ]] || { echo "found no 375G NVMe local SSDs (lsblk -d)" >&2; exit 1; }
      echo "creating RAID-0 over ${#DRIVES[@]} local SSDs"
      sudo mdadm --create /dev/md0 --level=0 --raid-devices="${#DRIVES[@]}" --run "${DRIVES[@]}"
      MD=/dev/md0
    fi
    [[ "$(sudo blkid -o value -s TYPE "$MD" 2>/dev/null || true)" == "ext4" ]] || sudo mkfs.ext4 -F -q "$MD"
    sudo mkdir -p /mnt/local_storage
    sudo mount "$MD" /mnt/local_storage
    sudo chmod 777 /mnt/local_storage
  fi
  # HF and uv caches grow to 50-100GB: keep them off the boot disk (guide 3). They are lost with a preemption.
  mkdir -p /mnt/local_storage/.cache
  if [[ ! -L "$HOME/.cache" ]]; then
    if [[ -d "$HOME/.cache" ]]; then cp -a "$HOME/.cache/." /mnt/local_storage/.cache/ 2>/dev/null || true; fi
    rm -rf "$HOME/.cache"
    ln -s /mnt/local_storage/.cache "$HOME/.cache"
  fi
  # Checkpoints, exports, data and logs: on the boot disk, which a preemption leaves intact.
  mkdir -p "$STORE/logs"
  cat > "$HOME/.opd_cluster_env" <<'EOF'
# Written by node_setup.sh. Sourced before `ray start` (Ray workers inherit the raylet's environment) and by
# 00_env.sh (the driver).
export PATH="$HOME/.local/bin:$PATH"
# One node: plain sockets and no gIB plugin, which would fail to initialise without the RDMA NICs
# (Charlie's guide, "For single-node training").
export NCCL_NET=Socket
export NCCL_NET_PLUGIN=none
# The first run on a cold node spends minutes building each worker's uv environment (SkyRL troubleshooting doc).
export RAY_worker_register_timeout_seconds=600
EOF
  grep -qF '.opd_cluster_env' "$HOME/.bashrc" 2>/dev/null || echo '[ -f ~/.opd_cluster_env ] && . ~/.opd_cluster_env' >> "$HOME/.bashrc"
  df -h /mnt/local_storage "$STORE" | sed 's/^/  /'
  ;;

software)
  rm -f "$HOME/.opd_software_ok"
  export PATH="$HOME/.local/bin:$PATH"
  : "${SKYRL_REPO:?}" "${SKYRL_COMMIT:?}" "${KIT_REPO:?}" "${KIT_BRANCH:?}"
  mountpoint -q /mnt/local_storage || { echo "/mnt/local_storage is not mounted: run the storage phase first" >&2; exit 1; }
  # SkyRL install doc: build-essential, and libnuma for the FSDP backend
  dpkg -s build-essential libnuma-dev >/dev/null 2>&1 || { $APT update; $APT install build-essential libnuma-dev; }
  command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
  [[ -d "$HOME/SkyRL/.git" ]] || git clone "$SKYRL_REPO" "$HOME/SkyRL"
  git -C "$HOME/SkyRL" fetch origin "$SKYRL_COMMIT"
  git -C "$HOME/SkyRL" checkout --detach "$SKYRL_COMMIT"
  # The workspace rule: the kit is cloned inside SkyRL/.
  [[ -d "$HOME/SkyRL/skyrl-test/.git" ]] || git clone "$KIT_REPO" "$HOME/SkyRL/skyrl-test"
  git -C "$HOME/SkyRL/skyrl-test" fetch origin "$KIT_BRANCH"
  git -C "$HOME/SkyRL/skyrl-test" checkout "$KIT_BRANCH"
  git -C "$HOME/SkyRL/skyrl-test" pull --ff-only origin "$KIT_BRANCH"
  cd "$HOME/SkyRL"
  # Base environment, outside the checkout as the install doc recommends (Ray ships the working directory).
  # Its `ray` starts the cluster, so the cluster runs exactly the Ray the lockfile pins.
  [[ -x "$HOME/venvs/skyrl/bin/python" ]] || uv venv --python 3.12 "$HOME/venvs/skyrl"
  VIRTUAL_ENV="$HOME/venvs/skyrl" UV_LINK_MODE=copy uv sync --active --extra fsdp
  # The runs and every Ray worker use `uv run --isolated --extra fsdp`: build that environment once now.
  uv run --isolated --extra fsdp python -c "import torch, vllm, ray; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| gpus', torch.cuda.device_count(), '| vllm', vllm.__version__, '| ray', ray.__version__)"
  echo "SkyRL    : $(git -C "$HOME/SkyRL" rev-parse --short HEAD)"
  echo "kit      : $(git -C "$HOME/SkyRL/skyrl-test" rev-parse --abbrev-ref HEAD) $(git -C "$HOME/SkyRL/skyrl-test" rev-parse --short HEAD)"
  touch "$HOME/.opd_software_ok"
  ;;

software-bg)
  if software_running; then
    echo "the software phase is already running (pid $(cat "$HOME/.opd_software.pid")); attaching"
  else
    nohup bash "$0" software > "$HOME/opd_setup_software.log" 2>&1 < /dev/null &
    echo $! > "$HOME/.opd_software.pid"
    echo "the software phase started (log: ~/opd_setup_software.log on the VM)"
  fi
  ;;

software-status)
  if software_running; then
    echo RUNNING; tail -n 1 "$HOME/opd_setup_software.log" 2>/dev/null | cut -c1-160
  elif [[ -f "$HOME/.opd_software_ok" ]]; then
    echo OK; tail -n 3 "$HOME/opd_setup_software.log" 2>/dev/null
  else
    echo FAILED; tail -n 40 "$HOME/opd_setup_software.log" 2>/dev/null
  fi
  ;;

ray)
  source "$HOME/.opd_cluster_env"
  RAY="$HOME/venvs/skyrl/bin/ray"
  [[ -x "$RAY" ]] || { echo "no $RAY: the software phase has not run" >&2; exit 1; }
  mountpoint -q /mnt/local_storage || { echo "/mnt/local_storage is not mounted: the storage phase has not run" >&2; exit 1; }
  # The raylet too: GCS alone answers `ray status`, and a run then finds no node to join.
  if pgrep -x raylet >/dev/null && "$RAY" status >/dev/null 2>&1; then
    echo "Ray is already up"
  else
    ray_down
    "$RAY" start --head --disable-usage-stats < /dev/null
    sleep 5
    pgrep -x raylet >/dev/null || { echo "Ray did not come up: no raylet" >&2; exit 1; }
  fi
  "$RAY" status 2>/dev/null | sed -n '/Resources/,$p' | sed -n '1,12p'
  echo "open-file limit: $(ulimit -n)   NCCL_NET=$NCCL_NET NCCL_NET_PLUGIN=$NCCL_NET_PLUGIN"
  ;;

stop-run)
  script="${2:?stop-run needs the run's script, e.g. phase1-opd/run_m2rl.sh}"
  pattern="opd-blog-v2/${script//./\\.}"
  pkill -f "$pattern" || true
  pkill -f 'opd-blog-v2/common/kit_main\.py' || true
  # Ray and the vLLM engines would outlive the driver for a while, and the next run needs their GPUs. The ray phase
  # starts Ray again.
  ray_down
  for _ in $(seq 1 "${V2_STOP_TRIES:-30}"); do
    if ! pgrep -f "$pattern" >/dev/null && ! pgrep -x 'raylet|gcs_server' >/dev/null &&
      [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
      echo "stopped"
      exit 0
    fi
    sleep 2
  done
  echo "the run, Ray or its GPU processes are still there:" >&2
  nvidia-smi --query-compute-apps=pid,name --format=csv,noheader >&2 || true
  exit 1
  ;;

report)
  run="${2:?report needs the run name}"; script="${3:?report needs the run's script}"
  log="$STORE/logs/$run.log"
  echo "active=$(pgrep -f "opd-blog-v2/${script//./\\.}" >/dev/null && echo 1 || echo 0)"
  echo "ever_started=$([[ -f "$log" ]] && echo 1 || echo 0)"
  echo "done=$(cat "$STORE/logs/$run.done" 2>/dev/null || true)"
  echo "ckpt_step=$(cat "$STORE/ckpts/$run/latest_ckpt_global_step.txt" 2>/dev/null || true)"
  # The newest tqdm line: "Training Batches Processed:  41%|####  | 412/1000 [20:36:00<29:24:00, 180.00s/it]"
  bar="$(tail -c 300000 "$log" 2>/dev/null | tr '\r' '\n' | grep -a 'Training Batches Processed' | tail -n 1 || true)"
  echo "step=$(sed -n 's/.*| *\([0-9][0-9]*\)\/[0-9][0-9]* *\[.*/\1/p' <<<"$bar")"
  echo "total=$(sed -n 's/.*| *[0-9][0-9]*\/\([0-9][0-9]*\) *\[.*/\1/p' <<<"$bar")"
  echo "eta=$(sed -n 's/.*<\([^,]*\),.*/\1/p' <<<"$bar")"
  if [[ -f "$log" ]]; then echo "log_age=$(( $(date +%s) - $(stat -c %Y "$log") ))"; else echo "log_age="; fi
  echo "gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk '{s += $1} END {if (NR) printf "%d", s / NR}')"
  echo "disk=$(df -BG --output=avail "$HOME" 2>/dev/null | tail -n 1 | tr -d ' ')"
  ;;

*) echo "unknown phase: $phase" >&2; exit 2 ;;
esac
