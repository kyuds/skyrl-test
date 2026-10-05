#!/usr/bin/env bash
# Runs ON the VM. The gcp/*.sh scripts on your Mac copy it to ~/opd_node_setup.sh and call one phase at a time;
# every phase can be run again safely (after a reboot, a preemption, or a dropped connection).
#   driver           NVIDIA driver 580 + gIB and the nccl.h symlink (guide 2.1). SkyRL's torch is a CUDA 13
#                    build, which the image's 570 driver cannot run. Prints REBOOT_NEEDED when a reboot is due.
#   storage          RAID-0 the local NVMe SSDs at /mnt/local_storage (guide 3), point ~/.cache at it, create
#                    the store on the boot disk, write ~/.opd_cluster_env (single-node NCCL settings).
#   software         build tools, uv, SkyRL at $SKYRL_BRANCH with this kit inside it at $KIT_BRANCH, the base
#                    environment at ~/venvs/skyrl, and a warm uv cache for the runs' isolated environment.
#   software-bg      start `software` detached (it takes 10-20 minutes on a fresh node); software-status reports.
#   ray [--restart]  start the single-node Ray cluster from the base environment.
#   status           one screen: GPUs, disks, Ray, finished runs, the newest log.
set -euo pipefail
APT="sudo DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 -y"
STORE="$HOME/opd-store"
phase="${1:?phase: driver | storage | software | software-bg | software-status | ray | status}"

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
# Written by gcp/node_setup.sh. Sourced before `ray start` (Ray workers inherit the raylet's environment) and
# by 00_env.sh (the driver).
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
  : "${SKYRL_REPO:?}" "${SKYRL_BRANCH:?}" "${KIT_REPO:?}" "${KIT_BRANCH:?}"
  mountpoint -q /mnt/local_storage || { echo "/mnt/local_storage is not mounted: run the storage phase first" >&2; exit 1; }
  # SkyRL install doc: build-essential, and libnuma for the FSDP backend
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
  cd "$HOME/SkyRL"
  # Base environment, outside the checkout as the install doc recommends (Ray ships the working directory).
  # Its `ray` starts the cluster, so the cluster runs exactly the Ray the lockfile pins.
  [[ -x "$HOME/venvs/skyrl/bin/python" ]] || uv venv --python 3.12 "$HOME/venvs/skyrl"
  VIRTUAL_ENV="$HOME/venvs/skyrl" UV_LINK_MODE=copy uv sync --active --extra fsdp
  # The runs and every Ray worker use `uv run --isolated --extra fsdp`: build that environment once now.
  uv run --isolated --extra fsdp python -c "import torch, vllm, ray; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| gpus', torch.cuda.device_count(), '| vllm', vllm.__version__, '| ray', ray.__version__)"
  echo "SkyRL    : $(git -C "$HOME/SkyRL" rev-parse --abbrev-ref HEAD) $(git -C "$HOME/SkyRL" rev-parse --short HEAD)"
  echo "kit      : $(git -C "$HOME/SkyRL/skyrl-test" rev-parse --abbrev-ref HEAD) $(git -C "$HOME/SkyRL/skyrl-test" rev-parse --short HEAD)"
  touch "$HOME/.opd_software_ok"
  ;;

software-bg)
  if [[ -f "$HOME/.opd_software.pid" ]] && kill -0 "$(cat "$HOME/.opd_software.pid")" 2>/dev/null; then
    echo "the software phase is already running (pid $(cat "$HOME/.opd_software.pid")); attaching"
  else
    nohup bash "$0" software > "$HOME/opd_setup_software.log" 2>&1 < /dev/null &
    echo $! > "$HOME/.opd_software.pid"
    echo "the software phase started (log: ~/opd_setup_software.log on the VM)"
  fi
  ;;

software-status)
  if [[ -f "$HOME/.opd_software.pid" ]] && kill -0 "$(cat "$HOME/.opd_software.pid")" 2>/dev/null; then
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
  [[ -x "$RAY" ]] || { echo "no $RAY: run 02_setup_node.sh first" >&2; exit 1; }
  mountpoint -q /mnt/local_storage || { echo "/mnt/local_storage is not mounted: run 02_setup_node.sh first" >&2; exit 1; }
  if [[ "${2:-}" == "--restart" ]]; then "$RAY" stop --force || true; sleep 3; fi
  if "$RAY" status >/dev/null 2>&1; then
    echo "Ray is already up (pass --restart to restart it; that kills a run in progress)"
  else
    "$RAY" start --head --disable-usage-stats < /dev/null
    sleep 5
  fi
  "$RAY" status 2>/dev/null | sed -n '/Resources/,$p' | sed -n '1,12p'
  echo "open-file limit: $(ulimit -n)   NCCL_NET=$NCCL_NET NCCL_NET_PLUGIN=$NCCL_NET_PLUGIN"
  ;;

status)
  echo "gpus   : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort | uniq -c | sed 's/^ *//' | tr '\n' ' ') driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
  echo "load   : $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ';')"
  echo "nvme   : $(mountpoint -q /mnt/local_storage && df -h --output=used,avail /mnt/local_storage | tail -1 || echo 'NOT MOUNTED (run 02_setup_node.sh)')"
  echo "store  : $(df -h --output=used,avail "$STORE" 2>/dev/null | tail -1)   (used, free on the boot disk)"
  echo "ray    : $("$HOME/venvs/skyrl/bin/ray" status 2>/dev/null | grep -m1 GPU || echo 'not running (run 03_start_ray.sh)')"
  echo "runs   :"
  for f in "$STORE"/logs/*.done; do
    [[ -e "$f" ]] || { echo "  none finished yet"; break; }
    run="$(basename "$f" .done)"
    echo "  $run  done $(cat "$f")  $(cat "$STORE/logs/$run.uploaded" 2>/dev/null || echo 'NOT uploaded')"
  done
  echo "active : $(pgrep -af '0[234]_run_' | grep -v pgrep | sed 's/^[0-9]* //' | head -3 | tr '\n' ';' || true)"
  newest="$(ls -t "$STORE"/logs/*.log 2>/dev/null | head -1 || true)"
  if [[ -n "$newest" ]]; then echo "newest log: $newest"; tail -n 6 "$newest" | cut -c1-200 | sed 's/^/  /'; fi
  ;;

*) echo "unknown phase: $phase" >&2; exit 2 ;;
esac
