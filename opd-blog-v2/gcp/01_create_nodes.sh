#!/usr/bin/env bash
# Run on your Mac. Finds out how much spot compute there is: asks for up to 2 nodes of 8 x H100 and 2 nodes of
# 8 x B200, in every zone of every region that offers them, until it holds them all or 4 hours have passed.
#
#   bash skyrl-test/opd-blog-v2/gcp/01_create_nodes.sh          # prints the plan, creates nothing
#   nohup caffeinate -i bash skyrl-test/opd-blog-v2/gcp/01_create_nodes.sh --yes > ~/opd_v2_nodes.log 2>&1 < /dev/null &
#   touch ~/opd-blog-v2-gcp/stop                                # stops it early, keeping what it got
#
# Each node has its own worker, all running at once. A worker goes round the zones asking for its node, each from a
# different starting zone, and stops at the first one that gives it. The node names are fixed, so the caps cannot be
# exceeded, and running this again picks up where it left off.
#
#   Refused for capacity   go on to the next zone. Going round again a minute after the last pass.
#   Refused for quota      leave that kind alone in that region for 30 minutes. Not for good: requests in flight hold
#                          quota for a moment, and other people in the project free theirs.
#   Rate limit, 5xx        carry on.
#   gcloud login expired   keep waiting. `gcloud auth login` in another terminal is picked up by the running script.
#   Anything else          print it, and leave that kind alone in that zone for 10 minutes.
#
# Until the end, a node that was obtained and then preempted is started again in its own zone. At the end, requests
# still in flight may finish for 5 minutes, then the script prints what it holds and what refused it. It deletes
# nothing, and nodes keep billing until gcp/down.sh stops them.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/config.sh"
YES=0
case "${1:-}" in
  "") ;;
  --yes) YES=1 ;;
  *) echo "usage: $0 [--yes]" >&2; exit 2 ;;
esac
STATE="$GCP_STATE_DIR"
mkdir -p "$STATE/held" "$STATE/cool"
if acquire_running; then echo "01_create_nodes.sh is already running (pid $(cat "$STATE/acquire.pid"))" >&2; exit 1; fi

log() { echo "$(date +%H:%M:%S) $*"; }
hm()  { printf '%dh%02dm' $(($1 / 3600)) $(($1 % 3600 / 60)); }

# ---- the plan ----
EXISTING="$(nodes_rows)"     # also shows that the gcloud login works
IMAGE="$(gcloud compute images describe-from-family "$GCP_IMAGE_FAMILY" --project "$GCP_IMAGE_PROJECT" --format='value(name)' 2>/dev/null || true)"
if [[ -z "$IMAGE" ]]; then
  echo "image family $GCP_IMAGE_FAMILY is not available in $GCP_IMAGE_PROJECT (families are retired over time). Current ones:" >&2
  gcloud compute images list --project "$GCP_IMAGE_PROJECT" --no-standard-images --format='value(family)' 2>/dev/null | sort -u | sed 's/^/  /' >&2
  echo "Pick the PyTorch one for Ubuntu 22.04 and re-run with GCP_IMAGE_FAMILY=<family>." >&2
  exit 1
fi
gcloud compute networks describe "$GCP_NETWORK" --project "$GCP_PROJECT" >/dev/null 2>&1 || {
  echo "network $GCP_NETWORK does not exist in $GCP_PROJECT; set GCP_NETWORK/GCP_SUBNET" >&2; exit 1; }

echo "project $GCP_PROJECT: spot nodes, for at most $(hm "$GCP_TIMEOUT_SECS")"
TOTAL=0
for kind in $GCP_KINDS; do
  n="$(kind_count "$kind")"
  [[ "$n" -gt 0 ]] || continue
  zones="$(kind_zones "$kind")"
  if [[ -z "${zones// /}" ]]; then echo "no zone offers $(kind_machine "$kind")" >&2; exit 1; fi
  echo "$zones" > "$STATE/zones.$kind"
  regions="$(tr ' ' '\n' <<<"$zones" | sed -E 's/-[a-z]$//' | grep . | sort -u | wc -l | tr -d ' ')"
  echo "  $kind: up to $n x $(kind_machine "$kind"), $(node_name "$kind" 1) to -$n, in $(wc -w <<<"$zones" | tr -d ' ') zones of $regions regions"
  echo "    $zones"
  TOTAL=$((TOTAL + n))
done
[[ $TOTAL -gt 0 ]] || { echo "nothing to ask for: GCP_H100_NODES and GCP_B200_NODES are both 0" >&2; exit 1; }
echo "  image $IMAGE, ${GCP_BOOT_DISK_GB}GB boot disk, network $GCP_NETWORK/$GCP_SUBNET, tags $GCP_TAGS"
if [[ -n "$EXISTING" ]]; then echo "  already there:"; sed 's/^/    /' <<<"$EXISTING"; fi
if [[ $YES == 0 ]]; then
  echo "Plan only, nothing was created. Each node bills for 8 GPUs from the moment it is up. To start, add --yes."
  exit 0
fi

# ---- what a refusal means ----
# Order matters: an API rate limit says "Quota exceeded" too, and must not read as a quota refusal for GPUs.
AUTH_ERRORS='reauth|auth login|invalid_grant'
RATE_ERRORS='rate.?limit|quota metric|quota group|per minute|per 100 seconds'
CAPACITY_ERRORS='RESOURCE_POOL_EXHAUSTED|not have enough resources|currently unavailable|stockout|try again later'
QUOTA_ERRORS='QUOTA_EXCEEDED|quota .*exceeded'
TRANSIENT_ERRORS='database is locked|internal error|backendError|HTTPError 50[0-9]|Error 50[0-9]|connection (reset|aborted|refused)|timed out|temporary failure|server not found|ServerNotFound|remote end closed'
classify() {
  if grep -qiE "$AUTH_ERRORS" <<<"$1"; then echo auth
  elif grep -qiE "$RATE_ERRORS" <<<"$1"; then echo transient
  elif grep -qiE "$CAPACITY_ERRORS" <<<"$1"; then echo capacity
  elif grep -qiE "$QUOTA_ERRORS" <<<"$1"; then echo quota
  elif grep -qiE "$TRANSIENT_ERRORS" <<<"$1"; then echo transient
  else echo other
  fi
}

# ---- one worker per node ----
# attempts.tsv: epoch, node, kind, zone, result, detail. The report at the end is read from it.
record()  { printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%s)" "$1" "$2" "$3" "$4" "$(tr '\t\n' '  ' <<<"${5:-}" | cut -c1-220)" >> "$STATE/attempts.tsv"; }
time_up() { [[ -e "$STATE/stop" || $(date +%s) -ge $DEADLINE ]]; }
nap()     { local end=$(($(date +%s) + $1)); while [[ $(date +%s) -lt $end ]] && ! time_up; do sleep 1; done; }
cooling() { [[ -f "$STATE/cool/$1" && $(date +%s) -lt $(cat "$STATE/cool/$1") ]]; }
cool()    { echo $(($(date +%s) + $2)) > "$STATE/cool/$1"; }
first_error() { local e; e="$(sed -n '/^ERROR/,$p' <<<"$1" | grep . || true)"; head -"$2" <<<"${e:-$1}"; }

# "<status> <zone>" of a node, empty when it does not exist. Fails when gcloud cannot say.
node_row() {
  local rows
  rows="$(gcloud compute instances list --project "$GCP_PROJECT" --filter="name=$1" --format='value(status,zone.basename())' 2>"$STATE/err.$1")" || return 1
  grep -m1 '^RUNNING' <<<"$rows" || head -1 <<<"$rows"
}
# The login can expire in the middle of a run. Says so every 10 minutes per node, and keeps going.
LAST_WARNING=0
warn_gcloud() {
  local now; now="$(date +%s)"
  [[ $((now - LAST_WARNING)) -ge 600 ]] || return 0
  LAST_WARNING=$now
  if [[ "$(classify "$2")" == auth ]]; then log "$1: gcloud's login has expired. Run  gcloud auth login  and this script carries on."
  else log "$1: gcloud failed, still trying: $(head -2 <<<"$2" | tr '\n' ' ' | cut -c1-200)"; fi
}
create_in() {   # $1 = kind, $2 = zone, $3 = node. --instance-termination-action=STOP keeps the boot disk on preemption.
  local disk=()
  if [[ -n "$(kind_disk "$1")" ]]; then disk=(--boot-disk-type "$(kind_disk "$1")"); fi
  gcloud compute instances create "$3" \
    --project "$GCP_PROJECT" --zone "$2" \
    --machine-type "$(kind_machine "$1")" \
    --provisioning-model=SPOT --instance-termination-action=STOP --maintenance-policy=TERMINATE \
    --scopes=cloud-platform \
    --image-project "$GCP_IMAGE_PROJECT" --image-family "$GCP_IMAGE_FAMILY" \
    --boot-disk-size "${GCP_BOOT_DISK_GB}GB" ${disk[@]+"${disk[@]}"} \
    --tags "$GCP_TAGS" \
    --network-interface "nic-type=GVNIC,network=$GCP_NETWORK,subnet=$GCP_SUBNET" \
    --metadata-from-file "startup-script=$GCP_DIR/vm_startup.sh"
}
# The kind's zones, starting somewhere different for each of its nodes.
rotated_zones() {
  local zones n off
  read -r -a zones < "$STATE/zones.$1"
  n=${#zones[@]}
  off=$((($2 - 1) * n / $(kind_count "$1")))
  echo "${zones[@]:off}" "${zones[@]:0:off}"
}
# Asks for the node in each zone once. Returns 0 when it was created, 1 otherwise.
sweep() {   # $1 = kind, $2 = id, $3 = node
  local kind="$1" name="$3" zone region out result row
  for zone in $(rotated_zones "$kind" "$2"); do
    time_up && return 1
    region="${zone%-*}"
    if cooling "$kind-$region" || cooling "$kind-$zone"; then continue; fi
    if out="$(create_in "$kind" "$zone" "$name" 2>&1)"; then
      record "$name" "$kind" "$zone" created ""
      log "GOT $name in $zone after $(hm $(($(date +%s) - START)))"
      return 0
    fi
    result="$(classify "$out")"
    case "$result" in
      capacity)
        record "$name" "$kind" "$zone" capacity ""
        if [[ "${GCP_VERBOSE:-0}" == "1" ]]; then log "$name $zone: $(first_error "$out" 2 | tr '\n' ' ' | cut -c1-200)"; fi ;;
      quota)
        out="$(grep -iE "$QUOTA_ERRORS" <<<"$out" | head -1)"
        record "$name" "$kind" "$zone" quota "$out"
        if ! cooling "$kind-$region"; then log "$kind in $region refused for quota, leaving it for $(hm "$GCP_QUOTA_COOLDOWN"): $(cut -c1-160 <<<"$out")"; fi
        cool "$kind-$region" "$GCP_QUOTA_COOLDOWN" ;;
      auth)
        record "$name" "$kind" "$zone" auth ""
        warn_gcloud "$name" "$out"
        return 1 ;;
      *)
        out="$(first_error "$out" 4)"
        record "$name" "$kind" "$zone" "$result" "$out"
        if [[ "$result" == other ]]; then
          if ! cooling "$kind-$zone"; then log "$kind in $zone refused, leaving it for $(hm "$GCP_OTHER_COOLDOWN"):"; sed 's/^/    /' <<<"$out"; fi
          cool "$kind-$zone" "$GCP_OTHER_COOLDOWN"
        fi
        # The request may have worked even though gcloud reported a failure. Asking elsewhere would make a second node.
        if row="$(node_row "$name")"; then
          case "$row" in RUNNING*|PROVISIONING*|STAGING*) log "$name exists after all ($row)"; return 0 ;; esac
        fi ;;
    esac
  done
  return 1
}
worker() {   # $1 = kind, $2 = id
  local kind="$1" id="$2" name row st zone out result had=0
  name="$(node_name "$kind" "$id")"
  while ! time_up; do
    if ! row="$(node_row "$name")"; then
      warn_gcloud "$name" "$(cat "$STATE/err.$name" 2>/dev/null)"
      nap "$GCP_POLL_SECS"
      continue
    fi
    st=""; zone=""
    read -r st zone _ <<<"$row"
    if [[ "$st" == RUNNING ]]; then
      if [[ ! -e "$STATE/held/$name" ]]; then touch "$STATE/held/$name"; log "$name is RUNNING in $zone"; fi
      had=1
      nap "$GCP_POLL_SECS"
      continue
    fi
    rm -f "$STATE/held/$name"
    case "$st" in
      "")
        if [[ $had == 1 ]]; then log "$name no longer exists; asking for it again"; had=0; fi
        sweep "$kind" "$id" "$name" || nap "$GCP_RETRY_SLEEP" ;;
      TERMINATED|SUSPENDED)
        if [[ $had == 1 ]]; then record "$name" "$kind" "$zone" preempted ""; log "$name was preempted in $zone; starting it again"; had=0; fi
        if out="$(gcloud compute instances start "$name" --project "$GCP_PROJECT" --zone "$zone" 2>&1)"; then
          record "$name" "$kind" "$zone" started ""
          log "GOT $name back in $zone"
        else
          result="$(classify "$out")"
          record "$name" "$kind" "$zone" "$result" "$(first_error "$out" 2)"
          if [[ "$result" == auth ]]; then warn_gcloud "$name" "$out"; fi
          nap "$GCP_RETRY_SLEEP"
        fi ;;
      *) nap 15 ;;     # PROVISIONING, STAGING, REPAIRING, or the record a refused request leaves for a few seconds
    esac
  done
}

# ---- run them ----
progress() {
  local kind want have line="" counts left
  for kind in $GCP_KINDS; do
    want="$(kind_count "$kind")"
    [[ "$want" -gt 0 ]] || continue
    have="$(ls "$STATE/held" | grep -c -- "-$kind-" || true)"
    line="$line $kind $have/$want"
  done
  counts="$(awk -F'\t' -v s="$START" '$1 >= s {c[$5]++} END {printf "%d capacity, %d quota, %d other", c["capacity"], c["quota"], c["other"] + c["transient"] + c["auth"]}' "$STATE/attempts.tsv")"
  left=$((DEADLINE - $(date +%s))); if [[ $left -lt 0 ]]; then left=0; fi
  log "held:$line   $(hm "$left") left   refusals so far: $counts"
}
report() {
  local rows kind want have gpus=0
  echo
  echo "==== finished after $(hm $(($(date +%s) - START))) ===="
  if rows="$(nodes_rows)"; then
    if [[ -n "$rows" ]]; then column -t <<<"$rows"; fi
    for kind in $GCP_KINDS; do
      want="$(kind_count "$kind")"
      have="$(awk -v k="-$kind-" 'index($1, k) && $4 == "RUNNING"' <<<"$rows" | wc -l | tr -d ' ')"
      echo "$kind: $have of $want running"
      gpus=$((gpus + have * 8))
    done
    echo "GPUs held: $gpus"
    if [[ -n "$(awk '{print $1}' <<<"$rows" | sort | uniq -d)" ]]; then
      echo "WARNING: a node name exists in two zones. Delete one: GCP_ZONE=<zone> bash gcp/down.sh delete <node>"
    fi
  else
    echo "gcloud could not list the nodes, so what exists is not known. After logging in: bash gcp/status.sh"
  fi
  echo
  echo "requests by region:"
  printf '%-5s %-26s %9s %6s %6s %5s\n' kind region capacity quota other got
  awk -F'\t' -v s="$START" '$1 >= s && $4 != "" {
      z = $4; sub(/-[a-z]$/, "", z); k = $3 " " z; seen[k] = 1; c[k, $5]++ }
    END { for (k in seen) { split(k, p, " ")
      printf "%-5s %-26s %9d %6d %6d %5d\n", p[1], p[2], c[k, "capacity"], c[k, "quota"], c[k, "other"] + c[k, "transient"] + c[k, "auth"], c[k, "created"] + c[k, "started"] } }' \
    "$STATE/attempts.tsv" | sort
  echo
  echo "quota refusals:"
  awk -F'\t' -v s="$START" '$1 >= s && $5 == "quota" {print $6}' "$STATE/attempts.tsv" | sed -E 's/^.*(Quota|QUOTA)/\1/' | sort | uniq -c | sort -rn | head -12
  echo
  echo "Every node above keeps billing until you stop it: bash skyrl-test/opd-blog-v2/gcp/down.sh stop all"
}

rm -f "$STATE/stop" "$STATE"/held/* "$STATE"/cool/* "$STATE"/err.*
touch "$STATE/attempts.tsv"
START="$(date +%s)"
DEADLINE=$((START + GCP_TIMEOUT_SECS))
echo $$ > "$STATE/acquire.pid"
echo "$DEADLINE" > "$STATE/deadline"
trap 'touch "$STATE/stop"' INT TERM
log "started, $TOTAL workers. To stop early: touch $STATE/stop"

PIDS=()
for kind in $GCP_KINDS; do
  n="$(kind_count "$kind")"
  for id in $(seq 1 "$n"); do
    ( set +e; worker "$kind" "$id" ) &
    PIDS+=($!)
  done
done

NEXT=$(($(date +%s) + GCP_MONITOR_SECS))
while ! time_up; do
  if [[ "$(ls "$STATE/held" | wc -l | tr -d ' ')" -ge $TOTAL ]]; then
    log "holding all $TOTAL nodes"
    break
  fi
  if [[ $(date +%s) -ge $NEXT ]]; then progress; NEXT=$(($(date +%s) + GCP_MONITOR_SECS)); fi
  sleep 2
done
touch "$STATE/stop"
progress

# A request in flight may still come through, so give it time, and report from what GCP says exists.
END=$(($(date +%s) + GCP_GRACE_SECS))
while [[ $(date +%s) -lt $END ]]; do
  alive=0
  for pid in "${PIDS[@]}"; do if kill -0 "$pid" 2>/dev/null; then alive=$((alive + 1)); fi; done
  [[ $alive -gt 0 ]] || break
  sleep 2
done
for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
report
rm -f "$STATE/acquire.pid" "$STATE"/err.*
