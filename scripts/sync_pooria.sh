#!/usr/bin/env bash
set -euo pipefail

# Move work between this machine and pooria, which has no git repo of its own: code only ever travels
# out, results only ever come back.
#
#   bash scripts/sync_pooria.sh push     # send src/, scripts/ and configs/ before launching there
#   bash scripts/sync_pooria.sh pull     # bring finished runs back into models/
#   bash scripts/sync_pooria.sh          # push, then pull
#
# Set `weights=1` on a pull to include the checkpoints; they are excluded by default because a single
# run's *.pt is larger than every metric, curve and prediction in the project put together, and
# nothing local reads them -- reporting and plotting work off history.csv, metrics.csv and
# predictions/.

remote=${remote:-pooria}
remote_dir=${remote_dir:-UltrAi/projects/fm_adaptation}
results_dir=${results_dir:-models}
mode=${1:-both}

cd "$(dirname "$0")/.."

push() {
    for dir in src scripts configs; do
        echo "-> $remote:$remote_dir/$dir/"
        rsync -av --delete --exclude='__pycache__' "$dir/" "$remote:$remote_dir/$dir/"
    done
}

pull() {
    args=(-av)
    [[ "${weights:-0}" -eq 1 ]] || args+=(--exclude='*.pt')
    echo "<- $remote:$remote_dir/$results_dir/"
    rsync "${args[@]}" "$remote:$remote_dir/$results_dir/" "$results_dir/"
}

case "$mode" in
    push) push ;;
    pull) pull ;;
    both) push; pull ;;
    *) echo "usage: $0 [push|pull|both]" >&2; exit 1 ;;
esac
