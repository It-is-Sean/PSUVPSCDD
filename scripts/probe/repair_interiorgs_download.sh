#!/usr/bin/env bash

set -u

TARGET="${TARGET:-/home/JJ_Group/lih2511/datasets/InteriorGS}"
SCENES="${SCENES:-/home/JJ_Group/lih2511/datasets/InteriorGS_scenes.txt}"
LOG="${LOG:-/home/JJ_Group/lih2511/datasets/InteriorGS_repair_$(date +%Y%m%d_%H%M%S).log}"
FAILS="${FAILS:-/home/JJ_Group/lih2511/datasets/InteriorGS_repair_failed_$(date +%Y%m%d_%H%M%S).txt}"
HF="${HF:-/home/JJ_Group/lih2511/.conda/envs/nova3r/bin/hf}"
WORKERS="${WORKERS:-2}"
INNER_WORKERS="${INNER_WORKERS:-2}"
TRIES="${TRIES:-4}"
PER_SCENE_TIMEOUT="${PER_SCENE_TIMEOUT:-1800}"

export http_proxy="${http_proxy:-http://127.0.0.1:7896}"
export https_proxy="${https_proxy:-http://127.0.0.1:7896}"
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7896}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7896}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"

complete_scene() {
    local scene="$1"
    local dir="$TARGET/$scene"

    [ -s "$dir/3dgs_compressed.ply" ] &&
        [ -s "$dir/labels.json" ] &&
        [ -s "$dir/occupancy.json" ] &&
        [ -s "$dir/occupancy.png" ] &&
        [ -s "$dir/structure.json" ]
}

mkdir -p "$TARGET" "$(dirname "$LOG")" "$(dirname "$FAILS")"
: > "$FAILS"

echo "[$(date)] repair start target=$TARGET scenes=$SCENES" >> "$LOG"
echo "[$(date)] workers=$WORKERS inner_workers=$INNER_WORKERS tries=$TRIES timeout=${PER_SCENE_TIMEOUT}s" >> "$LOG"

total=$(wc -l < "$SCENES")

for wid in $(seq 0 $((WORKERS - 1))); do
    (
        i=0
        ok=0
        skip=0
        fail=0

        while IFS= read -r scene; do
            [ -n "$scene" ] || continue
            i=$((i + 1))
            shard=$(((i - 1) % WORKERS))
            [ "$shard" -eq "$wid" ] || continue

            if complete_scene "$scene"; then
                skip=$((skip + 1))
                echo "[$(date)] [w$wid] [$i/$total] skip $scene ok=$ok skip=$skip fail=$fail" >> "$LOG"
                continue
            fi

            done_scene=0
            attempt=1
            while [ "$attempt" -le "$TRIES" ]; do
                echo "[$(date)] [w$wid] [$i/$total] attempt $attempt start $scene" >> "$LOG"
                timeout "${PER_SCENE_TIMEOUT}s" "$HF" download spatialverse/InteriorGS \
                    "$scene/3dgs_compressed.ply" \
                    "$scene/labels.json" \
                    "$scene/occupancy.json" \
                    "$scene/occupancy.png" \
                    "$scene/structure.json" \
                    --repo-type dataset \
                    --local-dir "$TARGET" \
                    --max-workers "$INNER_WORKERS" \
                    --format quiet >> "$LOG" 2>&1
                rc=$?

                if complete_scene "$scene"; then
                    ok=$((ok + 1))
                    done_scene=1
                    echo "[$(date)] [w$wid] [$i/$total] done $scene attempt=$attempt rc=$rc ok=$ok skip=$skip fail=$fail" >> "$LOG"
                    break
                fi

                echo "[$(date)] [w$wid] [$i/$total] attempt $attempt incomplete $scene rc=$rc" >> "$LOG"
                sleep $((attempt * 20))
                attempt=$((attempt + 1))
            done

            if [ "$done_scene" -ne 1 ]; then
                fail=$((fail + 1))
                echo "$scene" >> "$FAILS"
                echo "[$(date)] [w$wid] [$i/$total] FAILED $scene ok=$ok skip=$skip fail=$fail" >> "$LOG"
            fi
        done < "$SCENES"

        echo "[$(date)] [w$wid] finished ok=$ok skip=$skip fail=$fail" >> "$LOG"
    ) &
done

wait
echo "[$(date)] repair finished" >> "$LOG"
