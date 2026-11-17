#!/bin/bash
# podwatch.sh — dead-man's switch for RunPod pods.
#
# Two trips:
#   TTL  — POD_TTL_MIN  (default 360): hard lifetime cap, always applies.
#   IDLE — POD_IDLE_MIN (default 60):  stop when nothing is happening.
#
# "Busy" = established SSH session on :22, GPU util >15%, load1 >0.5, or
# handler/pip log writes in the last IDLE_MIN — so nohup'd experiments,
# model downloads, and handler traffic all count; a forgotten pod doesn't.
#
# Set either env var to 0 to disable that trip. On trip: runpodctl stop
# pod, falling back to poweroff. Logs to /workspace/podwatch.log.

TTL_MIN=${POD_TTL_MIN:-360}
IDLE_MIN=${POD_IDLE_MIN:-60}
START=$(date +%s)
TICK=0

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> /workspace/podwatch.log; }
log "watchdog up: ttl=${TTL_MIN}m idle=${IDLE_MIN}m"

while sleep 60; do
    NOW=$(date +%s)
    AGE=$(( (NOW - START) / 60 ))
    TICK=$((TICK + 1))

    if [ "$TTL_MIN" -gt 0 ] && [ "$AGE" -ge "$TTL_MIN" ]; then
        REASON="hard ttl ${TTL_MIN}m reached"
        break
    fi

    if [ "$IDLE_MIN" -le 0 ]; then
        continue
    fi

    SSH_N=$(ss -tnH state established '( sport = :22 )' 2>/dev/null | wc -l)
    GPU=$(nvidia-smi --query-gpu=utilization.gpu \
          --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    GPU=${GPU:-0}
    LOAD=$(cut -d' ' -f1 /proc/loadavg)
    NEWEST=$(stat -c %Y /workspace/server.log /workspace/pip.log \
             2>/dev/null | sort -n | tail -1)
    [ -z "$NEWEST" ] && NEWEST=$START
    LOG_AGE=$(( (NOW - NEWEST) / 60 ))

    BUSY=0
    [ "$SSH_N" -gt 0 ] && BUSY=1
    [ "$GPU" -gt 15 ] 2>/dev/null && BUSY=1
    awk "BEGIN{exit !($LOAD > 0.5)}" && BUSY=1
    [ "$LOG_AGE" -lt "$IDLE_MIN" ] && BUSY=1

    if [ $((TICK % 15)) -eq 0 ]; then
        log "check: age=${AGE}m ssh=$SSH_N gpu=${GPU}% load=$LOAD log_age=${LOG_AGE}m busy=$BUSY"
    fi

    if [ "$BUSY" -eq 0 ]; then
        REASON="idle ${IDLE_MIN}m (no ssh, gpu ${GPU}%, load $LOAD, no handler activity)"
        break
    fi
done

log "tripping: ${REASON:-unknown}"
runpodctl stop pod >/dev/null 2>&1 || poweroff -f || halt -f
