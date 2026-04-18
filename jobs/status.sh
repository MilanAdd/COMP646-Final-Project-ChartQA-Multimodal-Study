#!/bin/bash
# Show status of a submitted ChartQA job.
#
# Usage:
#   ./status.sh train_frozen
#   ./status.sh train_lora
#   ./status.sh eval_zeroshot

source "$(dirname "$0")/config.sh"

JOB_NAME="${1:-train_frozen}"
JOB_FILE="$REMOTE_DIR/jobs/.${JOB_NAME}.jobid"

JOB_ID=$(ssh $REMOTE "cat $JOB_FILE 2>/dev/null")

if [ -z "$JOB_ID" ]; then
    echo "No active job found for: $JOB_NAME"
    exit 0
fi

INFO=$(ssh $REMOTE "
    STATUS=\$(squeue -j $JOB_ID -h -o '%T|%M|%N|%P|%l|%R' 2>/dev/null)

    if [ -z \"\$STATUS\" ]; then
        STATE=\$(sacct -j $JOB_ID --format=State,Elapsed,ExitCode -n -P 2>/dev/null | head -1)
        if [ -n \"\$STATE\" ]; then
            echo \"FINISHED|\$STATE\"
        else
            echo \"UNKNOWN\"
        fi
    else
        echo \"ACTIVE|\$STATUS\"
    fi

    # Tail latest log output
    OUTFILE=\$(ls $REMOTE_DIR/logs/${JOB_NAME}_${JOB_ID}.out 2>/dev/null)
    if [ -f \"\$OUTFILE\" ]; then
        echo '---OUTPUT---'
        tail -15 \"\$OUTFILE\"
    fi

    # Show any real errors (filter noise)
    ERRFILE=\$(ls $REMOTE_DIR/logs/${JOB_NAME}_${JOB_ID}.err 2>/dev/null)
    if [ -f \"\$ERRFILE\" ] && [ -s \"\$ERRFILE\" ]; then
        ERRS=\$(grep -v '^\[notice\]' \"\$ERRFILE\" | grep -v '^WARNING' | grep -v 'tqdm' | tail -5)
        if [ -n \"\$ERRS\" ]; then
            echo '---ERRORS---'
            echo \"\$ERRS\"
        fi
    fi
")

echo "Job: $JOB_NAME  (ID: $JOB_ID)"
echo ""

STATUS_LINE=$(echo "$INFO" | head -1)

if [[ "$STATUS_LINE" == ACTIVE* ]]; then
    IFS='|' read -r _ STATE ELAPSED NODE PARTITION TIMELIMIT REASON <<< "$STATUS_LINE"
    echo "Status:    $STATE"
    echo "Node:      $NODE"
    echo "Partition: $PARTITION"
    echo "Elapsed:   $ELAPSED / $TIMELIMIT"
    [ -n "$REASON" ] && echo "Reason:    $REASON"
elif [[ "$STATUS_LINE" == FINISHED* ]]; then
    IFS='|' read -r _ STATE ELAPSED EXITCODE <<< "$STATUS_LINE"
    echo "Status:    COMPLETED ($STATE)"
    echo "Elapsed:   $ELAPSED"
    echo "Exit code: $EXITCODE"
else
    echo "Status:    Unknown (job may have been cancelled or never submitted)"
fi

OUTPUT=$(echo "$INFO" | awk '/---ERRORS---/{found=0} found{print} /---OUTPUT---/{found=1}')
if [ -n "$OUTPUT" ]; then
    echo ""
    echo "Latest output:"
    echo "$OUTPUT"
fi

ERRORS=$(echo "$INFO" | awk 'found{print} /---ERRORS---/{found=1}')
if [ -n "$ERRORS" ]; then
    echo ""
    echo "Errors:"
    echo "$ERRORS"
fi
