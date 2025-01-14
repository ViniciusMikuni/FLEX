# load libs
module load pytorch

## you may also do the following if you want:
# module load conda
# conda activate your_env

MODEL=$1
SIZE=$2
BATCH_SIZE=$3


# for DDP
export MASTER_ADDR=$(hostname)

cmd="python train.py  --run-name ${MODEL}_v_${SIZE} --prediction-type v  --time-steps 2 --multi-node --batch-size $BATCH_SIZE  --model $MODEL --size $SIZE"
#cmd="python train.py  --run-name ${MODEL}_v_${SIZE} --prediction-type v  --time-steps 2 --multi-node --batch-size $BATCH_SIZE  --model $MODEL --size $SIZE --dataset climate"
#cmd="python train.py  --run-name ${MODEL}_v_${SIZE} --prediction-type v  --time-steps 2 --multi-node --batch-size $BATCH_SIZE  --model $MODEL --size $SIZE --dataset simple"

#cmd="python train.py  --run-name ${MODEL}_v_${SIZE}_sample --prediction-type v  --time-steps 2 --multi-node --batch-size $BATCH_SIZE  --model $MODEL --size $SIZE --sample_loss"


set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    " 
