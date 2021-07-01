LOG=$(date +%m%d_%H%M_logs)
echo $LOG
mkdir $LOG
python3 train.py --y_condition --output_dir $LOG \
                  --batch_size 16 \
                  --epochs 30 \
                  --dataroot "/home/arg/courses/machine_learning/homework/deep_learning_and_practice/Lab7/dataset/task_2/" \
                  --K 6 \
                  --L 3 \
                  # --saved_model 0614_2101_logs/glow_checkpoint_937.pt \

'''
--K 6 \ # Number of layers per block
--L 3 \ # Number of blocks
'''