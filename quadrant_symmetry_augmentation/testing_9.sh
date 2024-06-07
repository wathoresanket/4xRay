#!/bin/bash

initial_subset_size=80
increment=50
max_training_size=380

# no quadrant flip

for (( size=$initial_subset_size; size<=$max_training_size; size+=$increment )); do
    output_dir="unet_test/output_unet_enum9_$size"
    model_path=$(ls -t unet_train/output_unet_enum9_$size/*.pth | head -2 | tail -1)
    python test_unet.py \
        --output_dir "$output_dir" \
        --dataset_dir dentex_dataset/segmentation/enumeration9_train_val_test/test \
        --model_path "$model_path" \
        --num_classes 9 --model seunet --batch_size 32
done

# with quadrant flip

for (( size=$initial_subset_size; size<=$max_training_size; size+=$increment )); do
    output_dir="unet_test/output_unet_enum9_quadrant_flip_$size"
    model_path=$(ls -t unet_train/output_unet_enum9_quadrant_flip_$size/*.pth | head -2 | tail -1)
    python test_unet.py \
        --output_dir "$output_dir" \
        --dataset_dir dentex_dataset/segmentation/enumeration9_quadrant_flip/test \
        --model_path "$model_path" \
        --num_classes 9 --model seunet --batch_size 32
done