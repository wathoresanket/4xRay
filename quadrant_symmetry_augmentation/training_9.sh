#!/bin/bash

initial_subset_size=80
increment=50
max_training_size=380

# no quadrant flip

for (( size=$initial_subset_size; size<=$max_training_size; size+=$increment )); do
    python train_unet_no_augmentation.py \
        --output_dir "unet_train/output_unet_enum9_$size" \
        --train_dataset_dir "dentex_dataset/segmentation/enumeration9_train_val_test/train_$size" \
        --val_dataset_dir "dentex_dataset/segmentation/enumeration9_train_val_test/val" \
        --num_classes 9 --model seunet --batch_size 32 --epochs 100
done

# with quadrant flip

for (( size=$initial_subset_size; size<=$max_training_size; size+=$increment )); do
    python train_unet_no_augmentation.py \
        --output_dir "unet_train/output_unet_enum9_quadrant_flip_$size" \
        --train_dataset_dir "dentex_dataset/segmentation/enumeration9_quadrant_flip/train_$size" \
        --val_dataset_dir "dentex_dataset/segmentation/enumeration9_quadrant_flip/val" \
        --num_classes 9 --model seunet --batch_size 32 --epochs 100
done

