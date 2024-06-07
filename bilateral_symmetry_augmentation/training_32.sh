#!/bin/bash

initial_subset_size=80
increment=50
max_training_size=380

# no augmentation

for (( size=$initial_subset_size; size<=$max_training_size; size+=$increment )); do
    python train_unet_no_augmentation.py \
        --output_dir "unet_train/output_unet_enum32_16_no_augmentation_$size" \
        --train_dataset_dir "dentex_dataset/segmentation/enumeration32_train_val_test/train_$size" \
        --val_dataset_dir "dentex_dataset/segmentation/enumeration32_train_val_test/val" \
        --num_classes 32 --model seunet --batch_size 16
done

# default augmentation

for (( size=$initial_subset_size; size<=$max_training_size; size+=$increment )); do
    python train_unet_default_augmentation.py \
        --output_dir "unet_train/output_unet_enum32_16_default_augmentation_$size" \
        --train_dataset_dir "dentex_dataset/segmentation/enumeration32_train_val_test/train_$size" \
        --val_dataset_dir "dentex_dataset/segmentation/enumeration32_train_val_test/val" \
        --num_classes 32 --model seunet --batch_size 16
done

# my data augmentation

for (( size=$initial_subset_size; size<=$max_training_size; size+=$increment )); do
    python train_unet_no_augmentation.py \
        --output_dir "unet_train/output_unet_enum32_16_my_data_augmentation_$size" \
        --train_dataset_dir "dentex_dataset/segmentation/enumeration32_my_data_augmentation/train_$size" \
        --val_dataset_dir "dentex_dataset/segmentation/enumeration32_train_val_test/val" \
        --num_classes 32 --model seunet --batch_size 16
done