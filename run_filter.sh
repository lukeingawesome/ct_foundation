#!/bin/bash

# Filter corrupted NPZ files from the dataset
echo "Filtering corrupted NPZ files from dataset..."

python3 filter_corrupted_data.py \
    --input_csv llm/all_ct_with_labels_with_answer.csv \
    --output_csv llm/all_ct_with_labels_with_answer_cleaned.csv \
    --img_key img_path

echo "Filtering complete!"
echo "Original file: llm/all_ct_with_labels_with_answer.csv"
echo "Cleaned file: llm/all_ct_with_labels_with_answer_cleaned.csv"
echo ""
echo "You can now update your stage2.sh script to use the cleaned CSV file." 