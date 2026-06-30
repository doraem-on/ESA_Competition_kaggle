# Neural Debris Removal - ESA Kaggle Competition

This repository contains the baseline solution for the ESA "Neural Debris Removal" competition on Kaggle. The task is to de-poison (unlearn) a RetinaNet model trained for space debris streak detection, using an unlearn set of poisoned images.

## Project Structure

- `unlearn_baseline.py`: The baseline fine-tuning PyTorch script. It uses the `unlearn_set` with empty annotations to effectively unlearn the poisoned objects.
- `generate_submission.py`: The inference script that loads the unlearned model, processes the `test_set` (images 0-1999), formats the predictions into the required `confidence x y width height` structure, and outputs a CSV file.
- `probe_model.py`: Utility script to inspect the provided PyTorch model checkpoint.
- `download_data.py`: Script to download the competition data using `kagglehub`.

## How to Run

1. **Download the Data**: Run `python download_data.py` (ensure you have set up your Kaggle API credentials). Ensure the dataset is located in the `data/` directory.
2. **Train the De-poisoned Model**:
   ```bash
   python unlearn_baseline.py
   ```
   This will train the model for 20 epochs using the Adam optimizer (lr=0.0001) and output the de-poisoned weights to `data/depoisoned_model.pth`.
3. **Generate Submission**:
   ```bash
   python generate_submission.py
   ```
   This will generate `submission.csv` containing your test set predictions.

## Submission
Upload the generated `submission.csv` file directly to Kaggle.
