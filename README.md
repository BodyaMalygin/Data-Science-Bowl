# Data Science Bowl

Build a semantic segmentation model with UNet architecture using Keras.

## Prerequisites

- Windows 10 or OS Linux
- Python 3.8

## Installation

```bash
pip install -r requiriments.txt
```

## Usage

```python
python3 train.py # fit motel
python3 predict_masks.py # examples predict
```

## Describe
To solve the problem, data analysis was carried out. Therefore, the size of all pictures was changed to 256X256. 

A model with UТet architecture was built to solve semantic segmentation. Dice Score was used as a metric.


The model studied for 15 epochs. And the Dice score on the validation data was 0.868. The validation data was the last 10% of the training data.

