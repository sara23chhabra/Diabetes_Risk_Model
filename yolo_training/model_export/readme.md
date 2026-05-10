# YOLOv8 Food Detection Model

This repository contains the trained YOLOv8 food detection model and prediction logic.

## Setup

Install dependencies:

pip install -r requirements.txt

## Usage

from food_detector import detect_food

results = detect_food("image.jpg")

print(results)

## Output Format

[
  {
    "food": "rice",
    "confidence": 0.91
  }
]