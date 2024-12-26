# Ingredients List Detector and Analyzer

This project combines machine learning and rule-based approaches to detect and analyze ingredients lists in text.

## Features

- Detects ingredients lists using ML (XLM-RoBERTa model)
- Extracts individual ingredients
- Matches against Open Food Facts ingredients taxonomy
- Supports multiple languages
- Provides confidence scores

## Required Data

1. Open Food Facts OCR text JSONL file
2. Open Food Facts multilingual ingredients taxonomy

## How It Works

The script (`kaggle.py`) works in three stages:

1. **ML Model Training**
   - Trains on 1000 samples
   - Uses XLM-RoBERTa base model
   - Handles multiple languages
   - Returns confidence scores

2. **Taxonomy Processing**
   - Parses ingredients taxonomy
   - Creates standardized ingredient list
   - Maintains ingredient hierarchies

3. **Text Analysis**
   - Identifies ingredients lists
   - Extracts ingredients
   - Matches against known ingredients
   - Provides confidence scores

## Dependencies

- transformers
- torch
- pandas
- numpy
- sklearn
- re
- json
- pickle
- pathlib

## Usage Example

```python
text = "Ingredients: Water, Sugar, Salt"
result = analyze_food_text(text, model, tokenizer, clean_ingredients)
```
```python
{
'is_ingredients': True,
'confidence': 0.945,
'ingredients_found': 3,
'ingredients': ['water', 'sugar', 'salt']
}
```
