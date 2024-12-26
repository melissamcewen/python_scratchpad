# Import necessary libraries
import json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import torch
import re
import pickle
from pathlib import Path

# Load your JSONL data
data_list = []

# Open and read the JSONL file line by line
with open("/kaggle/input/open-food-facts-ocr-text-jsonl/ocr-text-latest.jsonl", 'r') as file:  # You'll need to upload this file to Colab
    for i, line in enumerate(file):
        data_list.append(json.loads(line))

        # Stop after loading a specific number of lines for testing
        if i >= 10000:  # Adjust this number based on memory capacity
            break

# Convert to DataFrame
data = pd.DataFrame(data_list)

# Basic preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove excessive whitespace
        text = ' '.join(text.split())
        return text
    return ''

def extract_ingredients_section(text):
    """
    Extract ingredients section using common patterns across languages
    """
    # Combined pattern for multiple languages
    pattern = r'(ingredients|ingrédients|zutaten|ingredientes|ingrediënten|ingredienti):(.*?)(?=\n\n|\Z)'

    matches = re.findall(pattern, text.lower(), re.IGNORECASE | re.DOTALL)
    if matches:
        return matches[0][1].strip(), True  # Return the captured ingredients text
    return text, False

# Process the dataset
processed_data = []
for _, row in data.iterrows():
    result = preprocess_text(row['text'])
    processed = extract_ingredients_section(result)
    processed_data.append({
        'source': row['source'],
        'text': row['text'],
        'processed_text': processed[0],
        'is_ingredients': processed[1]
    })

processed_df = pd.DataFrame(processed_data)

# Print some statistics
print(f"Total samples: {len(processed_df)}")
print(f"Identified ingredient sections: {processed_df['is_ingredients'].sum()}")
print("\nSample identified ingredients:")
print(processed_df[processed_df['is_ingredients']]['processed_text'].head())

# Create train/validation split
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(processed_df, test_size=0.2, random_state=42)

# Take a small subset for initial testing
train_df_small = train_df.sample(n=1000, random_state=42)
val_df_small = val_df.sample(n=200, random_state=42)

# First evaluate regex performance
total_samples = len(processed_df)
ingredients_found = processed_df['is_ingredients'].sum()
print("\nRegex Performance:")
print(f"Total samples: {total_samples}")
print(f"Ingredients sections found: {ingredients_found}")
print(f"Percentage: {(ingredients_found/total_samples)*100:.2f}%")

# Now set up model training
from transformers import TrainingArguments, Trainer

# Initialize tokenizer
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples['processed_text'],
        padding='max_length',
        truncation=True,
        max_length=256,
        return_tensors=None
    )

    labels = [1 if label else 0 for label in examples['is_ingredients']]
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

# Convert to HuggingFace datasets
train_dataset = Dataset.from_pandas(train_df_small)
val_dataset = Dataset.from_pandas(val_df_small)

# Apply tokenization
tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_val = val_dataset.map(tokenize_and_align_labels, batched=True)

# Remove unnecessary columns
columns_to_remove = ['processed_text', 'text', 'source', 'is_ingredients']
tokenized_train = tokenized_train.remove_columns(columns_to_remove)
tokenized_val = tokenized_val.remove_columns(columns_to_remove)

# Set format for pytorch
tokenized_train.set_format('torch')
tokenized_val.set_format('torch')

# Clear GPU memory if needed
import torch
torch.cuda.empty_cache()

# More verbose training arguments
training_args = TrainingArguments(
    output_dir="./ingredients-classifier",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    load_best_model_at_end=True,
    logging_dir='./logs',
    logging_steps=10,
    report_to="tensorboard",
    disable_tqdm=False,
    log_level="info"
)

# Add class weights to the model initialization
class_weights = torch.tensor([1.0, 2.0])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_weights = class_weights.to(device)

# Move model to GPU
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    problem_type="single_label_classification",
)
model = model.to(device)

# Custom trainer class to handle class weights
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Initialize trainer with compute_metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Then initialize the trainer
trainer = WeightedTrainer(
    class_weights=class_weights,
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)

# Print GPU info
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device:", torch.cuda.get_device_name(0))

# Print dataset info
print("\nDataset sizes:")
print(f"Training samples: {len(tokenized_train)}")
print(f"Validation samples: {len(tokenized_val)}")

# Print sample of data
print("\nSample of training data:")
print(next(iter(trainer.get_train_dataloader())))

print("\nStarting training...")
trainer.train()

# 1. Evaluate the model on validation set
print("\nEvaluating model on validation set...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# 2. Create a function to make predictions on new text
def predict_ingredients_section(text, model, tokenizer, confidence_threshold=0.6):
    """
    Predict whether a given text contains an ingredients list
    """
    # Preprocess the text
    processed_text = preprocess_text(text)

    # Tokenize
    inputs = tokenizer(
        processed_text,
        padding='max_length',
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    # Move inputs to GPU if available
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # Get probabilities for both classes
        probs = predictions[0].cpu().tolist()  # Move back to CPU for processing
        # Class 1 (ingredients) probability
        ingredients_prob = probs[1]
        is_ingredients = ingredients_prob > confidence_threshold

    return {
        'is_ingredients': bool(is_ingredients),
        'confidence': ingredients_prob,
        'processed_text': processed_text
    }

# 3. Test the model on some example texts
test_texts = [
    "Ingredients: Water, Sugar, Salt",
    "Product of USA. Best before 2024",
    "INGREDIENTS: Wheat flour, vegetable oil, salt.",
    "Chocolate, sugar, milk, cocoa butter, soy lecithin, vanilla extract"
]

print("\nTesting model on example texts:")
for text in test_texts:
    result = predict_ingredients_section(text, model, tokenizer)
    print(f"\nText: {text}")
    print(f"Is ingredients section: {result['is_ingredients']}")
    print(f"Confidence: {result['confidence']:.2%}")

# 4. Save the model for later use
output_dir = "./ingredients-classifier-final"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"\nModel saved to {output_dir}")

# After training, save both model and training metrics
import json
from pathlib import Path

def save_training_state(model, tokenizer, eval_results, output_dir="model_output"):
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save evaluation results
    with open(f"{output_dir}/eval_results.json", 'w') as f:
        json.dump(eval_results, f)

    # Save model configuration
    config = {
        "model_name": model_name,
        "max_length": 256,
        "num_labels": 2,
        "confidence_threshold": 0.6
    }
    with open(f"{output_dir}/config.json", 'w') as f:
        json.dump(config, f)

    print(f"Model and configuration saved to {output_dir}")

# Function to load everything back
def load_training_state(model_dir="model_output"):
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Load configuration
    with open(f"{model_dir}/config.json", 'r') as f:
        config = json.load(f)

    # Load evaluation results
    with open(f"{model_dir}/eval_results.json", 'r') as f:
        eval_results = json.load(f)

    return model, tokenizer, config, eval_results

# Save everything after training
save_training_state(model, tokenizer, eval_results)

def clean_ingredients_set(ingredients_set):
    """Clean the ingredients set to remove problematic entries"""
    cleaned = set()
    # Words that shouldn't be counted as ingredients
    skip_words = {
        'contains', 'may contain', 'produced in', 'ingredients',
        'allergens', 'warning', 'manufactured', 'processed',
        'and/or', 'added as', 'between', 'per', 'or'
    }

    for ingredient in ingredients_set:
        # Skip if empty or too long
        if not ingredient or len(ingredient) > 50:
            continue

        # Clean the ingredient
        cleaned_ingredient = ingredient.strip().lower()
        # Remove parenthetical content
        cleaned_ingredient = re.sub(r'\([^)]*\)', '', cleaned_ingredient)
        # Remove trailing punctuation and extra spaces
        cleaned_ingredient = re.sub(r'[.,;:]$', '', cleaned_ingredient)
        cleaned_ingredient = re.sub(r'\s+', ' ', cleaned_ingredient).strip()

        # Skip if it contains skip words, is too short, or contains numbers
        if (any(skip in cleaned_ingredient for skip in skip_words) or
            len(cleaned_ingredient) < 2 or
            any(char.isdigit() for char in cleaned_ingredient) or
            'and' in cleaned_ingredient.split() or
            'with' in cleaned_ingredient.split()):
            continue

        cleaned.add(cleaned_ingredient)

    # Remove ingredients that are substrings of other ingredients
    final_ingredients = set()
    sorted_ingredients = sorted(cleaned, key=len, reverse=True)

    for ingredient in sorted_ingredients:
        # Only add if it's not a substring of any longer ingredient
        if not any(
            ingredient != other and
            ingredient in other.split() # Only match whole words
            for other in final_ingredients
        ):
            final_ingredients.add(ingredient)

    return final_ingredients

def count_known_ingredients(text, ingredients_set):
    """Count how many known ingredients appear in the text"""
    matched = set()
    text = text.lower()

    # Remove the "ingredients:" prefix if present
    text = re.sub(r'^ingredients:?\s*', '', text)

    # Split text into potential ingredients and clean them
    text_ingredients = []
    for i in re.split(r'[,;]', text):
        cleaned = i.strip()
        # Remove parenthetical content
        cleaned = re.sub(r'\([^)]*\)', '', cleaned)
        # Remove trailing punctuation
        cleaned = re.sub(r'[.,;:]$', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if cleaned:
            text_ingredients.append(cleaned)

    # Try to match each text ingredient
    for text_ingredient in text_ingredients:
        # First try exact match
        if text_ingredient in ingredients_set:
            matched.add(text_ingredient)
            continue

        # Then try matching individual words for compound ingredients
        words = text_ingredient.split()
        for ingredient in ingredients_set:
            if (ingredient in text_ingredient and
                all(word in text_ingredient for word in ingredient.split())):
                matched.add(ingredient)
                break

    return len(matched), matched

def analyze_food_text(text, model, tokenizer, clean_ingredients, confidence_threshold=0.6):
    """
    Complete analysis of food text:
    1. Check if it's an ingredients list
    2. If yes, identify specific ingredients
    """
    # First check if it's an ingredients list
    model_result = predict_ingredients_section(text, model, tokenizer, confidence_threshold)

    # If it is, find the specific ingredients
    if model_result['is_ingredients']:
        count, matched = count_known_ingredients(text, clean_ingredients)
        return {
            'is_ingredients': True,
            'confidence': model_result['confidence'],
            'ingredients_found': count,
            'ingredients': sorted(list(matched))
        }

    return {
        'is_ingredients': False,
        'confidence': model_result['confidence'],
        'ingredients_found': 0,
        'ingredients': []
    }

# Clean the ingredients set again
cleaned_ingredients = clean_ingredients_set(ingredients_set)
print(f"\nCleaned ingredients set from {len(ingredients_set)} to {len(cleaned_ingredients)} ingredients")
print("\nSample cleaned ingredients:")
print(sorted(list(cleaned_ingredients))[:10])  # Sort for better readability

# Test the improved matching
test_texts = [
    "Ingredients: Water, Sugar, Salt",
    "Product of USA. Best before 2024",
    "INGREDIENTS: Wheat flour, vegetable oil, salt.",
    "Chocolate, sugar, milk, cocoa butter, soy lecithin, vanilla extract"
]

print("\nTesting improved ingredient matching:")
for text in test_texts:
    count, matched = count_known_ingredients(text, cleaned_ingredients)
    print(f"\nText: {text}")
    print(f"Known ingredients found: {count}")
    print(f"Matched ingredients: {', '.join(sorted(matched))}")  # Sort for readability

def parse_ingredients_taxonomy(file_path):
    """Parse the Open Food Facts ingredients taxonomy"""
    ingredients = {}
    current_parents = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Handle parent reference
            if line.startswith('<'):
                parent = line[1:].strip()  # Remove '<' and whitespace
                current_parents.append(parent)
                continue

            # Handle language entries
            if ':' in line:
                lang, names = line.split(':', 1)
                lang = lang.strip()
                names = [n.strip() for n in names.split(',')]

                # Only process English names for now
                if lang == 'en':
                    for name in names:
                        if name:
                            ingredients[name] = {
                                'parents': current_parents.copy(),
                                'variations': names
                            }

            # Reset parents if we hit a blank line or wikidata entry
            if line.startswith('wikidata:'):
                current_parents = []

    return ingredients

# After training, load and process the taxonomy
ingredients_dict = parse_ingredients_taxonomy('/kaggle/input/open-food-facts-multilingual-ingredients/ingredients.txt')

# Create clean ingredients set from taxonomy
clean_ingredients = set()
for ingredient, data in ingredients_dict.items():
    # Remove any text in parentheses and clean up
    clean_name = re.sub(r'\([^)]*\)', '', ingredient).strip().lower()
    if len(clean_name) > 1:  # Skip single characters
        clean_ingredients.add(clean_name)

print(f"\nExtracted {len(clean_ingredients)} clean ingredients")

# Test the combined system
test_texts = [
    "Ingredients: Water, Sugar, Salt",
    "Product of USA. Best before 2024",
    "INGREDIENTS: Wheat flour, vegetable oil, salt.",
    "Chocolate, sugar, milk, cocoa butter, soy lecithin, vanilla extract"
]

print("\nTesting complete analysis system:")
for text in test_texts:
    result = analyze_food_text(text, model, tokenizer, clean_ingredients)
    print(f"\nText: {text}")
    print(f"Is ingredients list: {result['is_ingredients']} (confidence: {result['confidence']:.2%})")
    if result['is_ingredients']:
        print(f"Found {result['ingredients_found']} ingredients:")
        print(f"Ingredients: {', '.join(result['ingredients'])}")
