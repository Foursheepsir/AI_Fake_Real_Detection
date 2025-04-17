import pandas as pd
import re
import numpy as np
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
import evaluate

# Text cleaning function for original datasets
def clean_original_text(text):
    if pd.isna(text):
        return ""
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = text.replace('\xa0', ' ').replace('&nbsp;', ' ')
    if '-' in text:
        text = text.split('-', 1)[1]
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Text cleaning function for Neural News dataset
def clean_neural_text(text):
    if text is None:
        return ""  # Handle None cases safely
    text = text.replace('\xa0', ' ').replace('&nbsp;', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Process each split of the Neural News dataset
def process_split(split_name, dataset, label_map):
    processed_data = [
        {
            "label": label_map[item["label"]],
            "text": clean_neural_text(item["body"])
        }
        for item in dataset[split_name] if item["body"] is not None and item["language"] == "en"
    ]
    return processed_data


# Metrics computation function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    
    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "precision": precision_metric.compute(predictions=predictions, references=labels, average="weighted")["precision"],
        "recall": recall_metric.compute(predictions=predictions, references=labels, average="weighted")["recall"],
        "f1": f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    }

def main():

    LABEL_MAP = {
        "neural": 1,
        "real": 0
    }
    
    # Step 1: Load original datasets
    print("Loading and preprocessing datasets...")
    df_fake = pd.read_csv('fake.csv')
    df_true = pd.read_csv('true.csv')

    # Drop unnecessary columns
    df_fake.drop(labels=['title', 'subject', 'date'], axis=1, inplace=True)
    df_true.drop(labels=['title', 'subject', 'date'], axis=1, inplace=True)

    # Clean text in original datasets
    df_true['text'] = df_true['text'].apply(clean_original_text)
    df_fake['text'] = df_fake['text'].apply(clean_original_text)

    # Step 2: Load Neural News Benchmark dataset
    ds = load_dataset("tum-nlp/neural-news-benchmark")

    # Process all splits
    test_data = process_split('test', ds, LABEL_MAP)
    train_data = process_split('train', ds, LABEL_MAP)
    valid_data = process_split('validation', ds, LABEL_MAP)
    all_data = train_data + valid_data + test_data
    df = pd.DataFrame(all_data)

    # Extract neural and real data
    df_neural = df[df['label'] == 1].reset_index(drop=True)
    df_real = df[df['label'] == 0].reset_index(drop=True)

    # Drop label column for merging
    df_neural = df_neural.drop(columns=['label'])
    df_real = df_real.drop(columns=['label'])

    # Combine real news from both sources
    df_true = pd.concat([df_true, df_real], ignore_index=True)

    # Remove duplicates
    df_true = df_true.drop_duplicates(subset='text').reset_index(drop=True)
    df_fake = df_fake.drop_duplicates(subset='text').reset_index(drop=True)
    df_neural = df_neural.drop_duplicates(subset='text').reset_index(drop=True)

    # Filter by text length (between 750 and 5000 characters)
    df_true = df_true[df_true['text'].str.len() <= 5000].reset_index(drop=True)
    df_fake = df_fake[df_fake['text'].str.len() <= 5000].reset_index(drop=True)
    df_neural = df_neural[df_neural['text'].str.len() <= 5000].reset_index(drop=True)

    df_true = df_true[df_true['text'].str.len() >= 750].reset_index(drop=True)
    df_fake = df_fake[df_fake['text'].str.len() >= 750].reset_index(drop=True)
    df_neural = df_neural[df_neural['text'].str.len() >= 750].reset_index(drop=True)

    # Assign labels
    df_true['label'] = 0
    df_fake['label'] = 1
    df_neural['label'] = 2

    # Combine all datasets and shuffle
    df_all = pd.concat([df_true, df_fake, df_neural], ignore_index=True)
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Splitting data into train/validation/test sets...")
    train_val, test = train_test_split(df_all, test_size=0.2, random_state=42, stratify=df_all['label'])
    train, valid = train_test_split(train_val, test_size=0.125, random_state=42, stratify=train_val['label'])

    # Convert to dictionaries for dataset creation
    train_data = train.to_dict(orient='records')
    valid_data = valid.to_dict(orient='records')
    test_data = test.to_dict(orient='records')

    print("Preparing tokenized datasets...")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

    # Create datasets
    train_dataset = Dataset.from_list(train_data)
    valid_dataset = Dataset.from_list(valid_data)
    test_dataset = Dataset.from_list(test_data)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)
    # Apply tokenization
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_valid = valid_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)
    # Prepare data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("Configuring and training the model...")
    # Define label mapping for the model
    id2label = {0: "real", 1: "fake", 2: "neural"}
    label2id = {"real": 0, "fake": 1, "neural": 2}

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased", 
        num_labels=3, 
        id2label=id2label, 
        label2id=label2id
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="my_awesome_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to="none",  # Disable wandb logging
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting model training...")
    trainer.train()
    print("Training completed.")

    print("Evaluating on test set...")
    results = trainer.predict(tokenized_test)

    print("\n--- Test Set Performance ---")
    print(f"Accuracy:  {results.metrics['test_accuracy']:.4f}")
    print(f"Precision: {results.metrics['test_precision']:.4f}")
    print(f"Recall:    {results.metrics['test_recall']:.4f}")
    print(f"F1 Score:  {results.metrics['test_f1']:.4f}")

    preds = np.argmax(results.predictions, axis=1)
    labels = results.label_ids
    report = classification_report(labels, preds, output_dict=True, target_names=['real', 'fake', 'neural'])
    print("\nDetailed Classification Report:")
    print(classification_report(labels, preds, target_names=['real', 'fake', 'neural']))

    # Store class-specific metrics
    bert_scores = {
        'precision': [report['real']['precision'], report['fake']['precision'], report['neural']['precision']],
        'recall':    [report['real']['recall'],    report['fake']['recall'],    report['neural']['recall']],
        'f1':        [report['real']['f1-score'],  report['fake']['f1-score'],  report['neural']['f1-score']]
    }
    
    print("\nClass-specific metrics:")
    print(f"Precision: {bert_scores['precision']}")
    print(f"Recall: {bert_scores['recall']}")
    print(f"F1 Score: {bert_scores['f1']}")
    
    print("\nModel saved to 'my_awesome_model' directory")

if __name__ == "__main__":
    main()