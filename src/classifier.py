"""Fine-tune CAMeLBERT for complaint topic classification (Lab 3).

Heavy libraries are imported inside the functions so the rest of the project
works without them.
"""
MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-mix"


def train_classifier(train_df, test_df, topics, epochs=4, batch_size=2,
                     learning_rate=2e-5, output_dir="./results"):
    import numpy as np
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def prepare(data):
        tokens = tokenizer(data["text"].tolist(), padding=True, truncation=True)
        tokens["labels"] = data["label"].tolist()
        return Dataset.from_dict(tokens)

    train_ds, test_ds = prepare(train_df), prepare(test_df)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(topics),
        id2label=dict(enumerate(topics)),
        label2id={t: i for i, t in enumerate(topics)},
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
        }

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=test_ds, compute_metrics=compute_metrics)
    trainer.train()
    metrics = trainer.evaluate()
    return trainer, tokenizer, metrics
