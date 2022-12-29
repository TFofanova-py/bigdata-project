import os
import pandas as pd
import numpy as np
import torch
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import wandb


class TokenizedTextsDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, label2id):
        super().__init__()
        self.tokenizer = tokenizer
        self.texts = df.text
        self.labels = df.writer.apply(lambda s: label2id[s])

    def __getitem__(self, idx):
        examples = self.tokenizer(self.texts[idx].lower(),
                                  truncation=True, padding="max_length", max_length=512)
        examples["label"] = torch.tensor(self.labels[idx])
        return examples

    def __len__(self):
        return len(self.labels)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1_macro = f1.compute(predictions=predictions, references=labels, average='macro')
    return f1_macro


if __name__ == "__main__":

    # Training and test data
    ds_train = pd.read_csv("train_data.csv", usecols=[0, 2])
    ds_test = pd.read_csv("test_data.csv", usecols=[0, 2])

    wr2ids = {'Paustovskiy': 0, 'Ostrovsky': 1, 'Gogol': 2, 'Dostoevsky': 3, 'Leskov': 4, 'Belyaev': 5, 'Kataev': 6,
              'Solzhenitsin': 7, 'Akunin': 8, 'Struhgatskie': 9, 'Lukyanenko': 10, 'Bulgakov': 11, 'Kazantsev': 12,
              'Dovlatov': 13, 'Goncharov': 14, 'Pikul': 15, 'Gorky': 16, 'Grin': 17, 'Chekhov': 18, 'Fray': 19,
              'Sergeev-Thsenskiy': 20, 'Pelevin': 21, 'Ilf_petrov': 22, 'Gaydar': 23, 'Serafimovich': 24, 'Prishvin': 25,
              'Kuprin': 26, 'Fadeev': 27, 'Averchenko': 28, 'Zoschenko': 29, 'Furmanov': 30, 'Saltykov-schedrin': 31,
              'Shukshin': 32, 'Pasternak': 33}

    ids2wr = {v: k for k, v in wr2ids.items()}

    # Training model
    huggingface_name = "sberbank-ai/ruBert-base"

    tokenizer = AutoTokenizer.from_pretrained(huggingface_name)
    model = AutoModelForSequenceClassification.from_pretrained(huggingface_name,
                                                               return_dict=True, num_labels=34,
                                                               ignore_mismatched_sizes=True)
    model.config.id2label = ids2wr
    model.config.label2id = wr2ids

    # Training logging
    wandb.login()
    wandb.init(project="hw-nlp")
    wandb.watch(model)

    # Training parameters
    batch_size = 6
    num_train_epochs = 2

    f1 = evaluate.load("f1")

    train_dataset = TokenizedTextsDataset(ds_train, tokenizer, model.config.label2id)
    test_dataset = TokenizedTextsDataset(ds_test, tokenizer, model.config.label2id)

    training_args = TrainingArguments(
        output_dir="test_trainer",
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        report_to=None,
        num_train_epochs=num_train_epochs,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer.save_model("big_model")
    os.system("wandb artifact put -t model big_model")
