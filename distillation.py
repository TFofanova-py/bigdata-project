import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel
from transformers import TrainingArguments, Trainer
import evaluate


class TokenizedTextsDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, label2id):
        super().__init__()
        self.tokenizer = tokenizer
        self.texts = df.text
        self.labels = df.writer.apply(lambda s: label2id[s])

    def __getitem__(self, idx):
        examples = self.tokenizer(self.texts[idx].lower(),
                                  truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        examples["label"] = torch.tensor(self.labels[idx])
        return {k: v.squeeze() for k, v in examples.items()}

    def __len__(self):
        return len(self.labels)


class DistillTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, temperature=5, alpha=0.5):
        celoss = nn.CrossEntropyLoss()
        kldloss = nn.KLDivLoss()

        teacher_logits = big_model(**inputs).logits
        student_logits = model(**inputs).logits
        labels = inputs.get("labels")

        soft_predictions = F.softmax(student_logits / temperature, dim=1)
        soft_labels = F.softmax(teacher_logits / temperature, dim=1)

        distillation_loss = kldloss(soft_predictions, soft_labels)
        student_loss = celoss(student_logits, labels)

        loss = alpha * distillation_loss + (1 - alpha) * student_loss
        return (loss, {"outputs": student_logits}) if return_outputs else loss


def distill_bert_weights(
        teacher: nn.Module,
        student: nn.Module,
) -> None:
    """
    Recursively copies the weights of the (teacher) to the (student).
    This function is meant to be first called on a BertFor... model, but is then called on every children of that model recursively.
    The only part that's not fully copied is the encoder, of which only half is copied.
    """
    # If the part is an entire BERT model or a BertFor..., unpack and iterate
    if isinstance(teacher, BertModel) or type(teacher).__name__.startswith('BertFor'):
        for teacher_part, student_part in zip(teacher.children(), student.children()):
            distill_bert_weights(teacher_part, student_part)
    # Else if the part is an encoder, copy one out of every layer
    elif isinstance(teacher, BertEncoder):
        teacher_encoding_layers = [layer for layer in next(teacher.children())]
        student_encoding_layers = [layer for layer in next(student.children())]
        for i in range(len(student_encoding_layers)):
            student_encoding_layers[i].load_state_dict(teacher_encoding_layers[2 * i].state_dict())
    # Else the part is a head or something else, copy the state_dict
    else:
        student.load_state_dict(teacher.state_dict())


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1_macro = f1.compute(predictions=predictions, references=labels, average='macro')
    return f1_macro


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # download the big model
    run = wandb.init()
    artifact = run.use_artifact('sava_ml/hw-nlp/model_34_baseline:v1', type='model')
    artifact_dir = artifact.download()

    big_tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/ruBert-base")
    big_model = AutoModelForSequenceClassification.from_pretrained("artifacts/model_34_baseline:v1",
                                                                   return_dict=True, num_labels=34,
                                                                   ignore_mismatched_sizes=True).to(device)

    # training and test data
    ds_train = pd.read_csv("../input/authorstexts/train_data.csv", usecols=[0, 2])
    ds_test = pd.read_csv("../input/authorstexts/test_data.csv", usecols=[0, 2])

    BATCH_SIZE = 6

    # datasets
    train_dataset = TokenizedTextsDataset(ds_train, big_tokenizer, label2id=big_model.config.label2id)
    test_dataset = TokenizedTextsDataset(ds_test, big_tokenizer, label2id=big_model.config.label2id)

    # dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=2 * BATCH_SIZE, drop_last=True)

    # configure small model
    configuration = big_model.config.to_dict()
    configuration['num_hidden_layers'] //= 2
    configuration = BertConfig.from_dict(configuration)

    small_model = type(big_model)(configuration)

    # Initialize the student's weights
    distill_bert_weights(teacher=big_model, student=small_model)

    training_args = TrainingArguments(
        output_dir="test_trainer",
        evaluation_strategy="steps",
        eval_steps=5000,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        report_to=None,
        num_train_epochs=2,
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        max_steps=15000)

    f1 = evaluate.load("f1")

    trainer = DistillTrainer(
        model=small_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer.save_model("distilled_model")

    os.system("wandb artifact put -n distilled -t model distilled_model")
