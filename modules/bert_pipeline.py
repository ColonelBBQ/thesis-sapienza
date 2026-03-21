from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer


DEFAULT_TARGET_LIST = [
    "compute",
    "data_handling",
    "network",
    "security_compliance",
    "management_monitoring",
    "cloud_service_essentials",
]

DEFAULT_DROP_COLUMNS = ["source", "goal", "ambiguity_type", "ambuiguity", "comments"]


@dataclass(frozen=True)
class BertExperimentConfig:
    max_len: int = 300
    train_batch_size: int = 16
    valid_batch_size: int = 16
    epochs: int = 5
    learning_rate: float = 3e-5
    weight_decay: float = 1e-6
    test_size: float = 0.15
    seed: int = 42
    tokenizer_name: str = "bert-base-uncased"
    do_lower_case: bool = True


class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizer, max_len: int, target_list: list[str]):
        self.tokenizer = tokenizer
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.titles = self.df["description"].astype(str).apply(lambda title: " ".join(title.split())).tolist()
        self.targets = torch.tensor(self.df[target_list].values, dtype=torch.float)

        # Pre-tokenize the whole split once so each training step only indexes tensors.
        self.encodings = self.tokenizer(
            self.titles,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.titles)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][index],
            "attention_mask": self.encodings["attention_mask"][index],
            "token_type_ids": self.encodings["token_type_ids"][index],
            "targets": self.targets[index],
        }


class BERTClass(torch.nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.bert_model = BertModel.from_pretrained("bert-base-uncased", return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, num_labels)

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        output = self.bert_model(
            input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
        )
        output_dropout = self.dropout(output.pooler_output)
        return self.linear(output_dropout)


def get_tokenizer(config: BertExperimentConfig) -> BertTokenizer:
    return BertTokenizer.from_pretrained(config.tokenizer_name, do_lower_case=config.do_lower_case)


def prepare_baseline_dataframe(
    df: pd.DataFrame,
    target_list: list[str] | None = None,
    drop_columns: list[str] | None = None,
) -> pd.DataFrame:
    target_list = target_list or DEFAULT_TARGET_LIST
    drop_columns = drop_columns or DEFAULT_DROP_COLUMNS

    baseline_df = df.drop(columns=drop_columns, errors="ignore").reset_index(drop=True).copy()
    baseline_df[target_list] = baseline_df[target_list].astype(int)
    return baseline_df


def filter_subset(
    df: pd.DataFrame,
    mask: pd.Series | None = None,
    query: str | None = None,
    subset_name: str = "subset",
) -> pd.DataFrame:
    subset_df = df
    if query is not None:
        subset_df = subset_df.query(query)
    if mask is not None:
        subset_df = subset_df.loc[mask]

    subset_df = subset_df.reset_index(drop=True).copy()
    print(f"{subset_name}: {len(subset_df)} rows")
    return subset_df


def split_dataframe(
    df: pd.DataFrame,
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def compute_distribution_classes(
    train_set: pd.DataFrame,
    val_set: pd.DataFrame,
    class_columns: list[str],
) -> tuple[dict[str, int], dict[str, int]]:
    train_class_distribution = train_set[class_columns].sum()
    val_class_distribution = val_set[class_columns].sum()
    return train_class_distribution.to_dict(), val_class_distribution.to_dict()


def build_dataloaders(
    df: pd.DataFrame,
    tokenizer: BertTokenizer,
    config: BertExperimentConfig,
    target_list: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, DataLoader, DataLoader]:
    target_list = target_list or DEFAULT_TARGET_LIST
    train_df, val_df = split_dataframe(df, test_size=config.test_size, seed=config.seed)

    training_set = CustomDataset(train_df, tokenizer, config.max_len, target_list)
    validation_set = CustomDataset(val_df, tokenizer, config.max_len, target_list)

    train_loader = DataLoader(
        training_set,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        validation_set,
        batch_size=config.valid_batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_df, val_df, train_loader, val_loader


def build_model_and_optimizer(
    config: BertExperimentConfig,
    num_labels: int,
    device: torch.device,
) -> tuple[BERTClass, torch.optim.Optimizer]:
    model = BERTClass(num_labels=num_labels)
    model.to(device)
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    return model, optimizer


def loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def train_model(
    n_epochs: int,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_model_path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, list[float], list[float], list[int]]:
    training_losses: list[float] = []
    validation_losses: list[float] = []
    epochs_list: list[int] = []
    valid_loss_min = np.inf

    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        print(f"############# Epoch {epoch}: Training Start #############")

        train_iterator = tqdm(training_loader, desc=f"Epoch {epoch} [train]", leave=False)
        for data in train_iterator:
            ids = data["input_ids"].to(device, dtype=torch.long)
            mask = data["attention_mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_iterator.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(training_loader)
        print(f"############# Epoch {epoch}: Training End #############")

        model.eval()
        print(f"############# Epoch {epoch}: Validation Start #############")

        with torch.no_grad():
            validation_iterator = tqdm(validation_loader, desc=f"Epoch {epoch} [valid]", leave=False)
            for data in validation_iterator:
                ids = data["input_ids"].to(device, dtype=torch.long)
                mask = data["attention_mask"].to(device, dtype=torch.long)
                token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
                targets = data["targets"].to(device, dtype=torch.float)

                outputs = model(ids, mask, token_type_ids)
                loss = loss_fn(outputs, targets)
                valid_loss += loss.item()
                validation_iterator.set_postfix(loss=f"{loss.item():.4f}")

        valid_loss /= len(validation_loader)
        print(f"############# Epoch {epoch}: Validation End #############")
        print(
            "Epoch: {} \tAverage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}".format(
                epoch, train_loss, valid_loss
            )
        )

        training_losses.append(train_loss)
        validation_losses.append(valid_loss)
        epochs_list.append(epoch)

        if valid_loss <= valid_loss_min:
            print(f"Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving model ...")
            torch.save(model.state_dict(), best_model_path)
            valid_loss_min = valid_loss
        else:
            print("Early stopping")
            break

        print(f"############# Epoch {epoch} Done #############\n")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model, training_losses, validation_losses, epochs_list


def validate_multilabel(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[list[list[int]], list[list[float]]]:
    model.eval()
    all_targets: list[list[int]] = []
    all_probabilities: list[list[float]] = []

    with torch.no_grad():
        for data in data_loader:
            ids = data["input_ids"].to(device, dtype=torch.long)
            mask = data["attention_mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids)
            probabilities = torch.sigmoid(outputs)

            all_targets.extend(targets.int().cpu().detach().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().detach().numpy().tolist())

    return all_targets, all_probabilities


def test_model(
    sentence: str,
    model: torch.nn.Module,
    tokenizer: BertTokenizer,
    max_len: int,
    threshold: float,
    device: torch.device,
    target_list: list[str],
) -> tuple[list[str], list[list[float]]]:
    inputs = tokenizer(
        sentence,
        None,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        return_token_type_ids=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    ids = inputs["input_ids"].to(device, dtype=torch.long)
    mask = inputs["attention_mask"].to(device, dtype=torch.long)
    token_type_ids = inputs["token_type_ids"].to(device, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        outputs = model(ids, mask, token_type_ids)
        predictions = torch.sigmoid(outputs).cpu().detach().numpy()

    predicted_labels = [target_list[i] for i, value in enumerate(predictions[0]) if value > threshold]
    return predicted_labels, predictions.tolist()


def run_subset_experiment(
    df: pd.DataFrame,
    device: torch.device,
    best_model_path: str,
    subset_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    config: BertExperimentConfig | None = None,
    target_list: list[str] | None = None,
) -> dict[str, object]:
    config = config or BertExperimentConfig()
    target_list = target_list or DEFAULT_TARGET_LIST

    working_df = df.copy()
    if subset_fn is not None:
        working_df = subset_fn(working_df).reset_index(drop=True)

    tokenizer = get_tokenizer(config)
    train_df, val_df, train_loader, val_loader = build_dataloaders(
        working_df,
        tokenizer=tokenizer,
        config=config,
        target_list=target_list,
    )
    train_distribution, val_distribution = compute_distribution_classes(train_df, val_df, target_list)
    model, optimizer = build_model_and_optimizer(config, num_labels=len(target_list), device=device)
    model, training_losses, validation_losses, epochs_list = train_model(
        n_epochs=config.epochs,
        training_loader=train_loader,
        validation_loader=val_loader,
        model=model,
        optimizer=optimizer,
        best_model_path=best_model_path,
        device=device,
    )

    return {
        "tokenizer": tokenizer,
        "model": model,
        "train_df": train_df,
        "val_df": val_df,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_distribution": train_distribution,
        "val_distribution": val_distribution,
        "training_losses": training_losses,
        "validation_losses": validation_losses,
        "epochs_list": epochs_list,
    }
