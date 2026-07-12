import os
import glob
import time
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    DistilBertTokenizerFast, DistilBertForSequenceClassification,
    RobertaTokenizerFast, RobertaForSequenceClassification,
    Trainer, TrainingArguments
)

warnings.filterwarnings("ignore")

# ============================================
# CONFIG
# ============================================

DATA_GLOB = "datasets/*.csv"    # folder containing your CSVs
RANDOM_STATE = 42
TEST_SIZE = 0.2

NUM_EPOCHS = 1  # single outer epoch for all models
TRAIN_SAMPLES_PER_EPOCH_CLASSICAL = 200_000
TRAIN_SAMPLES_PER_EPOCH_CNN = 200_000
TRAIN_SAMPLES_PER_EPOCH_TRANSFORMER = 200_000  # reduce if too slow

# Cap total aspect-level samples to avoid MemoryError
MAX_ABSA_SAMPLES = 500_000

TFIDF_MAX_FEATURES = 20_000      # lower to reduce memory
VOCAB_FIT_SAMPLES = 200_000      # subset for fitting TF-IDF vocab

CNN_BATCH_SIZE = 64
CNN_LR = 1e-3

TRANSFORMER_BATCH = 8
TRANSFORMER_MAX_LEN = 256

OUTPUT_CSV = "absa_model_comparison_1epoch_balanced.csv"

CLASS_MAP = {0: "negative", 1: "neutral", 2: "positive"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================
# ASPECT DEFINITIONS PER DOMAIN (heuristic)
# ============================================

ASPECT_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "DEFAULT": {
        "price": ["price", "cost", "value", "worth", "cheap", "expensive"],
        "quality": ["quality", "build", "durable", "broken", "defect", "poor", "excellent"],
        "delivery": ["delivery", "shipping", "shipped", "arrived", "late", "on time", "packaging", "packed"],
        "customer_service": ["customer service", "support", "refund", "replacement", "return", "service"]
    },

    "Books": {
        "story_plot": ["story", "plot", "narrative", "ending", "beginning", "twist"],
        "writing_style": ["writing", "style", "prose", "author", "language"],
        "characters": ["character", "protagonist", "villain", "hero", "cast"],
        "price": ["price", "cost", "value"]
    },

    "Electronics": {
        "battery": ["battery", "charge", "charging", "power life"],
        "screen_display": ["screen", "display", "resolution", "brightness", "panel"],
        "performance": ["performance", "speed", "lag", "slow", "fast", "processor"],
        "sound": ["sound", "audio", "speaker", "volume", "noise"],
        "camera": ["camera", "photo", "video", "picture", "image quality"],
        "connectivity": ["wifi", "bluetooth", "connection", "network", "signal"],
        "price": ["price", "cost", "value", "expensive", "cheap"],
        "delivery": ["delivery", "shipping", "arrived", "packaging"]
    },

    "Cell_Phones_and_Accessories": {
        "battery": ["battery", "charge", "charging", "power life"],
        "screen": ["screen", "display", "glass", "touch"],
        "camera": ["camera", "photo", "picture", "image"],
        "case_fit": ["fit", "case", "cover", "protect"],
        "connectivity": ["network", "signal", "reception", "wifi"],
        "price": ["price", "cost", "value"],
        "delivery": ["delivery", "shipping", "arrived", "packaging"]
    },

    "Clothing_Shoes_and_Jewelry": {
        "size_fit": ["size", "fit", "fitting", "tight", "loose", "small", "large"],
        "material_quality": ["material", "fabric", "cloth", "quality", "stitch", "thread"],
        "comfort": ["comfortable", "comfort", "itchy", "soft", "rough"],
        "style_appearance": ["style", "look", "design", "fashion", "color", "colour"],
        "price": ["price", "cost", "expensive", "cheap", "value"],
        "delivery": ["delivery", "shipping", "arrived", "packaging"]
    },

    "AMAZON_FASHION": {
        "size_fit": ["size", "fit", "fitting", "tight", "loose", "small", "large"],
        "material_quality": ["material", "fabric", "cloth", "quality", "stitch", "thread"],
        "comfort": ["comfortable", "comfort", "itchy", "soft", "rough"],
        "style_appearance": ["style", "look", "design", "fashion", "color", "colour"],
        "price": ["price", "cost", "expensive", "cheap", "value"],
        "delivery": ["delivery", "shipping", "arrived", "packaging"]
    },

    "All_Beauty": {
        "effectiveness": ["effective", "works", "result", "results", "helped"],
        "ingredients": ["ingredient", "paraben", "sulfate", "natural", "organic"],
        "scent": ["smell", "fragrance", "scent", "odor", "odour"],
        "skin_reaction": ["irritation", "rash", "allergy", "breakout", "acne"],
        "price": ["price", "cost", "value"],
    },

    "Home_and_Kitchen": {
        "build_quality": ["quality", "sturdy", "durable", "broke", "broken"],
        "ease_of_use": ["easy to use", "difficult", "hard to use", "setup", "install"],
        "design": ["design", "look", "style", "color"],
        "cleaning": ["clean", "wash", "washing", "easy to clean"],
        "price": ["price", "cost", "value"],
        "delivery": ["delivery", "shipping", "arrived", "packaging"]
    },

    "Grocery_and_Gourmet_Food": {
        "taste": ["taste", "flavor", "flavour", "tasty", "delicious", "yummy"],
        "freshness": ["fresh", "stale", "expired", "rotten", "quality"],
        "quantity": ["quantity", "amount", "size", "portion"],
        "price": ["price", "cost", "value"],
        "packaging": ["packaging", "packed", "box", "bag"],
        "delivery": ["delivery", "shipping", "arrived"]
    },

    "Pet_Supplies": {
        "pet_likes": ["dog loves", "cat loves", "my dog", "my cat", "pet likes", "pet loves"],
        "quality": ["quality", "durable", "broke", "sturdy"],
        "nutrition": ["nutrition", "healthy", "ingredients", "allergy"],
        "price": ["price", "cost", "value"],
        "delivery": ["delivery", "shipping", "arrived", "packaging"]
    },

    "Sports_and_Outdoors": {
        "durability": ["durable", "broke", "broken", "sturdy"],
        "performance": ["performance", "works well", "usability", "function"],
        "comfort": ["comfortable", "comfort", "fit"],
        "price": ["price", "cost", "value"],
        "delivery": ["delivery", "shipping", "arrived", "packaging"]
    },

    "Toys_and_Games": {
        "fun_factor": ["fun", "enjoy", "enjoyed", "kids love", "children love"],
        "safety": ["safe", "dangerous", "choking", "hazard"],
        "durability": ["durable", "broke", "broken", "sturdy"],
        "price": ["price", "cost", "value"],
        "delivery": ["delivery", "shipping", "arrived", "packaging"]
    },

    "CDs_and_Vinyl": {
        "audio_quality": ["sound", "audio", "quality", "noise"],
        "content": ["songs", "tracks", "album", "music"],
        "packaging": ["case", "packaging", "scratched", "scratch"],
        "price": ["price", "cost", "value"],
    },

    "Musical_Instruments": {
        "sound_quality": ["sound", "tone", "audio"],
        "build_quality": ["quality", "durable", "broke", "broken"],
        "playability": ["play", "playable", "feel", "action"],
        "price": ["price", "cost", "value"]
    },

    "Office_Products": {
        "quality": ["quality", "broke", "broken", "sturdy", "durable"],
        "usability": ["use", "easy", "difficult", "hard"],
        "price": ["price", "cost", "value"],
        "delivery": ["delivery", "shipping", "arrived", "packaging"]
    },

    "Industrial_and_Scientific": {
        "quality": ["quality", "durable", "broke", "broken"],
        "performance": ["performance", "accuracy", "precise", "works"],
        "price": ["price", "cost", "value"]
    },

    "Patio_Lawn_and_Garden": {
        "durability": ["durable", "broke", "broken", "sturdy"],
        "appearance": ["look", "design", "color"],
        "installation": ["install", "setup", "easy to install"],
        "price": ["price", "cost", "value"]
    },

    "Appliances": {
        "performance": ["works", "performance", "function", "cool", "heat"],
        "noise": ["noise", "noisy", "loud", "quiet"],
        "energy": ["energy", "power", "watt", "electric"],
        "price": ["price", "cost", "value"]
    },

    "Arts_Crafts_and_Sewing": {
        "quality": ["quality", "durable", "broke", "broken"],
        "color_accuracy": ["color", "colour", "shade", "vibrant"],
        "ease_of_use": ["easy", "difficult", "hard", "use"],
        "price": ["price", "cost", "value"]
    },
}

# ============================================
# UTILS
# ============================================

def load_all_csvs_with_domain(pattern=DATA_GLOB) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSVs found matching: {pattern}")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            base = os.path.basename(f)
            if "_5_" in base:
                domain = base.split("_5_")[0]
            else:
                domain = os.path.splitext(base)[0]
            df["domain"] = domain
            print(f"Loaded {f} -> {len(df)} rows | domain={domain}")
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")
    return pd.concat(dfs, ignore_index=True)


def rating_to_label(r):
    try:
        r = float(r)
    except Exception:
        return np.nan
    if r <= 2:
        return 0
    elif r == 3:
        return 1
    else:
        return 2


def evaluate_and_record(name, y_true, y_pred, train_time, infer_time, records):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"\n{name} results:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1 (wtd):  {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  Train time:    {train_time:.2f}s")
    print(f"  Inference time:{infer_time:.2f}s")
    records.append({
        "model": name,
        "accuracy": acc,
        "f1_weighted": f1,
        "precision_weighted": prec,
        "recall_weighted": rec,
        "train_time_s": train_time,
        "inference_time_s": infer_time
    })


def build_absa_dataframe(df: pd.DataFrame, aspect_keywords_map: Dict) -> pd.DataFrame:
    """
    Input df needs: reviewText, overall, domain
    Output: text, aspect, label, domain
    """
    rows = []
    it = tqdm(df.itertuples(index=False), total=len(df), desc="Building aspect-level dataset")
    for row in it:
        text = str(getattr(row, "reviewText", ""))
        overall = getattr(row, "overall", None)
        domain = getattr(row, "domain", "DEFAULT")
        label = rating_to_label(overall)
        if np.isnan(label):
            continue
        label = int(label)
        text_low = text.lower()
        aspect_dict = aspect_keywords_map.get(domain, aspect_keywords_map["DEFAULT"])
        for aspect_name, kw_list in aspect_dict.items():
            if any(k.lower() in text_low for k in kw_list):
                rows.append({
                    "text": text,
                    "aspect": aspect_name,
                    "label": label,
                    "domain": domain
                })
    return pd.DataFrame(rows)


def balanced_sample_by_group(df, n_total, group_col, epoch, base_seed=RANDOM_STATE):
    """
    Sample approximately n_total rows, balanced across values of group_col (e.g., 'domain').
    Oversamples (with replacement) if a group is too small.
    """
    groups = df[group_col].unique()
    n_groups = len(groups)
    if n_groups == 0:
        raise ValueError("No groups found for balanced sampling.")

    n_per = max(1, n_total // n_groups)
    samples = []

    for i, g in enumerate(groups):
        g_df = df[df[group_col] == g]
        if len(g_df) == 0:
            continue
        rs = base_seed + epoch * 100 + i
        if len(g_df) >= n_per:
            s = g_df.sample(n=n_per, random_state=rs, replace=False)
        else:
            s = g_df.sample(n=n_per, random_state=rs, replace=True)
        samples.append(s)

    if not samples:
        raise RuntimeError("balanced_sample_by_group produced no samples.")

    batch = pd.concat(samples).reset_index(drop=True)

    if len(batch) > n_total:
        batch = batch.sample(n=n_total, random_state=base_seed + epoch).reset_index(drop=True)

    return batch


# ============================================
# PLOTTING HELPERS
# ============================================

def plot_training_curve(steps, losses, title, filename):
    if not steps or not losses:
        return
    plt.figure()
    plt.plot(steps, losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_confusion(y_true, y_pred, model_name, filename):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])
    plt.figure()
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_test_bar_chart(results_df, filename):
    plt.figure()
    x = np.arange(len(results_df))
    width = 0.35
    plt.bar(x - width/2, results_df["accuracy"], width, label="Accuracy")
    plt.bar(x + width/2, results_df["f1_weighted"], width, label="F1")
    plt.xticks(x, results_df["model"], rotation=45, ha="right")
    plt.ylabel("Score")
    plt.title("Test Accuracy & F1 for All Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# ============================================
# CNN Dataset & Model
# ============================================

class TFIDFDataset(Dataset):
    def __init__(self, sparse_matrix, labels):
        self.X = sparse_matrix
        self.y = labels.values.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        row = self.X[idx].toarray().squeeze().astype(np.float32)
        return torch.from_numpy(row), torch.tensor(self.y[idx], dtype=torch.long)


class SimpleCNN(nn.Module):
    def __init__(self, input_len, num_classes):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)           # (B, 1, L)
        x = self.conv(x)             # (B, 64, L)
        x = self.act(x)
        x = self.pool(x).squeeze(-1) # (B, 64)
        return self.fc(x)


# ============================================
# HF Dataset for transformers
# ============================================

class HFDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.values.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ============================================
# MAIN
# ============================================

print("Loading raw CSV data...")
df_raw = load_all_csvs_with_domain(DATA_GLOB)

needed_cols = [c for c in ["reviewText", "overall", "domain"] if c in df_raw.columns]
if not set(["reviewText", "overall", "domain"]).issubset(needed_cols):
    raise ValueError("CSVs must contain 'reviewText' and 'overall' columns.")

df_base = df_raw[needed_cols].dropna(subset=["reviewText", "overall"]).reset_index(drop=True)
print(f"Base cleaned reviews: {len(df_base)}")

print("Extracting aspect-level samples...")
absa_df = build_absa_dataframe(df_base, ASPECT_KEYWORDS)
print(f"Total aspect-level samples before cap: {len(absa_df)}")
print("Label distribution (0=neg,1=neu,2=pos):")
print(absa_df["label"].value_counts())

# Cap total aspect-level samples in a balanced way across domains
if len(absa_df) > MAX_ABSA_SAMPLES:
    absa_df = balanced_sample_by_group(
        absa_df,
        n_total=MAX_ABSA_SAMPLES,
        group_col="domain",
        epoch=0,
        base_seed=RANDOM_STATE
    )
    print(f"\nCapped aspect-level samples (balanced across domains) to: {len(absa_df)}")

if len(absa_df) == 0:
    raise RuntimeError("No aspect-level data created. Check aspect keyword lists.")

absa_df["input_text"] = "aspect: " + absa_df["aspect"].astype(str) + " [SEP] review: " + absa_df["text"].astype(str)

full_df = absa_df[["input_text", "label", "domain"]]

train_df, test_df = train_test_split(
    full_df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=full_df["label"]
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"Train samples (aspect-level): {len(train_df)}")
print(f"Test samples  (aspect-level): {len(test_df)}")

y_all = full_df["label"]
n_classes = len(np.unique(y_all))
classes_arr = np.arange(n_classes)

# ============================================
# TF-IDF setup
# ============================================

print("\nFitting TF-IDF vocabulary...")
vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2))

if len(train_df) > VOCAB_FIT_SAMPLES:
    fit_sample = train_df.sample(n=VOCAB_FIT_SAMPLES, random_state=RANDOM_STATE)["input_text"]
else:
    fit_sample = train_df["input_text"]

vectorizer.fit(fit_sample)
print("TF-IDF vocab size:", len(vectorizer.get_feature_names_out()))

X_test_tfidf = vectorizer.transform(test_df["input_text"])
y_test_series = test_df["label"]

test_cnn_ds = TFIDFDataset(X_test_tfidf, y_test_series)
test_cnn_loader = DataLoader(test_cnn_ds, batch_size=CNN_BATCH_SIZE)

# ============================================
# Initialize models
# ============================================

sgd_log = SGDClassifier(loss="log_loss", random_state=RANDOM_STATE)
sgd_svm = SGDClassifier(loss="hinge", random_state=RANDOM_STATE)

dummy_vec = vectorizer.transform(train_df["input_text"].iloc[:1])
n_features = dummy_vec.shape[1]
cnn_model = SimpleCNN(n_features, n_classes).to(device)
cnn_criterion = nn.CrossEntropyLoss()
cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=CNN_LR)

distil_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
distil_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=n_classes
).to(device)

roberta_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
roberta_model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=n_classes
).to(device)

distil_args = TrainingArguments(
    output_dir="./distilbert_absa",
    num_train_epochs=1,  # Trainer will run 1 epoch per call
    per_device_train_batch_size=TRANSFORMER_BATCH,
    per_device_eval_batch_size=TRANSFORMER_BATCH,
    logging_steps=200,
    seed=RANDOM_STATE,
    fp16=torch.cuda.is_available(),
    logging_dir="./distilbert_logs"
)

roberta_args = TrainingArguments(
    output_dir="./roberta_absa",
    num_train_epochs=1,
    per_device_train_batch_size=TRANSFORMER_BATCH,
    per_device_eval_batch_size=TRANSFORMER_BATCH,
    logging_steps=200,
    seed=RANDOM_STATE,
    fp16=torch.cuda.is_available(),
    logging_dir="./roberta_logs"
)

distil_test_enc = distil_tokenizer(
    test_df["input_text"].tolist(),
    truncation=True, padding=True, max_length=TRANSFORMER_MAX_LEN
)
roberta_test_enc = roberta_tokenizer(
    test_df["input_text"].tolist(),
    truncation=True, padding=True, max_length=TRANSFORMER_MAX_LEN
)
distil_test_ds = HFDataset(distil_test_enc, y_test_series)
roberta_test_ds = HFDataset(roberta_test_enc, y_test_series)

# ============================================
# Training over 1 epoch, 200k balanced samples/epoch
# ============================================

results = []

time_log_sgd = 0.0
time_svm_sgd = 0.0
time_cnn = 0.0
time_distil = 0.0
time_roberta = 0.0

# For training curves
cnn_train_losses = []
cnn_train_steps = []
cnn_step_count = 0

distil_train_steps = []
distil_train_losses = []

roberta_train_steps = []
roberta_train_losses = []

for epoch in range(NUM_EPOCHS):
    print(f"\n================= OUTER EPOCH {epoch+1}/{NUM_EPOCHS} =================")

    # --- Balanced batch for classical & CNN ---
    batch_size_epoch = min(TRAIN_SAMPLES_PER_EPOCH_CLASSICAL, len(train_df))
    train_batch = balanced_sample_by_group(
        train_df,
        n_total=batch_size_epoch,
        group_col="domain",
        epoch=epoch,
        base_seed=RANDOM_STATE
    )

    X_batch_text = train_batch["input_text"]
    y_batch = train_batch["label"]

    X_batch_tfidf = vectorizer.transform(X_batch_text)

    # ========= SGD Logistic =========
    print("Epoch", epoch+1, "- SGD Logistic partial_fit...")
    start = time.time()
    if epoch == 0:
        sgd_log.partial_fit(X_batch_tfidf, y_batch, classes=classes_arr)
    else:
        sgd_log.partial_fit(X_batch_tfidf, y_batch)
    time_log_sgd += (time.time() - start)

    # ========= SGD SVM =========
    print("Epoch", epoch+1, "- SGD SVM partial_fit...")
    start = time.time()
    if epoch == 0:
        sgd_svm.partial_fit(X_batch_tfidf, y_batch, classes=classes_arr)
    else:
        sgd_svm.partial_fit(X_batch_tfidf, y_batch)
    time_svm_sgd += (time.time() - start)

    # ========= CNN =========
    print("Epoch", epoch+1, "- CNN training...")
    X_batch_tfidf_cnn = X_batch_tfidf
    cnn_train_ds = TFIDFDataset(X_batch_tfidf_cnn, y_batch)
    cnn_train_loader = DataLoader(cnn_train_ds, batch_size=CNN_BATCH_SIZE, shuffle=True)

    start = time.time()
    cnn_model.train()
    loop = tqdm(cnn_train_loader, desc=f"CNN epoch {epoch+1}")
    for xb, yb in loop:
        xb = xb.to(device)
        yb = yb.to(device)
        cnn_optimizer.zero_grad()
        out = cnn_model(xb)
        loss = cnn_criterion(out, yb)
        loss.backward()
        cnn_optimizer.step()

        cnn_step_count += 1
        cnn_train_steps.append(cnn_step_count)
        cnn_train_losses.append(loss.item())

        loop.set_postfix(loss=loss.item())
    time_cnn += (time.time() - start)

    # --- Balanced batch for transformers ---
    trans_batch_size = min(TRAIN_SAMPLES_PER_EPOCH_TRANSFORMER, len(train_df))
    trans_batch = balanced_sample_by_group(
        train_df,
        n_total=trans_batch_size,
        group_col="domain",
        epoch=epoch + 1000,
        base_seed=RANDOM_STATE
    )

    # ========= DistilBERT =========
    print("Epoch", epoch+1, "- DistilBERT training...")
    distil_train_enc = distil_tokenizer(
        trans_batch["input_text"].tolist(),
        truncation=True, padding=True, max_length=TRANSFORMER_MAX_LEN
    )
    distil_train_ds = HFDataset(distil_train_enc, trans_batch["label"])

    distil_trainer = Trainer(model=distil_model, args=distil_args, train_dataset=distil_train_ds)

    start = time.time()
    distil_trainer.train()
    time_distil += (time.time() - start)

    # Extract training loss from DistilBERT trainer log
    distil_hist = distil_trainer.state.log_history
    for h in distil_hist:
        if "loss" in h and "step" in h:
            distil_train_steps.append(h["step"])
            distil_train_losses.append(h["loss"])

    # ========= RoBERTa =========
    print("Epoch", epoch+1, "- RoBERTa training...")
    roberta_train_enc = roberta_tokenizer(
        trans_batch["input_text"].tolist(),
        truncation=True, padding=True, max_length=TRANSFORMER_MAX_LEN
    )
    roberta_train_ds = HFDataset(roberta_train_enc, trans_batch["label"])

    roberta_trainer = Trainer(model=roberta_model, args=roberta_args, train_dataset=roberta_train_ds)

    start = time.time()
    roberta_trainer.train()
    time_roberta += (time.time() - start)

    # Extract training loss from RoBERTa trainer log
    roberta_hist = roberta_trainer.state.log_history
    for h in roberta_hist:
        if "loss" in h and "step" in h:
            roberta_train_steps.append(h["step"])
            roberta_train_losses.append(h["loss"])

# ============================================
# PLOT TRAINING CURVES
# ============================================

plot_training_curve(
    cnn_train_steps,
    cnn_train_losses,
    "CNN Training Loss (ABSA, 1 epoch)",
    "cnn_training_loss.png"
)

plot_training_curve(
    distil_train_steps,
    distil_train_losses,
    "DistilBERT Training Loss (ABSA, 1 epoch)",
    "distilbert_training_loss.png"
)

plot_training_curve(
    roberta_train_steps,
    roberta_train_losses,
    "RoBERTa Training Loss (ABSA, 1 epoch)",
    "roberta_training_loss.png"
)

print("Saved training loss curves: cnn_training_loss.png, distilbert_training_loss.png, roberta_training_loss.png")

# ============================================
# FINAL EVALUATION
# ============================================

print("\n================= FINAL EVALUATION =================")

results = []

# --- SGD Logistic ---
start = time.time()
sgd_log_pred = sgd_log.predict(X_test_tfidf)
infer_time = time.time() - start
evaluate_and_record("SGD Logistic (ABSA, 1 epoch)", y_test_series, sgd_log_pred, time_log_sgd, infer_time, results)
plot_confusion(y_test_series, sgd_log_pred, "SGD Logistic (ABSA, 1 epoch)", "cm_sgd_log_absa.png")

# --- SGD SVM ---
start = time.time()
sgd_svm_pred = sgd_svm.predict(X_test_tfidf)
infer_time = time.time() - start
evaluate_and_record("SGD SVM (ABSA, 1 epoch)", y_test_series, sgd_svm_pred, time_svm_sgd, infer_time, results)
plot_confusion(y_test_series, sgd_svm_pred, "SGD SVM (ABSA, 1 epoch)", "cm_sgd_svm_absa.png")

# --- CNN ---
start = time.time()
cnn_model.eval()
cnn_preds = []
with torch.no_grad():
    for xb, yb in test_cnn_loader:
        xb = xb.to(device)
        out = cnn_model(xb)
        preds = out.argmax(dim=1).cpu().numpy()
        cnn_preds.extend(preds)
infer_time = time.time() - start
evaluate_and_record("Simple CNN (ABSA, 1 epoch)", y_test_series, cnn_preds, time_cnn, infer_time, results)
plot_confusion(y_test_series, cnn_preds, "Simple CNN (ABSA, 1 epoch)", "cm_cnn_absa.png")

# --- DistilBERT ---
start = time.time()
distil_eval_out = distil_trainer.predict(distil_test_ds)
distil_test_pred = np.argmax(distil_eval_out.predictions, axis=1)
infer_time = time.time() - start
evaluate_and_record("DistilBERT (ABSA, 1 epoch)", y_test_series, distil_test_pred, time_distil, infer_time, results)
plot_confusion(y_test_series, distil_test_pred, "DistilBERT (ABSA, 1 epoch)", "cm_distilbert_absa.png")

# --- RoBERTa ---
start = time.time()
roberta_eval_out = roberta_trainer.predict(roberta_test_ds)
roberta_test_pred = np.argmax(roberta_eval_out.predictions, axis=1)
infer_time = time.time() - start
evaluate_and_record("RoBERTa (ABSA, 1 epoch)", y_test_series, roberta_test_pred, time_roberta, infer_time, results)
plot_confusion(y_test_series, roberta_test_pred, "RoBERTa (ABSA, 1 epoch)", "cm_roberta_absa.png")

# ============================================
# SAVE RESULTS
# ============================================

df_results = pd.DataFrame(results).sort_values(by="f1_weighted", ascending=False).reset_index(drop=True)
print("\nFinal ABSA model comparison (after 1 epoch):")
print(df_results)
df_results.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved results to {OUTPUT_CSV}")
print("Label mapping:", CLASS_MAP)

# Test metrics bar chart
plot_test_bar_chart(df_results, "test_metrics_bar.png")
print("Saved test metrics bar chart as test_metrics_bar.png")
print("Saved confusion matrices as cm_*.png")
