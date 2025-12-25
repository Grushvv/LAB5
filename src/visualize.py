import os
import json
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def _strip_bio(tag: str) -> str:
    if tag == "O":
        return "O"
    if "-" in tag and len(tag) > 2 and tag[1] == "-":
        return tag.split("-", 1)[1]
    return tag


def plot_token_confusion(true_tags: List[List[str]], pred_tags: List[List[str]], path: str):
    """
    Token-level confusion matrix AFTER stripping BIO prefixes.
    Useful to see what entities are confused with what.
    """
    y_true = [_strip_bio(t) for seq in true_tags for t in seq]
    y_pred = [_strip_bio(t) for seq in pred_tags for t in seq]

    labels = sorted(list(set(y_true) | set(y_pred)))
    # keep O first
    if "O" in labels:
        labels = ["O"] + [x for x in labels if x != "O"]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Token Confusion Matrix (BIO stripped)")
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_f1_per_entity(per_entity: Dict[str, Dict[str, float]], focus: List[str], path: str):
    """
    Bar chart for F1 per selected entity types.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    xs = []
    ys = []
    for ent in focus:
        if ent in per_entity:
            xs.append(ent)
            ys.append(per_entity[ent]["f1"])

    if not xs:
        return

    plt.figure(figsize=(9, 4))
    plt.bar(xs, ys)
    plt.ylim(0, 1.0)
    plt.ylabel("F1")
    plt.title("Entity-level F1 (selected types)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_token_confusion_no_o(y_true, y_pred, path="plots/token_confusion_no_O.png"):
    """
    Token-level confusion matrix WITHOUT class 'O'.
    Expects y_true/y_pred in seqeval format: List[List[str]] with BIO tags or plain tags.
    If BIO tags are present, they will be stripped (B-XXX/I-XXX -> XXX).
    """
    def strip_bio(tag: str) -> str:
        if tag == "O":
            return "O"
        if "-" in tag and len(tag) > 2 and tag[1] == "-":  # B-XXX / I-XXX
            return tag.split("-", 1)[1]
        return tag

    # Flatten, strip BIO, and remove O
    t_flat = []
    p_flat = []
    for t_seq, p_seq in zip(y_true, y_pred):
        for t, p in zip(t_seq, p_seq):
            tt = strip_bio(t)
            pp = strip_bio(p)
            if tt == "O" and pp == "O":
                continue
            if tt == "O" or pp == "O":
                # ми будуємо матрицю "між сутностями", тому O відкидаємо повністю
                continue
            t_flat.append(tt)
            p_flat.append(pp)

    labels = sorted(set(t_flat) | set(p_flat))
    if not labels:
        print("[warn] No entity tokens found after removing 'O'. Nothing to plot.")
        return

    idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for tt, pp in zip(t_flat, p_flat):
        cm[idx[tt], idx[pp]] += 1

    # Plot
    plt.figure(figsize=(10, 8))
    plt.title("Token Confusion Matrix (entities only, BIO stripped)")
    plt.imshow(cm, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"[plots] Saved: {path}")
def plot_learning_curves(output_dir: str, path: str = "plots/learning_curves.png"):
    """
    Reads output_dir/trainer_state.json (if exists) and plots train/eval loss per epoch/step.
    """
    state_path = os.path.join(output_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        return

    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    hist = state.get("log_history", [])
    train_steps = []
    train_loss = []
    eval_steps = []
    eval_loss = []

    for item in hist:
        if "loss" in item and "eval_loss" not in item:
            train_steps.append(item.get("step", len(train_steps)))
            train_loss.append(item["loss"])
        if "eval_loss" in item:
            eval_steps.append(item.get("step", len(eval_steps)))
            eval_loss.append(item["eval_loss"])

    if not train_loss and not eval_loss:
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.figure(figsize=(8, 4))
    if train_loss:
        plt.plot(train_steps, train_loss, label="train loss")
    if eval_loss:
        plt.plot(eval_steps, eval_loss, label="eval loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Learning Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
