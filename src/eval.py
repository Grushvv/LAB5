import os
import json
from typing import Dict, Tuple

import numpy as np
from transformers import AutoModelForTokenClassification
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score

from .config import TrainingConfig
from .dataset import LabelInfo
from .train import build_trainer
from .visualize import plot_token_confusion, plot_f1_per_entity
from .visualize import plot_token_confusion_no_o

def _load_model_from_dir(cfg: TrainingConfig, labels: LabelInfo):
    if not os.path.isdir(cfg.output_dir):
        raise FileNotFoundError(
            f"Output dir '{cfg.output_dir}' not found. "
            "Set train_mode=True once to train & save the model."
        )

    return AutoModelForTokenClassification.from_pretrained(
        cfg.output_dir,
        num_labels=len(labels.label_list),
        id2label=labels.id2label,
        label2id=labels.label2id,
    )


def _to_seqeval_strings(pred_ids, label_ids, id2label: Dict[int, str]):
    true_tags, pred_tags = [], []
    for p_row, l_row in zip(pred_ids, label_ids):
        t_seq, p_seq = [], []
        for p_id, l_id in zip(p_row, l_row):
            if l_id == -100:
                continue
            t_seq.append(id2label[int(l_id)])
            p_seq.append(id2label[int(p_id)])
        true_tags.append(t_seq)
        pred_tags.append(p_seq)
    return true_tags, pred_tags


def _extract_per_entity(report_dict: Dict) -> Dict[str, Dict[str, float]]:
    per = {}
    for k, v in report_dict.items():
        if k in ["micro avg", "macro avg", "weighted avg", "accuracy"]:
            continue
        if not isinstance(v, dict) or "f1-score" not in v:
            continue

        ent = k
        if "-" in k and len(k) > 2 and k[1] == "-":  # B-XXX / I-XXX
            ent = k.split("-", 1)[1]

        per.setdefault(ent, {"precision": [], "recall": [], "f1": [], "support": []})
        per[ent]["precision"].append(v.get("precision", 0.0))
        per[ent]["recall"].append(v.get("recall", 0.0))
        per[ent]["f1"].append(v.get("f1-score", 0.0))
        per[ent]["support"].append(v.get("support", 0.0))

    out = {}
    for ent, vals in per.items():
        out[ent] = {
            "precision": float(np.mean(vals["precision"])) if vals["precision"] else 0.0,
            "recall": float(np.mean(vals["recall"])) if vals["recall"] else 0.0,
            "f1": float(np.mean(vals["f1"])) if vals["f1"] else 0.0,
            "support": float(np.sum(vals["support"])) if vals["support"] else 0.0,
        }
    return out


# 🔧 ВАЖЛИВО: конвертація numpy → builtin Python
def _to_builtin(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_builtin(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def evaluate_and_save(
    cfg: TrainingConfig, raw, tokenized, labels: LabelInfo
) -> Tuple[Dict, Dict]:

    os.makedirs("reports", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    model = _load_model_from_dir(cfg, labels)
    trainer = build_trainer(cfg, tokenized, labels, model=model)

    pred = trainer.predict(tokenized["test"])
    logits = pred.predictions
    pred_ids = np.argmax(logits, axis=-1)
    label_ids = pred.label_ids

    true_tags, pred_tags = _to_seqeval_strings(
        pred_ids, label_ids, labels.id2label
    )

    metrics = {
        "precision": float(precision_score(true_tags, pred_tags)),
        "recall": float(recall_score(true_tags, pred_tags)),
        "f1": float(f1_score(true_tags, pred_tags)),
    }

    # 📄 Текстовий звіт
    report_txt = classification_report(true_tags, pred_tags, digits=4)
    with open("reports/report.txt", "w", encoding="utf-8") as f:
        f.write(report_txt)

    # 📦 JSON звіт (вже без numpy типів)
    report_dict = classification_report(
        true_tags, pred_tags, output_dict=True
    )
    report_dict = _to_builtin(report_dict)

    with open("reports/report.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)

    per_entity = _extract_per_entity(report_dict)
    per_entity = _to_builtin(per_entity)

    with open("reports/per_entity.json", "w", encoding="utf-8") as f:
        json.dump(per_entity, f, ensure_ascii=False, indent=2)

    # 📊 Графіки
    plot_token_confusion_no_o(true_tags, pred_tags, path="plots/token_confusion_no_O.png")
    

    plot_f1_per_entity(
        per_entity,
        focus=[
            "PERSON", "ORG", "GPE", "LOC",
            "DATE", "TIME", "MONEY", "PERCENT", "QUANTITY"
        ],
        path="plots/f1_per_entity.png",
    )

    return metrics, per_entity