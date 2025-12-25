import os
import json
from typing import Dict

import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)
from seqeval.metrics import precision_score, recall_score, f1_score

from .config import TrainingConfig
from .dataset import LabelInfo


def _model_files_exist(output_dir: str) -> bool:
    if not os.path.isdir(output_dir):
        return False
    for fn in ["pytorch_model.bin", "model.safetensors"]:
        if os.path.exists(os.path.join(output_dir, fn)):
            return True
    # also accept adapter-style folders etc. but for full FT these are enough
    return False


def _compute_metrics_factory(id2label: Dict[int, str]):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        true_tags = []
        pred_tags = []
        for p_row, l_row in zip(preds, labels):
            t_seq = []
            p_seq = []
            for p_id, l_id in zip(p_row, l_row):
                if l_id == -100:
                    continue
                t_seq.append(id2label[int(l_id)])
                p_seq.append(id2label[int(p_id)])
            true_tags.append(t_seq)
            pred_tags.append(p_seq)

        return {
            "precision": precision_score(true_tags, pred_tags),
            "recall": recall_score(true_tags, pred_tags),
            "f1": f1_score(true_tags, pred_tags),
        }
    return compute_metrics


def build_trainer(cfg: TrainingConfig, tokenized, labels: LabelInfo, model=None) -> Trainer:
    if model is None:
        model = AutoModelForTokenClassification.from_pretrained(
            cfg.model_name,
            num_labels=len(labels.label_list),
            id2label=labels.id2label,
            label2id=labels.label2id,
        )

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        fp16=cfg.fp16,
        dataloader_num_workers=cfg.dataloader_num_workers,
        eval_strategy=cfg.evaluation_strategy,
        save_strategy=cfg.save_strategy,
        logging_strategy=cfg.logging_strategy,
        logging_steps=cfg.logging_steps,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        report_to="none",
        seed=cfg.seed,
        save_total_limit=2,
    )

    return Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=_compute_metrics_factory(labels.id2label),
    )


def train_if_needed(cfg: TrainingConfig, tokenized, labels: LabelInfo):
    """
    Train only if:
      - cfg.train_mode=True
      - and no saved model exists in cfg.output_dir
    """
    if not cfg.train_mode:
        return

    os.makedirs(cfg.output_dir, exist_ok=True)

    if _model_files_exist(cfg.output_dir):
        print(f"[train] Model already exists in '{cfg.output_dir}' -> skipping training.")
        return

    trainer = build_trainer(cfg, tokenized, labels)
    trainer.train()
    trainer.save_model(cfg.output_dir)

    # Save trainer state/logs to make plotting easy later
    state_path = os.path.join(cfg.output_dir, "trainer_state.json")
    try:
        trainer.state.save_to_json(state_path)
    except Exception:
        pass
