# main.py  (LOCAL RUN)
# IMPORTANT: set HF cache dirs BEFORE importing datasets/transformers
import os

HF_ROOT = r"F:\TEMP\hf"
os.environ["HF_HOME"] = HF_ROOT
os.environ["HF_DATASETS_CACHE"] = rf"{HF_ROOT}\datasets"
os.environ["HF_HOME"] = rf"{HF_ROOT}\models"

from src.config import TrainingConfig
from src.dataset import load_ontonotes_ner
from src.train import train_if_needed
from src.eval import evaluate_and_save
from src.visualize import plot_learning_curves
from src.demo import make_demo_html

def main():
    cfg = TrainingConfig()

    raw, tokenized, labels = load_ontonotes_ner(cfg)

    # Train once and save (if cfg.train_mode=True)
    train_if_needed(cfg, tokenized, labels)

    # Always run evaluation (loads best model from cfg.output_dir)
    metrics, per_type = evaluate_and_save(cfg, raw, tokenized, labels)

    # Optional: plot loss curves from trainer logs (if training happened at least once)
    plot_learning_curves(cfg.output_dir)

    print("\n========================================")
    print("FINAL (Test) — Entity-level (seqeval)")
    print("========================================")
    print(f"precision={metrics['precision']:.4f}  recall={metrics['recall']:.4f}  f1={metrics['f1']:.4f}")
    print("\nTop entities in report (see reports/):")
    for k in ["PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", "MONEY", "PERCENT", "QUANTITY"]:
        if k in per_type:
            v = per_type[k]
            print(f"  {k:9s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}")

    print("\nSaved:")
    print(f"  model/checkpoints: {cfg.output_dir}")
    print("  reports/: report.txt, report.json, per_entity.json")
    print("  plots/: learning_curves.png (if logs exist), token_confusion.png, f1_per_entity.png")

    demo_sentences = [
        "On 12 March 2024, Apple paid $3,000 to John Doe in New York.",
        "Microsoft signed a contract on 01/01/2023 for €50 million in London at 3:00 PM.",
        "In Texas Will sells burgers in Smith`s Burgers restaurant.",
        "In Texas, Will Brown sells burgers at Smith's Burgers restaurant.",
        "Mike Gray bought an apple for $1 on Saturday, 12/04/2025.",
    ]
    make_demo_html(cfg, labels, demo_sentences, out_path="demo/demo.html")

if __name__ == "__main__":
    main()
