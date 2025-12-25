from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # ========= DATASET =========
    dataset_name: str = "tner/ontonotes5"
    tokens_column: str = "tokens"
    tags_column: str = "tags"

    # ========= MODEL =========
    model_name: str = "distilbert-base-cased"
    output_dir: str = "checkpoints/bert_ner_ontonotes"
    train_mode: bool = False   # True -> train once & save | False -> load only

    # ========= TOKENIZATION =========
    max_length: int = 96
    label_all_tokens: bool = False  # subwords -> -100

    # ========= TRAINING (GTX 1650 safe defaults) =========
    num_epochs: int = 2
    train_batch_size: int = 8
    eval_batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    gradient_accumulation_steps: int = 1
    dataloader_pin_memory = False

    # ========= PRECISION / HW =========
    fp16: bool = True
    dataloader_num_workers: int = 2

    # ========= LOGGING / SAVE =========
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_strategy: str = "steps"
    logging_steps: int = 50
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"

    # ========= REPRO =========
    seed: int = 42
