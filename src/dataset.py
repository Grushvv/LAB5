from dataclasses import dataclass
from typing import Dict, List, Tuple

from datasets import load_dataset as hf_load_dataset
from transformers import AutoTokenizer

from .config import TrainingConfig


@dataclass
class LabelInfo:
    id2label: Dict[int, str]
    label2id: Dict[str, int]
    label_list: List[str]


def _get_label_list(ds_train) -> List[str]:
    # Try to read labels from features (works in some versions)
    feats = ds_train.features
    tags_feat = feats.get("tags", None)
    if tags_feat is not None and hasattr(tags_feat, "feature") and hasattr(tags_feat.feature, "names"):
        return list(tags_feat.feature.names)

    # Fallback: hardcoded label list for tner/ontonotes5
    # (must match your label2id screenshot)
    label2id = {
        "O": 0,
        "B-CARDINAL": 1,
        "B-DATE": 2,
        "I-DATE": 3,
        "B-PERSON": 4,
        "I-PERSON": 5,
        "B-NORP": 6,
        "B-GPE": 7,
        "I-GPE": 8,
        "B-LAW": 9,
        "I-LAW": 10,
        "B-ORG": 11,
        "I-ORG": 12,
        "B-PERCENT": 13,
        "I-PERCENT": 14,
        "B-ORDINAL": 15,
        "B-MONEY": 16,
        "I-MONEY": 17,
        "B-WORK_OF_ART": 18,
        "I-WORK_OF_ART": 19,
        "B-FAC": 20,
        "B-TIME": 21,
        "I-CARDINAL": 22,
        "B-LOC": 23,
        "B-QUANTITY": 24,
        "I-QUANTITY": 25,
        "I-NORP": 26,
        "I-LOC": 27,
        "B-PRODUCT": 28,
        "I-TIME": 29,
        "B-EVENT": 30,
        "I-EVENT": 31,
        "I-FAC": 32,
        "B-LANGUAGE": 33,
        "I-PRODUCT": 34,
        "I-ORDINAL": 35,
        "I-LANGUAGE": 36,
    }
    # create label list where index == id
    label_list = [""] * (max(label2id.values()) + 1)
    for k, v in label2id.items():
        label_list[v] = k
    return label_list


def _align_labels_with_tokens(labels: List[int], word_ids: List[int], label_all_tokens: bool) -> List[int]:
    """
    Align word-level labels to subword tokens:
    - first subword of a word gets the label
    - other subwords get -100 (ignored), unless label_all_tokens=True
    """
    aligned = []
    prev_word_id = None
    for word_id in word_ids:
        if word_id is None:
            aligned.append(-100)
        elif word_id != prev_word_id:
            aligned.append(labels[word_id])
        else:
            if label_all_tokens:
                # convert B-xxx to I-xxx if possible, else keep
                aligned.append(labels[word_id])
            else:
                aligned.append(-100)
        prev_word_id = word_id
    return aligned


def load_ontonotes_ner(cfg: TrainingConfig):
    """
    Returns:
      raw: DatasetDict with original columns (tokens, tags)
      tokenized: DatasetDict with input_ids, attention_mask, labels
      labels: LabelInfo
    """
    raw = hf_load_dataset(cfg.dataset_name)

    label_list = _get_label_list(raw["train"])
    label2id = {name: i for i, name in enumerate(label_list)}
    id2label = {i: name for name, i in label2id.items()}
    labels = LabelInfo(id2label=id2label, label2id=label2id, label_list=label_list)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    def tokenize_and_align(batch):
        # tokens is a list of words
        tokenized = tokenizer(
            batch[cfg.tokens_column],
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=cfg.max_length,
        )

        all_labels = []
        for i, word_labels in enumerate(batch[cfg.tags_column]):
            word_ids = tokenized.word_ids(batch_index=i)
            aligned = _align_labels_with_tokens(word_labels, word_ids, cfg.label_all_tokens)
            all_labels.append(aligned)

        tokenized["labels"] = all_labels
        return tokenized

    remove_cols = list(raw["train"].features.keys())

    tokenized = raw.map(
        tokenize_and_align,
        batched=True,
        remove_columns=remove_cols,
    )



    tokenized.set_format("torch")
    return raw, tokenized, labels
