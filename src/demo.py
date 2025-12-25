import os
import html
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from .config import TrainingConfig
from .dataset import LabelInfo


# Простенька палітра (повторюється по колу)
PALETTE = [
    "#ffd54f", "#81c784", "#64b5f6", "#e57373", "#ba68c8",
    "#4dd0e1", "#ff8a65", "#a1887f", "#90a4ae", "#dce775",
]

def _strip_bio(tag: str) -> str:
    if tag == "O":
        return "O"
    if "-" in tag and len(tag) > 2 and tag[1] == "-":
        return tag.split("-", 1)[1]
    return tag

def _build_color_map(entity_types: List[str]) -> Dict[str, str]:
    cmap = {"O": "transparent"}
    i = 0
    for ent in sorted(set(entity_types)):
        if ent == "O":
            continue
        cmap[ent] = PALETTE[i % len(PALETTE)]
        i += 1
    return cmap

def _merge_wordpieces(tokens: List[str], tags: List[str]) -> List[Tuple[str, str]]:
    """
    Склеює wordpiece токени в "людські" слова для красивого відображення.
    Повертає список (word, entity_type).
    """
    out = []
    cur_word = ""
    cur_ent = "O"

    def flush():
        nonlocal cur_word, cur_ent
        if cur_word:
            out.append((cur_word, cur_ent))
        cur_word, cur_ent = "", "O"

    for tok, tag in zip(tokens, tags):
        if tok in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        ent = _strip_bio(tag)

        # WordPiece для BERT: "##" означає продовження слова
        if tok.startswith("##"):
            cur_word += tok[2:]
        else:
            flush()
            cur_word = tok
            cur_ent = ent

        # якщо всередині слова різні теги — залишимо той, який не O
        if cur_ent == "O" and ent != "O":
            cur_ent = ent

    flush()
    return out

def _render_html(sentences: List[str], rendered: List[List[Tuple[str, str]]]) -> str:
    # зберемо список всіх entity types
    all_ents = []
    for sent in rendered:
        for _, ent in sent:
            all_ents.append(ent)

    cmap = _build_color_map(all_ents)

    legend_items = []
    for ent, color in cmap.items():
        if ent == "O":
            continue
        legend_items.append(
            f'<span class="legend-item"><span class="swatch" style="background:{color}"></span>{html.escape(ent)}</span>'
        )
    legend_html = "\n".join(legend_items)

    blocks = []
    for s, items in zip(sentences, rendered):
        spans = []
        for word, ent in items:
            w = html.escape(word)
            if ent == "O":
                spans.append(f"<span class='tok'>{w}</span>")
            else:
                color = cmap.get(ent, "#ffd54f")
                spans.append(
                    f"<span class='tok ent' style='background:{color}' title='{html.escape(ent)}'>{w}</span>"
                )
        blocks.append(
            "<div class='card'>"
            f"<div class='orig'><b>Text:</b> {html.escape(s)}</div>"
            f"<div class='pred'><b>NER:</b> {' '.join(spans)}</div>"
            "</div>"
        )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>NER Demo</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 24px; background:#fafafa; }}
  .legend {{ margin: 12px 0 18px; padding: 12px; background: #fff; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,.06); }}
  .legend-item {{ display:inline-flex; align-items:center; gap:8px; margin-right:14px; margin-bottom:8px; }}
  .swatch {{ width:14px; height:14px; border-radius:4px; display:inline-block; border:1px solid rgba(0,0,0,.12); }}
  .card {{ background:#fff; border-radius: 14px; padding: 14px 16px; margin: 12px 0; box-shadow: 0 2px 10px rgba(0,0,0,.06); }}
  .orig {{ color:#444; margin-bottom: 10px; }}
  .pred {{ line-height: 2.2; }}
  .tok {{ padding: 3px 6px; border-radius: 8px; margin-right: 4px; border:1px solid rgba(0,0,0,.06); }}
  .ent {{ font-weight: 600; }}
</style>
</head>
<body>
<h2>NER Demo (highlighted entities)</h2>
<div class="legend"><b>Legend:</b><div style="margin-top:10px;">{legend_html}</div></div>
{''.join(blocks)}
</body>
</html>
"""

@torch.inference_mode()
def make_demo_html(cfg: TrainingConfig, labels: LabelInfo, sentences: List[str], out_path: str = "demo/demo.html"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)


    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(cfg.output_dir)
    model.eval()

    rendered_all = []

    for s in sentences:
        enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=cfg.max_length)
        logits = model(**enc).logits[0].cpu().numpy()
        pred_ids = np.argmax(logits, axis=-1).tolist()

        toks = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
        tags = [model.config.id2label[i] for i in pred_ids]

        rendered_all.append(_merge_wordpieces(toks, tags))

    html_text = _render_html(sentences, rendered_all)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_text)

    print(f"[demo] Saved: {out_path}")