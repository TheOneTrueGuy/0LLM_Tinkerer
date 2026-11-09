"""
Quantitative eval probe — the fitness signal the agent loop was missing.

Two cheap metrics, both single forward passes per item:

- MC accuracy: multiple-choice questions scored by comparing the logprob
  of the answer letter at the final position (MMLU-style). Works on weak
  models where free-form generation can't be scored.
- Perplexity canary: mean NLL on a fixed reference passage — detects
  general fluency breakage from weight edits even when a targeted
  change "worked".

Run an eval before and after an op; the delta in mc_acc/ppl is the
keep/revert signal. Results accumulate in ``history`` so the UI and the
agent can compare sequential measurements.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

LETTERS = ["A", "B", "C", "D"]

# ~28 probes across domains, correct answer positions shuffled to avoid
# letter-position bias. Easy-to-medium difficulty so small models sit
# above the 25% chance floor.
PROBES: List[Dict[str, Any]] = [
    {"q": "What is the capital of France?",
     "choices": ["Paris", "London", "Berlin", "Madrid"], "a": 0},
    {"q": "What is 7 + 5?", "choices": ["10", "12", "13", "15"], "a": 1},
    {"q": "What is the chemical formula for water?",
     "choices": ["CO2", "O2", "H2O", "NaCl"], "a": 2},
    {"q": "Which is the largest planet in our solar system?",
     "choices": ["Mars", "Earth", "Venus", "Jupiter"], "a": 3},
    {"q": "Who wrote Romeo and Juliet?",
     "choices": ["Charles Dickens", "William Shakespeare",
                 "Jane Austen", "Mark Twain"], "a": 1},
    {"q": "What color do you get by mixing blue and yellow paint?",
     "choices": ["Purple", "Orange", "Green", "Brown"], "a": 2},
    {"q": "How many days are there in a week?",
     "choices": ["5", "6", "7", "8"], "a": 2},
    {"q": "Which ocean is the largest?",
     "choices": ["Atlantic", "Indian", "Arctic", "Pacific"], "a": 3},
    {"q": "What is 3 times 9?",
     "choices": ["27", "24", "21", "18"], "a": 0},
    {"q": "Which animal is known as man's best friend?",
     "choices": ["Cat", "Dog", "Horse", "Rabbit"], "a": 1},
    {"q": "What is the boiling point of water at sea level?",
     "choices": ["50 degrees C", "75 degrees C", "90 degrees C",
                 "100 degrees C"], "a": 3},
    {"q": "Which of these is a fruit?",
     "choices": ["Carrot", "Potato", "Apple", "Onion"], "a": 2},
    {"q": "What is the capital of Japan?",
     "choices": ["Beijing", "Seoul", "Bangkok", "Tokyo"], "a": 3},
    {"q": "How many legs does a spider have?",
     "choices": ["6", "8", "10", "12"], "a": 1},
    {"q": "Which planet is closest to the sun?",
     "choices": ["Mercury", "Venus", "Earth", "Mars"], "a": 0},
    {"q": "What is 20 divided by 4?",
     "choices": ["4", "6", "5", "8"], "a": 2},
    {"q": "Which of these is a mammal?",
     "choices": ["Shark", "Eagle", "Whale", "Lizard"], "a": 2},
    {"q": "What is the opposite of 'hot'?",
     "choices": ["Warm", "Cold", "Cool", "Mild"], "a": 1},
    {"q": "How many sides does a triangle have?",
     "choices": ["4", "5", "6", "3"], "a": 3},
    {"q": "What do bees produce?",
     "choices": ["Honey", "Milk", "Silk", "Wax only"], "a": 0},
    {"q": "Which instrument has keys, pedals, and strings?",
     "choices": ["Guitar", "Violin", "Piano", "Drum"], "a": 2},
    {"q": "What is the capital of Canada?",
     "choices": ["Toronto", "Vancouver", "Montreal", "Ottawa"], "a": 3},
    {"q": "If yesterday was Wednesday, what day is tomorrow?",
     "choices": ["Thursday", "Friday", "Saturday", "Tuesday"], "a": 1},
    {"q": "Which gas do plants absorb from the air?",
     "choices": ["Oxygen", "Nitrogen", "Hydrogen", "Carbon dioxide"], "a": 3},
    {"q": "What is the square root of 16?",
     "choices": ["4", "8", "2", "16"], "a": 0},
    {"q": "Which is the fastest land animal?",
     "choices": ["Lion", "Cheetah", "Horse", "Greyhound"], "a": 1},
    {"q": "How many months have 31 days?",
     "choices": ["5", "6", "7", "8"], "a": 2},
    {"q": "What is the main language spoken in Brazil?",
     "choices": ["Spanish", "English", "French", "Portuguese"], "a": 3},
]

# fixed reference passage for the perplexity canary — plain, factual,
# and stable so ppl deltas reflect the edit, not the text
PPL_TEXT = (
    "The water cycle describes how water moves continuously through "
    "Earth's atmosphere, surface, and underground. Water evaporates from "
    "oceans, lakes, and rivers, rises into the atmosphere, and cools to "
    "form clouds through condensation. When droplets in clouds grow heavy "
    "enough, they fall as precipitation in the form of rain, snow, sleet, "
    "or hail. Some of this water flows into streams and rivers, returning "
    "to the oceans, while some soaks into the ground, replenishing "
    "groundwater supplies. Plants also release water vapor into the air "
    "through a process called transpiration. This cycle is driven by "
    "energy from the sun and is essential for distributing fresh water "
    "across the planet, regulating temperature, and supporting all forms "
    "of life."
)


class EvalProber:
    def __init__(self, model, tokenizer):
        self.model, self.tok = model, tokenizer
        self.history: List[Dict[str, Any]] = []

    def _dev(self):
        return next(self.model.parameters()).device

    # ---------- MC accuracy ----------

    def _mc_prompt(self, q: str, choices: Sequence[str]) -> str:
        lines = [f"Question: {q}"]
        lines += [f"{LETTERS[i]}) {c}" for i, c in enumerate(choices)]
        lines.append("Answer:")
        return "\n".join(lines)

    def mc_accuracy(self, probes: Optional[Sequence[Dict]] = None,
                    detail: bool = False) -> Dict[str, Any]:
        probes = list(probes) if probes else PROBES
        opt_ids = [
            self.tok(f" {L}", add_special_tokens=False).input_ids[0]
            for L in LETTERS
        ]
        correct, rows = 0, []
        with torch.no_grad():
            for p in probes:
                ids = torch.tensor(
                    [self.tok(self._mc_prompt(p["q"], p["choices"]),
                              add_special_tokens=False).input_ids],
                    device=self._dev())
                lp = self.model(ids).logits[0, -1].log_softmax(-1)
                scores = [float(lp[opt_ids[i]])
                          for i in range(len(p["choices"]))]
                pick = max(range(len(scores)), key=lambda i: scores[i])
                ok = pick == p["a"]
                correct += ok
                if detail:
                    rows.append({"q": p["q"], "pick": LETTERS[pick],
                                 "gold": LETTERS[p["a"]], "ok": ok})
        acc = correct / len(probes)
        out: Dict[str, Any] = {"accuracy": acc, "correct": correct,
                               "n": len(probes)}
        if detail:
            out["rows"] = rows
        return out

    # ---------- perplexity canary ----------

    def perplexity(self, text: Optional[str] = None) -> float:
        text = text or PPL_TEXT
        ids = torch.tensor(
            [self.tok(text, add_special_tokens=False).input_ids],
            device=self._dev())
        ctx = int(getattr(self.model.config, "n_positions",
                          getattr(self.model.config, "max_position_embeddings",
                                  1024)))
        ids = ids[:, :ctx]
        with torch.no_grad():
            logits = self.model(ids).logits[:, :-1].float()
            targets = ids[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return float(torch.exp(loss))

    # ---------- combined ----------

    def evaluate(self, label: str = "", probes=None,
                 ppl_text=None) -> Dict[str, Any]:
        mc = self.mc_accuracy(probes)
        ppl = self.perplexity(ppl_text)
        rec = {
            "label": label or f"eval{len(self.history)}",
            "t": round(time.time(), 1),
            "mc_acc": round(mc["accuracy"], 4),
            "mc_n": mc["n"],
            "ppl": round(ppl, 3),
        }
        self.history.append(rec)
        return rec

    def history_text(self) -> str:
        if not self.history:
            return "No evals yet."
        lines = [f"{'label':<16} {'mc_acc':>8} {'ppl':>8}"]
        for r in self.history:
            lines.append(f"{r['label']:<16.16} {r['mc_acc']:>8.3f} "
                         f"{r['ppl']:>8.2f}")
        return "\n".join(lines)


def load_probes_jsonl(path: str) -> List[Dict[str, Any]]:
    """Optional custom probe set: one JSON per line with keys
    q, choices (list), a (int index)."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
