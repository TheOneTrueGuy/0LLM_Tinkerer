"""
Closed-form fact writing — teach the model a fact by solving for a weight
change directly (MEMIT-style, no gradients). Ported and generalized from
the method documented in areu01or00/qwen0.8B-wsp, which is itself in the
ROME/MEMIT family (Meng et al.). Reimplemented here against our
architecture-agnostic explorer so it works on GPT-2 (Conv1D c_proj) as
well as Llama/Qwen-family down_proj.

For one fact, given several question phrasings + a target answer:

  keys   k : inputs to each edited down-proj at positions where the model
             is about to emit an answer token (one teacher-forced pass)
  values v : residual-stream delta that raises that token's logit by its
             shortfall + 3 nats, along the direction the output head reads
  protect  : key covariance of ~40 general Q&As (cached, recorded once),
             the fact's own question tokens, contrast questions, and
             previously-written facts
  solve    : dW_x = (Kp @ share^T) @ M^-1, M = lam*C + KpKp^T + mu*I,
             spread bottom-up over `spread` layers, capped at
             max_rel * ||W||, then refined per-token over `rounds`

Deltas live in x-space (out += x @ delta_x) so Linear and Conv1D differ
only at commit time. The edit can be toggled for paired generate and each
fact is individually undoable (deltas are additive — subtract the stored
dW exactly).
"""

import hashlib
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

# general questions whose answers must not change; the model's own
# answers are recorded on first use (cached per model+layers)
PROTECT = [
    "What is the capital of France?", "What is the capital of Spain?",
    "What is the capital of Germany?", "What is the capital of India?",
    "Who wrote Romeo and Juliet?", "Who wrote the novel 1984?",
    "Who painted Starry Night?", "Who discovered penicillin?",
    "Who invented the telephone?", "Who invented the light bulb?",
    "Who founded Microsoft?", "Who created Linux?",
    "What is the chemical formula of water?", "What is the largest ocean?",
    "What is the largest mammal?", "How many days are in a leap year?",
    "What is the smallest prime number?", "What is 12 times 8?",
    "What does CPU stand for?", "What is TensorFlow?",
    "What language is spoken in Argentina?", "What color is a flamingo?",
    "What is the boiling point of ethanol?",
    "What is the largest moon of Saturn?",
    "When was the Eiffel Tower finished?",
    "What is the national dish of Spain?",
    "What instrument does Yo-Yo Ma play?", "What does HTTPS encrypt?",
    "What is photosynthesis?", "What is a noun?",
    "Who was Albert Einstein?", "What is machine learning?",
    "Hi, how are you?", "Tell me a joke.", "Write a haiku about rain.",
    "Give me a tip for sleeping better.",
    "Write a Python function that reverses a string.",
    "Translate 'thank you' into French.",
    "Summarize the plot of Cinderella in one sentence.",
    "What is model editing in machine learning?",
]


class FactEditor:
    def __init__(self, model, tokenizer, layer: Optional[int] = None,
                 spread: int = 5, lam: float = 3.0, mu: float = 1e-2,
                 max_rel: float = 0.25, rounds: int = 6,
                 contrast_weight: float = 10.0, max_boost: float = 10.0):
        from neuron_explorer import NeuronExplorer

        self.model, self.tok = model, tokenizer
        self.lam, self.mu = lam, mu
        self.max_rel, self.rounds = max_rel, rounds
        self.cw, self.max_boost = contrast_weight, max_boost
        self.ex = NeuronExplorer(model, tokenizer)

        _, self.blocks = self.ex.find_block_stack()
        n = len(self.blocks)
        self.L = layer if layer is not None else min(n - 2, round(0.875 * n))
        self.Ls = list(range(max(0, self.L - spread + 1), self.L + 1))
        self.W = {}
        for l in self.Ls:
            _, mod = self.ex.find_down_proj(l)
            self.W[l] = mod
        self.Wn = {
            l: self.W[l].weight.float().norm().item() for l in self.Ls}
        self.head = model.get_output_embeddings().weight
        self.norm = self._find_final_norm()
        self.is_linear = {l: isinstance(m, torch.nn.Linear)
                          for l, m in self.W.items()}

        wL = self.W[self.L].weight
        d_model = wL.shape[0] if self.is_linear[self.L] else wL.shape[1]
        # effective per-dim gain of the final norm: its weight vector if it
        # has one (covers LayerNorm + RMSNorm), else norm(ones)
        if isinstance(getattr(self.norm, "weight", None), torch.Tensor):
            self.g = self.norm.weight.detach().float()
        else:
            self.g = self.norm(
                torch.ones(1, d_model, device=next(model.parameters()).device,
                         dtype=next(model.parameters()).dtype)
            )[0].float()

        self.end = self._end_token()
        self.stop = list({self.end, self.tok.eos_token_id} - {None})
        din = self._din()
        self.delta = {l: torch.zeros(din, d_model, dtype=torch.float32)
                      for l in self.Ls}  # x-space: out += x @ delta
        self.facts: List[Dict[str, Any]] = []
        self.active = True   # hook applies deltas when True (paired-generate)
        self.cap = False     # capturing keys during a read pass
        self.mod = False     # internal: generate with/without edit
        self.k, self.h, self.hL = {}, None, None

        for l in self.Ls:
            self.W[l].register_forward_hook(self._w_hook(l))
        self.blocks[self.L].register_forward_hook(self._top_hook)
        self.norm.register_forward_pre_hook(self._norm_hook)
        logger.info(f"FactEditor: editing down-proj of layers "
                    f"{self.Ls[0]}-{self.Ls[-1]} ({len(self.Ls)} of {n})")
        self._protection_cache = None

    # ---------- plumbing ----------

    def _find_final_norm(self):
        cands = []
        for name, mod in self.model.named_modules():
            if name.count(".") <= 2 and re.search(
                    r"(ln_f|norm|final_layernorm|norm_f|ln)$", name):
                cands.append((len(name), name, mod))
        if not cands:
            raise RuntimeError("Could not locate the final norm module.")
        cands.sort()
        return cands[0][2]

    def _din(self) -> int:
        w = self.W[self.Ls[0]].weight
        return w.shape[0] if not self.is_linear[self.Ls[0]] else w.shape[1]

    def _end_token(self):
        try:
            full = self.tok.apply_chat_template(
                [{"role": "user", "content": "hi"},
                 {"role": "assistant", "content": "XQZ"}], tokenize=False)
            ids = self.tok(full[full.rindex("XQZ") + 3:],
                           add_special_tokens=False).input_ids
            if ids:
                return ids[0]
        except Exception:
            pass
        return self.tok.eos_token_id

    def _prompt(self, q: str) -> str:
        if getattr(self.tok, "chat_template", None):
            try:
                return self.tok.apply_chat_template(
                    [{"role": "user", "content": q}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                return self.tok.apply_chat_template(
                    [{"role": "user", "content": q}], tokenize=False,
                    add_generation_prompt=True)
        return f"Question: {q}\nAnswer: "

    def _enc(self, s: str):
        return self.tok(s, add_special_tokens=False).input_ids

    def _w_hook(self, l):
        def h(mod, inp, out):
            if self.cap:
                self.k[l] = inp[0].detach().float()
            if self.active or self.mod:
                d = self.delta[l].to(inp[0].device)
                if float(d.abs().max()) > 0:
                    return out + (inp[0].float() @ d).to(out.dtype)
        return h

    def _top_hook(self, mod, inp, out):
        if self.cap:
            self.hL = (out[0] if isinstance(out, tuple) else out) \
                .detach().float()

    def _norm_hook(self, mod, args):
        if self.cap:
            self.h = args[0].detach().float()

    # ---------- passes ----------

    def _device(self):
        return next(self.model.parameters()).device

    def answer(self, q: str, mod: bool, n: int = 80) -> str:
        prev, self.mod = self.active, mod
        self.active = False
        x = torch.tensor([self._enc(self._prompt(q))],
                         device=self._device())
        with torch.no_grad():
            out = self.model.generate(
                x, max_new_tokens=n, do_sample=False,
                eos_token_id=self.stop,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        self.mod, self.active = False, prev
        return self.tok.decode(out[0, x.shape[1]:],
                               skip_special_tokens=True).strip()

    def _gen_ids(self, q: str, n: int):
        prev, self.mod = self.active, True
        self.active = False
        x = torch.tensor([self._enc(self._prompt(q))],
                         device=self._device())
        with torch.no_grad():
            out = self.model.generate(
                x, max_new_tokens=n, do_sample=False,
                eos_token_id=self.stop,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        self.mod, self.active = False, prev
        return out[0, x.shape[1]:].tolist()

    def read(self, q: str, ans_ids: List[int], end: bool = False,
             mod: bool = False) -> Dict[str, Any]:
        """One teacher-forced pass: keys at answer/question positions,
        top-edited-layer output, target logits/logprobs, residual RMS."""
        p = self._enc(self._prompt(q))
        ids = p + list(ans_ids) + ([self.end] if end else [])
        x = torch.tensor([ids], device=self._device())
        prev_active, self.active = self.active, False
        self.mod, self.cap = mod, True
        with torch.no_grad():
            logits = self.model(input_ids=x).logits[0].float()
        self.mod = self.cap = False
        self.active = prev_active
        sl = slice(len(p) - 1, len(ids) - 1)
        tg = x[0, len(p):]
        lo = logits[sl]
        return {
            "K": {l: self.k[l][0, sl].double().cpu() for l in self.Ls},
            "Q": {l: self.k[l][0, :len(p) - 1].double().cpu()
                  for l in self.Ls},
            "hL": self.hL[0, sl].double().cpu(),
            "tg": tg.cpu(),
            "lg": lo.gather(1, tg[:, None])[:, 0].cpu(),
            "lp": lo.log_softmax(-1).gather(1, tg[:, None])[:, 0].cpu(),
            "r": self.h[0, sl].pow(2).mean(-1).sqrt().cpu(),
        }

    def _keys_all(self, q: str, ans_ids: List[int], mod: bool = False):
        r = self.read(q, ans_ids, mod=mod)
        return {l: torch.cat([r["Q"][l], r["K"][l]]).T for l in self.Ls}

    def _protection(self, progress=None):
        if self._protection_cache is not None:
            return self._protection_cache
        tag = hashlib.sha1(json.dumps(
            [self.model.config._name_or_path, self.Ls, PROTECT]
        ).encode()).hexdigest()[:10]
        cache_dir = os.path.join(os.path.expanduser("~"),
                                 ".cache", "llm_tinkerer_wsp")
        mname = re.sub(r"[^A-Za-z0-9.-]", "_",
                       self.model.config._name_or_path or "model")[-60:]
        path = os.path.join(cache_dir, f"{mname}-L{self.Ls[0]}-{self.L}-{tag}.pt")
        if os.path.exists(path):
            self._protection_cache = torch.load(path)
            return self._protection_cache
        din = self._din()
        C = {l: torch.zeros(din, din, dtype=torch.float64) for l in self.Ls}
        for i, q in enumerate(PROTECT):
            if progress:
                progress(i, len(PROTECT), q)
            K = self._keys_all(q, self._enc(self.answer(q, False, n=48)))
            for l in self.Ls:
                C[l] += K[l] @ K[l].T
        self._protection_cache = {l: C[l].float() for l in self.Ls}
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(self._protection_cache, path)
        return self._protection_cache

    # ---------- the write ----------

    def teach(self, asks: Sequence[str], answer: str,
              contrast: Sequence[str] = (),
              progress=None) -> Dict[str, Any]:
        """Write one fact. Every step is a forward pass or a linear solve."""
        t0 = time.time()
        if progress:
            progress(0, 1, "building protection cache (one-time, ~40 passes)")
        self.C = self._protection(
            lambda i, n, q: progress and progress(i, n, f"protect: {q[:40]}"))
        a_ids = self._enc(answer)
        target = list(a_ids) + [self.end]
        base = {l: self.delta[l].clone() for l in self.Ls}
        reads = [self.read(q, a_ids, end=True, mod=True) for q in asks]

        V, D, offs = [], [], [0]
        for r in reads:
            offs.append(offs[-1] + len(r["tg"]))
            for j in range(len(r["tg"])):
                d = (self.g * self.head[r["tg"][j]].float()).double().cpu()
                Dj = max(0.0, -r["lp"][j].item()) + 3.0
                V.append(Dj * r["r"][j].item() / d.dot(d) * d)
                D.append(Dj)
        V = torch.stack(V, 1)
        h0 = torch.cat([r["hL"] for r in reads]).T
        Qk = {l: torch.cat([r["Q"][l] for r in reads]).T for l in self.Ls}
        Kc = [self._keys_all(c, self._enc(self.answer(c, True, n=48)), mod=True)
              for c in contrast]

        Cn = {}
        for l in self.Ls:
            Cn[l] = self.lam * self.C[l].double()
            for k in [Qk[l]] + [k[l] for f in self.facts for k in f["keys"]]:
                Cn[l] += self.lam * (k @ k.T)
            for k in Kc:
                Cn[l] += self.lam * self.cw * (k[l] @ k[l].T)

        def write_all(c):
            """Bottom-up: each layer delivers its share of what's still
            missing at the top edited layer's output."""
            for l in self.Ls:
                self.delta[l] = base[l].clone()
            Kp_top = None
            for i, l in enumerate(self.Ls):
                rr = [self.read(q, a_ids, end=True, mod=True) for q in asks]
                Kp = torch.cat([r["K"][l] for r in rr]).T
                missing = V * c - (torch.cat([r["hL"] for r in rr]).T - h0)
                share = missing / (len(self.Ls) - i)
                M = Cn[l] + Kp @ Kp.T
                M += self.mu * M.diagonal().mean() * torch.eye(
                    M.shape[0], dtype=M.dtype)
                dW_x = torch.linalg.solve(M, Kp @ share.T).float()
                rel = dW_x.norm().item() / self.Wn[l]
                if rel > self.max_rel:
                    dW_x *= self.max_rel / rel
                # delta stays on CPU; _w_hook moves a copy to the input's
                # device per call, so moving it here mixes devices
                self.delta[l] = base[l] + dW_x
                if l == self.L:
                    Kp_top = Kp
            return Kp_top

        c = torch.ones(V.shape[1], dtype=torch.float64)
        write_all(c)
        got = sum((self.read(q, a_ids, end=True, mod=True)["lg"]
                   - r["lg"]).sum().item()
                  for q, r in zip(asks, reads))
        gain = got / sum(D)
        c *= min(8.0, max(0.5, 1.0 / gain)) if gain > 0 else 8.0
        c0 = c.clone()

        hist = []
        for rnd in range(self.rounds):
            write_all(c)
            p = torch.cat([self.read(q, a_ids, end=True, mod=True)["lp"]
                           for q in asks]).exp()
            mism = []
            for q in asks:
                g = self._gen_ids(q, len(target) + 2)
                mism.append(next(
                    (j for j in range(len(target))
                     if j >= len(g) or g[j] != target[j]), None))
            exact = sum(m is None for m in mism)
            rels = {l: round((self.delta[l].cpu() - base[l].cpu())
                             .norm().item() / self.Wn[l], 3)
                    for l in self.Ls}
            hist.append({
                "round": rnd,
                "tokens_above_50pct": round(
                    (p > 0.5).float().mean().item(), 3),
                "free_exact": f"{exact}/{len(asks)}",
                "first_miss": mism,
                "dW_rel_max": max(rels.values()),
            })
            if progress:
                progress(rnd, self.rounds,
                         f"round {rnd}: exact {exact}/{len(asks)}")
            if exact == len(asks):
                break
            c[p < 0.5] *= 1.5
            c[p < 0.2] *= 1.4
            c[p > 0.98] *= 0.85
            for i, m in enumerate(mism):
                if m is not None:
                    c[offs[i] + m] *= 2.0
            c.clamp_(max=c0 * self.max_boost)

        dW = {l: (self.delta[l].cpu() - base[l].cpu())
              for l in self.Ls}
        fin = [self.read(q, a_ids, end=True, mod=True) for q in asks]
        keys = ([{l: torch.cat([r["K"][l] for r in fin]).T for l in self.Ls}]
                + [Qk] + Kc)
        rel = max(dW[l].norm().item() / self.Wn[l] for l in self.Ls)
        fit = hist[-1]["tokens_above_50pct"]
        self.facts.append({
            "ask": list(asks), "answer": answer, "contrast": list(contrast),
            "dW": dW, "keys": keys, "gain": float(gain),
            "scale": float(c.mean()), "rel": float(rel), "fit": fit,
            "rounds": hist, "committed": False,
        })
        return {
            "seconds": round(time.time() - t0, 1),
            "gain": round(float(gain), 3),
            "dW_rel_max": round(rel, 4),
            "answer_tokens_above_50pct": fit,
            "free_exact": hist[-1]["free_exact"],
            "rounds": len(hist), "history": hist,
        }

    # ---------- eval / undo / commit ----------

    def compare(self, q: str, n: int = 60) -> Dict[str, str]:
        """Base vs edited answer for paired evaluation."""
        return {"base": self.answer(q, mod=False, n=n),
                "edited": self.answer(q, mod=True, n=n)}

    def undo_last(self) -> str:
        """Remove the most recent fact — exact, since deltas are additive."""
        if not self.facts:
            return "No facts to undo."
        f = self.facts.pop()
        for l in self.Ls:
            if f["committed"]:
                w = self.W[l].weight.data
                dW = f["dW"][l].to(w.device, w.dtype)
                w -= (dW.T if self.is_linear[l] else dW)
            else:
                self.delta[l] -= f["dW"][l].to(self.delta[l].device)
        return f"Undid fact: {f['answer'][:60]}"

    def commit(self) -> str:
        """Bake all pending deltas into the real weights and clear them."""
        any_pending = any(float(d.abs().max()) > 0 for d in self.delta.values())
        if not any_pending:
            return "No pending deltas to commit."
        with torch.no_grad():
            for l in self.Ls:
                w = self.W[l].weight.data
                d = self.delta[l].to(w.device, w.dtype)
                w += (d.T if self.is_linear[l] else d)
                self.delta[l].zero_()
        for f in self.facts:
            f["committed"] = True
        return "Committed fact deltas to weights."

    def describe(self) -> str:
        if not self.facts:
            return "No facts written."
        lines = []
        for i, f in enumerate(self.facts):
            mark = "committed" if f["committed"] else "pending"
            lines.append(
                f"{i}: '{f['answer'][:50]}' rel={f['rel']:.3f} "
                f"fit={f['fit']} [{mark}]")
        return "\n".join(lines)
