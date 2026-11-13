"""
VLM fact/association writing — the closed-form writer extended to
vision-language models. asks/contrast are (image, question) pairs; the
keys at answer positions then bind *visual context + question* to the
answer, so the write is a visual association, not just a text fact.

The vision encoder is never touched — the edit lives in the LM decoder's
down-projections, downstream of the projector, where cross-modal context
already lives. Multiple images of the same subject as asks is the
multi-view analogue of multi-phrasing generalization.

Subclass of FactEditor: inherits the whole solve/protect/refine stack;
only the I/O passes are overridden for multimodal inputs.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from fact_editor import FactEditor

QImg = Tuple[Any, str]   # (PIL image or None, question text)


class VLMFactEditor(FactEditor):
    def __init__(self, model, tokenizer, processor, **kw):
        self.processor = processor
        super().__init__(model, tokenizer, **kw)
        self.img_tok = self._image_token_id()

    def _image_token_id(self):
        """Id of the image placeholder token, or None if the config has no
        such field (then no prompt positions are filtered)."""
        for attr in ("image_token_id", "image_token_index"):
            t = getattr(self.model.config, attr, None)
            if isinstance(t, int):
                return t
        return None

    # ---------- multimodal plumbing ----------

    def _prompt(self, q: str, image=None) -> str:
        proc = self.processor
        tmpl = getattr(proc, "chat_template", None) or getattr(
            self.tok, "chat_template", None)
        if tmpl:
            content = ([{"type": "image"}] if image is not None else []) + \
                      [{"type": "text", "text": q}]
            try:
                return proc.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        pre = "<image>\n" if image is not None else ""
        return f"{pre}Question: {q}\nAnswer: "

    def _mm_kwargs(self, image, qtext: str) -> Dict[str, torch.Tensor]:
        """Processor inputs for an (image, question) pair. The documented
        SmolVLM path is apply_chat_template with tokenize=True — it expands
        image placeholders and pixel tensors in one pass; the bare
        processor(text=, images=) call can't reconcile the two on 4.x."""
        content = ([{"type": "image", "image": image}]
                   if image is not None else []) + \
                  [{"type": "text", "text": qtext}]
        messages = [{"role": "user", "content": content}]
        kw = None
        try:
            kw = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt")
        except Exception:
            pass
        if kw is None or "input_ids" not in kw:
            prompt = self._prompt(qtext, image)
            kw = self.processor(
                text=prompt,
                images=[image] if image is not None else None,
                padding=True, return_tensors="pt")
        return {k: v.to(self._device()) if torch.is_tensor(v) else v
                for k, v in dict(kw).items()}

    def _split(self, q) -> QImg:
        """asks/contrast may be (image, q) tuples or bare strings."""
        if isinstance(q, (list, tuple)):
            return q[0], q[1]
        return None, q

    # ---------- overridden passes ----------

    def read(self, q, ans_ids: List[int], end: bool = False,
             mod: bool = False) -> Dict[str, Any]:
        img, qtext = self._split(q)
        kw = self._mm_kwargs(img, qtext)
        p_len = kw["input_ids"].shape[1]
        extra = torch.tensor(
            [list(ans_ids) + ([self.end] if end else [])],
            device=self._device())
        ids = torch.cat([kw.pop("input_ids"), extra], dim=1)
        kw["attention_mask"] = torch.ones_like(ids)
        prev_active, self.active = self.active, False
        self.mod, self.cap = mod, True
        with torch.no_grad():
            logits = self.model(input_ids=ids, **kw).logits[0].float()
        self.mod = self.cap = False
        self.active = prev_active
        sl = slice(p_len - 1, ids.shape[1] - 1)
        tg = ids[0, p_len:]
        lo = logits[sl]
        # Protected keys cover the question, so take text positions only.
        # Image placeholders hold no question content and the processor
        # emits one per patch: 1088 of a 1140-token SmolVLM prompt, 95%.
        # Weighting those at lam swamps the normal equations and starves
        # the solve; it saturates max_rel and degrades the model instead
        # of writing the fact.
        qi = torch.arange(p_len - 1, device=ids.device)
        if self.img_tok is not None:
            qi = qi[ids[0, :p_len - 1] != self.img_tok]
        return {
            "K": {l: self.k[l][0, sl].double().cpu() for l in self.Ls},
            "Q": {l: self.k[l][0, qi].double().cpu() for l in self.Ls},
            "hL": self.hL[0, sl].double().cpu(),
            "tg": tg.cpu(),
            "lg": lo.gather(1, tg[:, None])[:, 0].cpu(),
            "lp": lo.log_softmax(-1).gather(1, tg[:, None])[:, 0].cpu(),
            "r": self.h[0, sl].pow(2).mean(-1).sqrt().cpu(),
        }

    def answer(self, q, mod: bool, n: int = 80) -> str:
        img, qtext = self._split(q)
        kw = self._mm_kwargs(img, qtext)
        prev, self.mod = self.active, mod
        self.active = False
        with torch.no_grad():
            out = self.model.generate(
                **kw, max_new_tokens=n, do_sample=False,
                eos_token_id=self.stop,
                pad_token_id=self.tok.pad_token_id
                or self.tok.eos_token_id)
        self.mod, self.active = False, prev
        return self.tok.decode(out[0, kw["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip()

    def _gen_ids(self, q, n: int):
        img, qtext = self._split(q)
        kw = self._mm_kwargs(img, qtext)
        prev, self.mod = self.active, True
        self.active = False
        with torch.no_grad():
            out = self.model.generate(
                **kw, max_new_tokens=n, do_sample=False,
                eos_token_id=self.stop,
                pad_token_id=self.tok.pad_token_id
                or self.tok.eos_token_id)
        self.mod, self.active = False, prev
        return out[0, kw["input_ids"].shape[1]:].tolist()
