"""
Tasks for evaluating transformer models on character-level sequence problems.

A Task defines:
  - a vocabulary (list of single-character tokens)
  - a prompt length (number of input tokens)
  - an answer length (number of tokens to generate)
  - a generator that yields (prompt_str, answer_str) examples
  - encode/decode helpers between strings and integer token IDs

Add a new task by subclassing `Task` and registering it in `TASK_REGISTRY`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Example:
    prompt: str   # text shown to the model (e.g. "12345+67890=")
    answer: str   # tokens the model must generate (e.g. "080235")


class Task:
    name: str
    vocab: list[str]
    prompt_len: int
    answer_len: int

    def __init__(self):
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        self.itos = {i: c for i, c in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def seq_len(self) -> int:
        return self.prompt_len + self.answer_len

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def generate_examples(self, n: int, seed: int = 0) -> list[Example]:
        raise NotImplementedError

    def is_correct(self, prediction: str, target: str) -> bool:
        return prediction == target


class FiveDigitAdditionTask(Task):
    """Add two 5-digit non-negative integers.

    prompt: "AAAAA+BBBBB="    (12 tokens; leading zeros allowed)
    answer: "SSSSSS"          (6 tokens; left-padded with zeros)

    Example: "12345+67890=" -> "080235"  (since 12345+67890 = 80235)
    """

    name = "addition-five-digits"
    vocab = list("0123456789+=")
    prompt_len = 12   # 5 + 1 + 5 + 1
    answer_len = 6    # max sum is 199998 -> 6 digits

    def generate_examples(self, n: int, seed: int = None) -> list[Example]:
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        out = []
        for _ in range(n):
            a = rng.randint(0, 99999)
            b = rng.randint(0, 99999)
            prompt = f"{a:05d}+{b:05d}="
            answer = f"{a + b:06d}"
            out.append(Example(prompt=prompt, answer=answer))
        return out


class FiveDigitMultiplicationTask(Task):
    """Multiply two 5-digit non-negative integers.

    prompt: "AAAAA*BBBBB="    (12 tokens; leading zeros allowed)
    answer: "PPPPPPPPPP"      (10 tokens; left-padded with zeros)

    Example: "12345*67890=" -> "0838102050"  (since 12345*67890 = 838102050)
    """

    name = "multiplication-five-digits"
    vocab = list("0123456789*=")
    prompt_len = 12   # 5 + 1 + 5 + 1
    answer_len = 10   # max product is 99999*99999 = 9999800001 -> 10 digits

    def generate_examples(self, n: int, seed: int = None) -> list[Example]:
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        out = []
        for _ in range(n):
            a = rng.randint(0, 99999)
            b = rng.randint(0, 99999)
            prompt = f"{a:05d}*{b:05d}="
            answer = f"{a * b:010d}"
            out.append(Example(prompt=prompt, answer=answer))
        return out


class FiveDigitSortTask(Task):
    """Sort a string of 5 digits in ascending order.

    prompt: "DDDDD="     (6 tokens)
    answer: "SSSSS"      (5 tokens; sorted ascending)

    Example: "31415=" -> "11345"
    """

    name = "sort-five-digits"
    vocab = list("0123456789=")
    prompt_len = 6    # 5 + 1
    answer_len = 5

    def generate_examples(self, n: int, seed: int = None) -> list[Example]:
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        out = []
        for _ in range(n):
            digits = [rng.randint(0, 9) for _ in range(5)]
            prompt = "".join(str(d) for d in digits) + "="
            answer = "".join(str(d) for d in sorted(digits))
            out.append(Example(prompt=prompt, answer=answer))
        return out


class CountDigitTask(Task):
    """Count occurrences of a query digit in a string of 10 digits.

    The final digit before '=' is the query.

    prompt: "DDDDDDDDDDQ="  (12 tokens: 10 digits + query digit + '=')
    answer: "CC"            (2 tokens; count 00-10, left-padded)

    Example: "31415926533=" -> "02"  (digit '3' appears twice in "3141592653")
    """

    name = "digit-counting-10"
    vocab = list("0123456789=")
    prompt_len = 12   # 10 + 1 + 1
    answer_len = 2

    def generate_examples(self, n: int, seed: int = None) -> list[Example]:
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        out = []
        for _ in range(n):
            digits = [rng.randint(0, 9) for _ in range(10)]
            query = rng.randint(0, 9)
            digit_str = "".join(str(d) for d in digits)
            prompt = f"{digit_str}{query}="
            count = sum(1 for d in digits if d == query)
            answer = f"{count:02d}"
            out.append(Example(prompt=prompt, answer=answer))
        return out


class ParityTask(Task):
    """Compute parity (XOR) of a variable-length bit string, length 1-10.

    Inputs are left-padded with '_' to length 10.

    prompt: "__BBBBBBBB="  (11 tokens; left-padded bits + '=')
    answer: "P"            (1 token; '0' if even number of 1s, '1' if odd)

    Example: "____1011010=" -> "0"  (four 1s -> even parity)
    """

    name = "parity-upto10-bits"
    vocab = list("01=_")
    prompt_len = 11   # 10 + 1
    answer_len = 1

    def generate_examples(self, n: int, seed: int = None) -> list[Example]:
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        out = []
        for _ in range(n):
            length = rng.randint(1, 10)
            bits = [rng.randint(0, 1) for _ in range(length)]
            bit_str = "".join(str(b) for b in bits)
            prompt = bit_str.rjust(10, "_") + "="
            answer = str(sum(bits) % 2)
            out.append(Example(prompt=prompt, answer=answer))
        return out


class BooleanCircuitTask(Task):
    """Evaluate a boolean expression of 5 bits joined by 4 binary operators.

    Operators: '&' (AND), '|' (OR), '^' (XOR). Evaluated strictly left-to-right
    (no precedence), like a chain through a circuit.

    prompt: "BoBoBoBoB="   (10 tokens; 5 bits interleaved with 4 ops + '=')
    answer: "R"            (1 token; '0' or '1')

    Example: "1&0|1^1&0=" -> "0"
      ((((1&0)|1)^1)&0) = ((1|1)^1)&0 = (1^1)&0 = 0&0 = 0
    """

    name = "boolean-circuit-5-bits"
    vocab = list("01&|^=")
    prompt_len = 10   # 5 bits + 4 ops + '='
    answer_len = 1

    def generate_examples(self, n: int, seed: int = None) -> list[Example]:
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        ops = ["&", "|", "^"]
        out = []
        for _ in range(n):
            bits = [rng.randint(0, 1) for _ in range(5)]
            chosen_ops = [rng.choice(ops) for _ in range(4)]
            tokens = []
            for i, b in enumerate(bits):
                tokens.append(str(b))
                if i < 4:
                    tokens.append(chosen_ops[i])
            prompt = "".join(tokens) + "="
            result = bits[0]
            for i, op in enumerate(chosen_ops):
                rhs = bits[i + 1]
                if op == "&":
                    result = result & rhs
                elif op == "|":
                    result = result | rhs
                else:
                    result = result ^ rhs
            answer = str(result)
            out.append(Example(prompt=prompt, answer=answer))
        return out


class LinearInterpolationTask(Task):
    """Linear interpolation given two integer (x, y) pairs and a query x.

    The underlying line y = m*x + b uses integer slope m in [1, 5] and integer
    intercept b in [0, 9], so all y values are guaranteed integers in [0, 54].
    x values are single digits in [0, 9]; x1 != x2; query xq in [0, 9].

    prompt: "X1,Y1;X2,Y2;XQ="   (12 tokens; y values are 2 digits)
    answer: "YY"                (2 tokens)

    Example: m=2, b=3 -> "1,05;4,11;7=" -> "17"  (y = 2*7+3 = 17)
    """

    name = "linear-interpolation-two-points"
    vocab = list("0123456789,;=")
    prompt_len = 12   # 1 + 1 + 2 + 1 + 1 + 1 + 2 + 1 + 1 + 1
    answer_len = 2

    def generate_examples(self, n: int, seed: int = None) -> list[Example]:
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        out = []
        for _ in range(n):
            m = rng.randint(1, 5)
            b = rng.randint(0, 9)
            x1 = rng.randint(0, 9)
            x2 = rng.randint(0, 9)
            while x2 == x1:
                x2 = rng.randint(0, 9)
            xq = rng.randint(0, 9)
            y1 = m * x1 + b
            y2 = m * x2 + b
            yq = m * xq + b
            prompt = f"{x1},{y1:02d};{x2},{y2:02d};{xq}="
            answer = f"{yq:02d}"
            out.append(Example(prompt=prompt, answer=answer))
        return out


class WordReversalTask(Task):
    """Reverse the order of 3 fixed-length words separated by underscores.

    Each word is exactly 3 lowercase letters. Words are random letter triples
    (not real English), so the model cannot rely on linguistic priors.

    prompt: "abc_def_ghi="   (12 tokens)
    answer: "ghi_def_abc"    (11 tokens)

    Example: "cat_dog_pig=" -> "pig_dog_cat"
    """

    name = "word-reversal-3x3"
    vocab = list("abcdefghijklmnopqrstuvwxyz_=")
    prompt_len = 12   # 3 + 1 + 3 + 1 + 3 + 1
    answer_len = 11   # 3 + 1 + 3 + 1 + 3

    def generate_examples(self, n: int, seed: int = None) -> list[Example]:
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        letters = "abcdefghijklmnopqrstuvwxyz"
        out = []
        for _ in range(n):
            words = ["".join(rng.choice(letters) for _ in range(3)) for _ in range(3)]
            prompt = "_".join(words) + "="
            answer = "_".join(reversed(words))
            out.append(Example(prompt=prompt, answer=answer))
        return out


class GCDTask(Task):
    """Greatest common divisor of two 3-digit non-negative integers.

    Both operands are in [1, 999] (avoids the degenerate gcd(0, 0) case).
    Leading zeros are used to keep the format fixed-width.

    prompt: "AAA,BBB="   (8 tokens)
    answer: "GGG"        (3 tokens; gcd, left-padded with zeros)

    Example: "084,036=" -> "012"  (gcd(84, 36) = 12)
    """

    name = "gcd-three-digits"
    vocab = list("0123456789,=")
    prompt_len = 8    # 3 + 1 + 3 + 1
    answer_len = 3

    def generate_examples(self, n: int, seed: int = None) -> list[Example]:
        from math import gcd
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        out = []
        for _ in range(n):
            a = rng.randint(1, 999)
            b = rng.randint(1, 999)
            prompt = f"{a:03d},{b:03d}="
            answer = f"{gcd(a, b):03d}"
            out.append(Example(prompt=prompt, answer=answer))
        return out


class DecimalToBinaryTask(Task):
    """Convert a decimal integer in [0, 255] to its 8-bit binary representation.

    prompt: "DDD="       (4 tokens; decimal, left-padded with zeros)
    answer: "BBBBBBBB"   (8 tokens; binary, MSB first, zero-padded to 8 bits)

    Example: "042=" -> "00101010"  (42 in binary is 101010)
    """

    name = "decimal-to-binary-8bit"
    vocab = list("0123456789=")
    prompt_len = 4    # 3 + 1
    answer_len = 8

    def generate_examples(self, n: int, seed: int = None) -> list[Example]:
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        out = []
        for _ in range(n):
            x = rng.randint(0, 255)
            prompt = f"{x:03d}="
            answer = f"{x:08b}"
            out.append(Example(prompt=prompt, answer=answer))
        return out


class SentimentSST2Task(Task):
    """Binary sentiment classification on a small subset of SST-2.

    Sentences are lowercased, restricted to the task vocabulary (unknown
    characters become spaces), whitespace-collapsed, then truncated or
    right-padded with '_' to a fixed length. The model must emit a single
    token '0' (negative) or '1' (positive).

    prompt: "<sentence padded to 80 chars>="   (81 tokens)
    answer: "L"                                (1 token; '0' or '1')

    Data is loaded from the `stanfordnlp/sst2` HuggingFace dataset and
    cached locally by `datasets`. We keep a fixed in-memory subset of the
    train split (the first SUBSET_SIZE examples after filtering); seeded
    sampling draws from that pool.
    """

    name = "sentiment-sst2"
    vocab = list("abcdefghijklmnopqrstuvwxyz '_=01")
    prompt_len = 81   # 80 sentence chars + '='
    answer_len = 1

    SENTENCE_LEN = 80
    SUBSET_SIZE = 2000

    _pool: list[tuple[str, str]] | None = None

    @classmethod
    def _load_pool(cls) -> list[tuple[str, str]]:
        if cls._pool is not None:
            return cls._pool
        from datasets import load_dataset
        allowed = set("abcdefghijklmnopqrstuvwxyz '")
        ds = load_dataset("stanfordnlp/sst2", split="train")
        pool: list[tuple[str, str]] = []
        for ex in ds:
            text = ex["sentence"].lower()
            text = "".join(c if c in allowed else " " for c in text)
            text = " ".join(text.split())
            if not text:
                continue
            text = text[: cls.SENTENCE_LEN].ljust(cls.SENTENCE_LEN, "_")
            pool.append((text, str(ex["label"])))
            if len(pool) >= cls.SUBSET_SIZE:
                break
        cls._pool = pool
        return pool

    def generate_examples(self, n: int, seed: int = None) -> list[Example]:
        pool = self._load_pool()
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        if n <= len(pool):
            sampled = rng.sample(pool, n)
        else:
            sampled = [rng.choice(pool) for _ in range(n)]
        return [Example(prompt=t + "=", answer=a) for t, a in sampled]


def _clean_text(s: str, allowed: set[str], max_len: int) -> str:
    s = s.lower()
    s = "".join(c if c in allowed else " " for c in s)
    s = " ".join(s.split())
    return s[:max_len].ljust(max_len, "_")


class ParaphraseMRPCTask(Task):
    """Binary paraphrase detection on a small subset of GLUE/MRPC.

    Two sentences are lowercased, restricted to the task vocabulary, and
    each truncated/padded to SENT_LEN characters. They are joined with '|'
    and terminated with '='. The model emits '0' (not paraphrase) or '1'.

    prompt: "<s1 padded>|<s2 padded>="   (2*SENT_LEN + 2 tokens)
    answer: "L"                          (1 token; '0' or '1')

    Data is loaded from `nyu-mll/glue` (config `mrpc`), train split.
    """

    name = "paraphrase-mrpc"
    vocab = list("abcdefghijklmnopqrstuvwxyz '_|=01")
    SENT_LEN = 60
    prompt_len = 2 * SENT_LEN + 2   # s1 + '|' + s2 + '='
    answer_len = 1

    SUBSET_SIZE = 2000

    _pool: list[tuple[str, str, str]] | None = None

    @classmethod
    def _load_pool(cls) -> list[tuple[str, str, str]]:
        if cls._pool is not None:
            return cls._pool
        from datasets import load_dataset
        allowed = set("abcdefghijklmnopqrstuvwxyz '")
        ds = load_dataset("nyu-mll/glue", "mrpc", split="train")
        pool: list[tuple[str, str, str]] = []
        for ex in ds:
            s1 = _clean_text(ex["sentence1"], allowed, cls.SENT_LEN)
            s2 = _clean_text(ex["sentence2"], allowed, cls.SENT_LEN)
            label = str(ex["label"])
            if label not in ("0", "1"):
                continue
            pool.append((s1, s2, label))
            if len(pool) >= cls.SUBSET_SIZE:
                break
        cls._pool = pool
        return pool

    def generate_examples(self, n: int, seed: int = None) -> list[Example]:
        pool = self._load_pool()
        rng = random.Random(seed) if seed is not None else random.Random()
        if n <= len(pool):
            sampled = rng.sample(pool, n)
        else:
            sampled = [rng.choice(pool) for _ in range(n)]
        return [Example(prompt=f"{s1}|{s2}=", answer=lab) for s1, s2, lab in sampled]


class NLISNLITask(Task):
    """Three-way natural language inference on a small subset of SNLI.

    Premise and hypothesis are lowercased, restricted to the task vocabulary,
    and each truncated/padded to SENT_LEN characters. Joined with '|' and
    terminated with '='. The model emits a single digit:
      '0' = entailment, '1' = neutral, '2' = contradiction.

    Examples with no-consensus label (-1) are filtered out.

    prompt: "<premise padded>|<hypothesis padded>="   (2*SENT_LEN + 2 tokens)
    answer: "L"                                       (1 token; '0', '1', '2')

    Data is loaded from `stanfordnlp/snli`, train split.
    """

    name = "nli-snli"
    vocab = list("abcdefghijklmnopqrstuvwxyz '_|=012")
    SENT_LEN = 60
    prompt_len = 2 * SENT_LEN + 2
    answer_len = 1

    SUBSET_SIZE = 2000

    _pool: list[tuple[str, str, str]] | None = None

    @classmethod
    def _load_pool(cls) -> list[tuple[str, str, str]]:
        if cls._pool is not None:
            return cls._pool
        from datasets import load_dataset
        allowed = set("abcdefghijklmnopqrstuvwxyz '")
        ds = load_dataset("stanfordnlp/snli", split="train")
        pool: list[tuple[str, str, str]] = []
        for ex in ds:
            if ex["label"] not in (0, 1, 2):
                continue
            p = _clean_text(ex["premise"], allowed, cls.SENT_LEN)
            h = _clean_text(ex["hypothesis"], allowed, cls.SENT_LEN)
            pool.append((p, h, str(ex["label"])))
            if len(pool) >= cls.SUBSET_SIZE:
                break
        cls._pool = pool
        return pool

    def generate_examples(self, n: int, seed: int = None) -> list[Example]:
        pool = self._load_pool()
        rng = random.Random(seed) if seed is not None else random.Random()
        if n <= len(pool):
            sampled = rng.sample(pool, n)
        else:
            sampled = [rng.choice(pool) for _ in range(n)]
        return [Example(prompt=f"{p}|{h}=", answer=lab) for p, h, lab in sampled]


class EditDistanceTask(Task):
    """Levenshtein edit distance between two 8-letter strings over {a, b, c, d}.

    prompt: "AAAAAAAA,BBBBBBBB="   (18 tokens)
    answer: "D"                    (1 token; distance 0-8)

    The second string is the first after 0-8 random insertions, deletions and substitutions
    (kept at length 8), so distances are spread out instead of concentrated near 5.

    Example: "abcdabcd,abddabca=" -> "2"
    """

    name = "edit-distance-8"
    vocab = list("abcd,=012345678")
    prompt_len = 18
    answer_len = 1

    @staticmethod
    def distance(a: str, b: str) -> int:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    def generate_examples(self, n: int, seed: int = None) -> list[Example]:
        rng = random.Random(seed) if seed is not None else random.Random()
        out = []
        for _ in range(n):
            a = "".join(rng.choice("abcd") for _ in range(8))
            b = list(a)
            for _ in range(rng.randint(0, 8)):
                op, i = rng.choice("ids"), rng.randrange(8)
                if op == "s":
                    b[i] = rng.choice("abcd")
                elif op == "d":
                    del b[i]
                    b.append(rng.choice("abcd"))
                else:
                    b.insert(i, rng.choice("abcd"))
                    b.pop()
            b = "".join(b)
            out.append(Example(prompt=f"{a},{b}=", answer=str(self.distance(a, b))))
        return out


class SmallestPrimeFactorTask(Task):
    """Smallest prime factor of an integer below 100000 (a prime is its own smallest factor).

    prompt: "NNNNN="   (6 tokens; left-padded with zeros)
    answer: "SSSSS"    (5 tokens; left-padded with zeros)

    One third of the inputs are prime; the rest are composites whose smallest prime factor is drawn
    uniformly from the primes up to 313, so always answering 2 scores near chance.

    Example: "03127=" -> "00053"  (3127 = 53 * 59)
    """

    name = "smallest-prime-factor"
    vocab = list("0123456789=")
    prompt_len = 6
    answer_len = 5

    @staticmethod
    def spf_table(limit: int = 100000) -> list[int]:
        spf = list(range(limit))
        for i in range(2, int(limit ** 0.5) + 1):
            if spf[i] == i:
                for j in range(i * i, limit, i):
                    if spf[j] == j:
                        spf[j] = i
        return spf

    def generate_examples(self, n: int, seed: int = None) -> list[Example]:
        rng = random.Random(seed) if seed is not None else random.Random()
        spf = self.spf_table()
        primes = [i for i in range(2, 100000) if spf[i] == i]
        by_factor: dict[int, list[int]] = {}
        for i in range(4, 100000):
            if spf[i] != i:
                by_factor.setdefault(spf[i], []).append(i)
        small = sorted(by_factor)
        out = []
        for k in range(n):
            x = rng.choice(primes) if rng.random() < 1 / 3 else rng.choice(by_factor[rng.choice(small)])
            out.append(Example(prompt=f"{x:05d}=", answer=f"{spf[x]:05d}"))
        return out


class TRECCoarseTask(Task):
    """Six-way question classification on a small subset of TREC (coarse labels).

    Questions are lowercased, restricted to the task vocabulary (unknown characters become spaces),
    and truncated or padded with '_' to 80 characters.

    prompt: "<question padded to 80 chars>="   (81 tokens)
    answer: "L"                                (1 token; 0 ABBR, 1 ENTY, 2 DESC, 3 HUM, 4 LOC, 5 NUM)
    """

    name = "trec-coarse"
    vocab = list("abcdefghijklmnopqrstuvwxyz '_=012345")
    prompt_len = 81
    answer_len = 1

    SENTENCE_LEN = 80
    SUBSET_SIZE = 2000

    _pool: list[tuple[str, str]] | None = None

    @classmethod
    def _load_pool(cls) -> list[tuple[str, str]]:
        if cls._pool is not None:
            return cls._pool
        from datasets import load_dataset
        allowed = set("abcdefghijklmnopqrstuvwxyz '")
        ds = load_dataset("CogComp/trec", split="train", revision="refs/convert/parquet")
        pool: list[tuple[str, str]] = []
        for ex in ds:
            text = _clean_text(ex["text"], allowed, cls.SENTENCE_LEN)
            pool.append((text, str(ex["coarse_label"])))
            if len(pool) >= cls.SUBSET_SIZE:
                break
        cls._pool = pool
        return pool

    def generate_examples(self, n: int, seed: int = None) -> list[Example]:
        pool = self._load_pool()
        rng = random.Random(seed) if seed is not None else random.Random()
        sampled = rng.sample(pool, n) if n <= len(pool) else [rng.choice(pool) for _ in range(n)]
        return [Example(prompt=t + "=", answer=a) for t, a in sampled]


class DateNormalizationTask(Task):
    """Rewrite a date written in one of nine common formats as YYYY-MM-DD.

    Years run from 1900 to 2099. Slash dates are always month first, and two-digit years mean 20YY.

    prompt: "<date padded to 31 chars>="   (32 tokens)
    answer: "YYYY-MM-DD"                   (10 tokens)

    Example: "march 3rd 2021_________________=" -> "2021-03-03"
    """

    name = "date-normalization"
    vocab = list("abcdefghijklmnopqrstuvwxyz0123456789 ,./-_=")
    prompt_len = 32
    answer_len = 10

    MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
              "september", "october", "november", "december"]
    TEXT_LEN = 31

    @staticmethod
    def ordinal(d: int) -> str:
        suffix = "th" if 11 <= d <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")
        return f"{d}{suffix}"

    def generate_examples(self, n: int, seed: int = None) -> list[Example]:
        import calendar
        rng = random.Random(seed) if seed is not None else random.Random()
        out = []
        for _ in range(n):
            fmt = rng.randrange(9)
            y = rng.randint(2000, 2099) if fmt == 8 else rng.randint(1900, 2099)
            m = rng.randint(1, 12)
            d = rng.randint(1, calendar.monthrange(y, m)[1])
            name, abbr = self.MONTHS[m - 1], self.MONTHS[m - 1][:3]
            text = [
                f"{name} {self.ordinal(d)} {y}",
                f"{name} {d}, {y}",
                f"{d} {name} {y}",
                f"the {self.ordinal(d)} of {name}, {y}",
                f"{abbr}. {d}, {y}",
                f"{d}-{abbr}-{y}",
                f"{m}/{d}/{y}",
                f"{y}.{m:02d}.{d:02d}",
                f"{m:02d}/{d:02d}/{y % 100:02d}",
            ][fmt]
            out.append(Example(prompt=text.ljust(self.TEXT_LEN, "_") + "=", answer=f"{y:04d}-{m:02d}-{d:02d}"))
        return out


TASK_REGISTRY: dict[str, type[Task]] = {
    FiveDigitAdditionTask.name: FiveDigitAdditionTask,
    FiveDigitMultiplicationTask.name: FiveDigitMultiplicationTask,
    FiveDigitSortTask.name: FiveDigitSortTask,
    CountDigitTask.name: CountDigitTask,
    ParityTask.name: ParityTask,
    BooleanCircuitTask.name: BooleanCircuitTask,
    LinearInterpolationTask.name: LinearInterpolationTask,
    WordReversalTask.name: WordReversalTask,
    GCDTask.name: GCDTask,
    DecimalToBinaryTask.name: DecimalToBinaryTask,
    SentimentSST2Task.name: SentimentSST2Task,
    ParaphraseMRPCTask.name: ParaphraseMRPCTask,
    NLISNLITask.name: NLISNLITask,
    EditDistanceTask.name: EditDistanceTask,
    SmallestPrimeFactorTask.name: SmallestPrimeFactorTask,
    TRECCoarseTask.name: TRECCoarseTask,
    DateNormalizationTask.name: DateNormalizationTask,
}


def get_task(name: str) -> Task:
    if name not in TASK_REGISTRY:
        raise KeyError(f"Unknown task '{name}'. Available: {sorted(TASK_REGISTRY)}")
    return TASK_REGISTRY[name]()

if __name__ == '__main__':
    print(TASK_REGISTRY.keys())