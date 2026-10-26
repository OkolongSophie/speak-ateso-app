"""Scoring helpers for the Ateso answer checker (shared by the notebook and the app)."""
import re, difflib
from collections import Counter
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

def norm(s):
    s = re.sub(r"[^\w\s']", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()

def toks(s):
    return norm(s).split()

def chrf(hyp, ref, n=6, beta=2):
    """chrF: character n-gram F-score (Popovic, 2015), on text without spaces, 0-100."""
    h, r = norm(hyp).replace(" ", ""), norm(ref).replace(" ", "")
    if not h or not r:
        return 100.0 if h == r else 0.0
    precs, recs = [], []
    for k in range(1, n + 1):
        hg = Counter(h[i:i + k] for i in range(len(h) - k + 1))
        rg = Counter(r[i:i + k] for i in range(len(r) - k + 1))
        if not hg or not rg:
            continue
        match = sum((hg & rg).values())
        precs.append(match / sum(hg.values())); recs.append(match / sum(rg.values()))
    p, rc = sum(precs) / len(precs), sum(recs) / len(recs)
    if p == 0 and rc == 0:
        return 0.0
    return 100 * (1 + beta**2) * p * rc / (beta**2 * p + rc)

FEATURE_NAMES = ["exact", "chrf", "char_ratio", "token_sort_ratio", "ref_len", "hyp_len", "len_ratio",
                 "word_edits", "word_edit_rate", "missing_words", "extra_words", "shared_word_share",
                 "same_words_other_order", "english_words", "best_sub_similarity", "prefix_only_change"]

def _substitutions(r, h):
    sm = difflib.SequenceMatcher(a=r, b=h, autojunk=False)
    return [(r[i1:i2], h[j1:j2]) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag == "replace"]

def features(prompt, reference, learner):
    r, h = toks(reference), toks(learner)
    rn, hn = " ".join(r), " ".join(h)
    en = set(toks(prompt)) - set(r)
    rc, hc = Counter(r), Counter(h)
    edits = Levenshtein.distance(r, h)
    subs = [(a[0], b[0]) for a, b in _substitutions(r, h) if len(a) == 1 and len(b) == 1]
    best = max((fuzz.ratio(a, b) for a, b in subs), default=0.0)
    prefix = int(any(len(a) > 1 and a[1:] == b[1:] and a[0] != b[0] for a, b in subs))
    return {
        "exact": int(rn == hn),
        "chrf": chrf(learner, reference),
        "char_ratio": fuzz.ratio(rn, hn),
        "token_sort_ratio": fuzz.token_sort_ratio(rn, hn),
        "ref_len": len(r),
        "hyp_len": len(h),
        "len_ratio": len(h) / max(len(r), 1),
        "word_edits": edits,
        "word_edit_rate": edits / max(len(r), 1),
        "missing_words": sum((rc - hc).values()),
        "extra_words": sum((hc - rc).values()),
        "shared_word_share": sum((rc & hc).values()) / max(len(r), 1),
        "same_words_other_order": int(rc == hc and r != h),
        "english_words": sum(1 for w in h if w in en),
        "best_sub_similarity": best,
        "prefix_only_change": prefix,
    }

def diagnose(prompt, reference, learner):
    """Name the main mistake by lining up learner words against reference words."""
    r, h = toks(reference), toks(learner)
    if r == h:
        return "none"
    if Counter(r) == Counter(h):
        return "word_order"
    shared = sum((Counter(r) & Counter(h)).values()) / max(len(r), 1)
    sm = difflib.SequenceMatcher(a=r, b=h, autojunk=False)
    ops = [op for op in sm.get_opcodes() if op[0] != "equal"]
    if shared < 0.3:
        return "unrelated_sentence"
    if len(ops) > 1:
        return "multiple_errors"
    tag, i1, i2, j1, j2 = ops[0]
    if tag == "delete" and i2 - i1 == 1:
        return "missing_word"
    if tag == "insert" and j2 - j1 == 1:
        return "extra_word"
    if tag == "replace" and i2 - i1 == 1 and j2 - j1 == 1:
        a, b = r[i1], h[j1]
        if b in set(toks(prompt)) - set(r):
            return "english_word"
        if len(a) > 1 and a[1:] == b[1:]:
            return "wrong_prefix"
        if Levenshtein.distance(a, b) <= 2 or fuzz.ratio(a, b) >= 75:
            return "spelling"
        return "wrong_word"
    return "multiple_errors"

FEEDBACK = {
    "none": "Correct!",
    "spelling": "Check the spelling of one word.",
    "missing_word": "A word is missing.",
    "extra_word": "There is an extra word.",
    "word_order": "The words are right but in the wrong order.",
    "wrong_word": "One word is wrong.",
    "english_word": "You used an English word. Try the Ateso word.",
    "wrong_prefix": "Check the beginning (prefix) of one word.",
    "multiple_errors": "Several things are wrong.",
    "unrelated_sentence": "This does not match the sentence.",
}
