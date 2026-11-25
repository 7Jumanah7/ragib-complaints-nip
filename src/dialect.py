"""Rule-based dialect audit and clitic segmentation (Lab 4).

These are simple word-list / rule heuristics for demonstration, not trained
models. They will miss dialect words that are not in the lists.
"""
import re

from .preprocessing import normalize_arabic

MARKERS = {
    "Saudi White": ["وش", "ايش", "أبي", "أبغى", "ابغى", "الحين", "للحين", "كذا", "طيب", "ليه"],
    "Lebanese": ["شو", "هلق", "بدي", "منيح", "هيك", "كتير", "ليك", "مبارح"],
}


def detect_dialect(text):
    """Return the dialect with the most marker words, or 'Unknown'."""
    clean = normalize_arabic(re.sub(r"[؟?!.,،]", " ", text))
    padded = " " + " ".join(clean.split()) + " "
    scores = {
        dialect: sum(1 for w in words if " " + normalize_arabic(w) + " " in padded)
        for dialect, words in MARKERS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Unknown"


def segment(word):
    """Split attached Arabic clitics (و/ف، ب/ل/ك، ال) with simple rules.

    Known limitation: words that start with these letters naturally
    (for example 'بطاقتي') are split incorrectly.
    """
    parts, w = [], word
    if w[0] in "وف" and len(w) > 3:
        parts.append(w[0])
        w = w[1:]
    if w.startswith("لل") and len(w) > 3:
        parts += ["ل", "ال"]
        w = w[2:]
    else:
        if w[0] in "بلك" and len(w) > 3:
            parts.append(w[0])
            w = w[1:]
        if w.startswith("ال") and len(w) > 3:
            parts.append("ال")
            w = w[2:]
    parts.append(w)
    return parts
