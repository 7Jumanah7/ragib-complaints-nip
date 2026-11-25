"""Text cleaning, Arabic normalisation and PII masking (Labs 1 and 4)."""
import re

EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}")
# Saudi mobile numbers: 05XXXXXXXX, +9665XXXXXXXX or 9665XXXXXXXX
PHONE_RE = re.compile(r"(?<!\d)(?:\+?966|0)5\d{8}(?!\d)")


def clean_spaces(text):
    """Collapse repeated whitespace and trim the ends."""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_arabic(text):
    """Unify alef forms, alef maqsura and ta marbuta.

    Useful for dialect detection and search matching. It is NOT applied before
    feeding text to CAMeLBERT, because that model was pre-trained on text that
    keeps these distinctions.
    """
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    return text


def mask_pii(text):
    """Replace emails and Saudi phone numbers with placeholders."""
    text = EMAIL_RE.sub("[EMAIL]", str(text))
    text = PHONE_RE.sub("[PHONE]", text)
    return text


def preprocess(text):
    """Standard pipeline used before tokenisation: clean spaces, then mask PII."""
    return mask_pii(clean_spaces(text))
