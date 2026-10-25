import json
import re


def parse_json(text):
    """First JSON object in text (code fences tolerated), else None."""
    t = re.sub(r"```(?:json)?", "", text or "")
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def fallback_rating(text, lo=0, hi=10):
    """Pull 'rating: 7' / 'score = 3' out of unstructured text; None if absent."""
    m = re.search(r"(?:rating|score)\D{0,12}(\d+)", text or "", re.I)
    if not m:
        return None
    return max(lo, min(hi, int(m.group(1))))


def complete(client, model, prompt, temperature=0.0):
    r = client.chat.completions.create(model=model, temperature=temperature,
                                       messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content
