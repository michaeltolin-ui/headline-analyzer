# streamlit_app_distil.py
# MVP: DistilRoBERTa (financial sentiment) + deep trader-style rule boosts + highlights + sector tag
# Free to run (no paid API). Great fit for Streamlit Community Cloud.

import re
import torch
import streamlit as st
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------- Page ----------------
st.set_page_config(page_title="Headline Analyzer (Distil Financial + Rules)", page_icon="🧠", layout="centered")
st.title("🧠 AI Headline Analyzer — Distilled Financial Model + Rule Boosts")
st.caption("Paste a market/financial headline → model + trader-phrase rules return Bullish/Bearish/Neutral, confidence, sector, and highlights.")

# ---------------- Load distilled financial model ----------------
@st.cache_resource(show_spinner=True)
def load_model():
    """
    Loads a small financial-sentiment model.
    Model: DistilRoBERTa fine-tuned on financial news sentiment.
    (Fallback-friendly; CPU OK for headlines.)
    """
    model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    id2label = mdl.config.id2label
    label2id = mdl.config.label2id
    # Most financial sentiment models use: 0=negative, 1=neutral, 2=positive (but we rely on id2label just in case)
    return tok, mdl, id2label, label2id

tokenizer, model, id2label, label2id = load_model()

# ---------------- Sector detection ----------------
SECTOR_MAP = {
    "Tech": [r"\bAI\b", r"\bchip(s)?\b", r"\bsemiconductor(s)?\b", r"\bGPU\b", r"\bsoftware\b", r"\bcloud\b", r"\bsmartphone(s)?\b"],
    "Consumer": [r"\bretail\b", r"\bconsumer\b", r"\bstore(s)?\b", r"\bapparel\b", r"\bgrocery\b", r"\be-commerce\b"],
    "Auto": [r"\bEV\b", r"\bvehicle(s)?\b", r"\bautomaker(s)?\b", r"\bdealership(s)?\b", r"\bauto(s)?\b"],
    "Energy": [r"\boil\b", r"\bWTI\b", r"\bnatural gas\b", r"\bOPEC\b", r"\brefiner(y|ies)\b", r"\bBrent\b"],
    "Financials": [r"\bbank(s)?\b", r"\binsurer(s)?\b", r"\bcredit\b", r"\bbroker(s)?\b", r"\bfintech\b",
                   r"\bcentral bank\b", r"\bFederal Reserve\b", r"\bFed\b", r"\bECB\b", r"\bBank of England\b"],
    "Healthcare": [r"\bFDA\b", r"\btrial\b", r"\bphase (II|III)\b", r"\bdrug\b", r"\btherapy\b", r"\bbiotech\b"],
    "Industrial": [r"\bmanufactur(e|ing)\b", r"\bsupply chain\b", r"\blogistics\b", r"\bfreight\b", r"\baerospace\b"],
    "Communications": [r"\bmedia\b", r"\bstreaming\b", r"\badvertising\b", r"\bsocial\b", r"\btelecom\b"],
}

def detect_sector(text: str) -> str:
    t = text.lower()
    for sector, patterns in SECTOR_MAP.items():
        for pat in patterns:
            if re.search(pat, t, flags=re.IGNORECASE):
                return sector
    return "General"

# ---------------- Deep trader-style boosts ----------------
# Heavily used action words in headlines; we weight them modestly so the model remains primary.
# You can tune weights later (+/- 0.05 to 0.40 typical). We keep them small but meaningful.

BULLISH_PATTERNS = {
    # price/action
    r"\bsoar(s|ed|ing)?\b": 0.30, r"\bsurge(s|d|ing)?\b": 0.28, r"\bjump(s|ed|ing)?\b": 0.26,
    r"\brally(ies|ing)?\b": 0.26, r"\bpop(s|ped|ping)?\b": 0.20, r"\bspike(s|d|ing)?\b": 0.28,
    r"\bskyrocket(s|ed|ing)?\b": 0.35, r"\bexplode(s|d|ing)?\b": 0.35, r"\bclimb(s|ed|ing)?\b": 0.18,
    r"\brebound(s|ed|ing)?\b": 0.18, r"\brecover(s|ed|ing)?\b": 0.16,
    # fundamentals / guidance
    r"\bbeat(s|en)?\b": 0.22, r"\btops?\b": 0.20, r"\bcrush(es|ed|ing)?\b (estimates|EPS|revenue)": 0.30,
    r"\braise(s|d|ing)? (guidance|outlook|forecast)\b": 0.32, r"\blift(s|ed|ing)? (guidance|outlook|forecast)\b": 0.28,
    r"\bupgrade(s|d)?\b": 0.22, r"\bprice target (raised|hiked|lifted)\b": 0.24, r"\binitiates?\b (buy|overweight)\b": 0.26,
    r"\bstrong demand\b": 0.26, r"\brecord (sales|profit|revenue)\b": 0.24,
    # corporate actions / catalysts
    r"\bapprov(al|ed|es)\b": 0.24, r"\bFDA (approves|approval)\b": 0.28, r"\bwins?\b (contract|order)\b": 0.22,
    r"\bpartnership\b": 0.18, r"\bbuyback(s)?\b": 0.22, r"\bdividend (hike|increase|raised)\b": 0.22,
    r"\bacquisition\b": 0.16, r"\bmerger\b": 0.12, r"\bexpands?\b": 0.12, r"\bsecur(es|ed|ing) (funding|deal)\b": 0.20,
    # policy / macro favorable
    r"\bpermits? (approved|granted|greenlit)\b": 0.24, r"\bsubsid(y|ies)\b": 0.18, r"\btax credit(s)?\b": 0.16
}

BEARISH_PATTERNS = {
    # price/action
    r"\bplunge(s|d|ing)?\b": -0.32, r"\bsink(s|ing|ed)?\b": -0.26, r"\bslump(s|ed|ing)?\b": -0.24,
    r"\bdrop(s|ped|ping)?\b": -0.20, r"\btumbl(e|es|ed|ing)\b": -0.28, r"\bslide(s|slid|sliding)?\b": -0.20,
    r"\btank(s|ed|ing)?\b": -0.32, r"\bcrash(es|ed|ing)?\b": -0.36, r"\bcrater(s|ed|ing)?\b": -0.36,
    r"\bdive(s|d|ing)?\b": -0.26, r"\bsell[- ]?off\b": -0.22, r"\bretreat(s|ed|ing)?\b": -0.14,
    # fundamentals / guidance
    r"\bmiss(es|ed)?\b": -0.22, r"\bbelow (estimates|EPS|revenue)\b": -0.22,
    r"\bcut(s|ting)? (guidance|outlook|forecast)\b": -0.34, r"\blowers? (guidance|outlook|forecast)\b": -0.30,
    r"\bdowngrade(s|d)?\b": -0.26, r"\bprice target (cut|lowered)\b": -0.24,
    r"\bweak (demand|sales|outlook)\b": -0.24, r"\bdecline(s|d|ing)?\b": -0.16,
    # legal / regulatory / operational
    r"\bprobe(s|d|ing)?\b": -0.26, r"\binvestigat(e|ion|ed|es)\b": -0.26, r"\blawsuit(s)?\b": -0.24,
    r"\bsec (charges?|investigation)\b": -0.30, r"\brecall(s|ed|ing)?\b": -0.28, r"\blayoff(s)?\b": -0.24,
    r"\bhalt(s|ed|ing)?\b": -0.22, r"\bsuspens(ion|ions|e|ed)\b": -0.22, r"\bdelay(s|ed|ing)?\b": -0.18,
    # policy / macro negative
    r"\bban(s|ned|ning)?\b": -0.26, r"\bsanction(s|ed|ing)?\b": -0.26, r"\bshortage(s)?\b": -0.20,
    r"\binflation (spikes?|surges?)\b": -0.20, r"\brate(s)? (hike|hikes|rises|increase)\b": -0.18
}

# quick compile to speed matching
BULLISH_REGEX = [(re.compile(pat, re.I), w) for pat, w in BULLISH_PATTERNS.items()]
BEARISH_REGEX = [(re.compile(pat, re.I), w) for pat, w in BEARISH_PATTERNS.items()]

def rule_boost(text: str) -> Tuple[float, List[Tuple[str, str]]]:
    """
    Returns total bias (sum of weights) and a list of (phrase, 'bullish'|'bearish') it matched.
    """
    bias = 0.0
    hits: List[Tuple[str, str]] = []
    for rx, w in BULLISH_REGEX:
        m = rx.search(text)
        if m:
            bias += w
            hits.append((m.group(0), "bullish"))
    for rx, w in BEARISH_REGEX:
        m = rx.search(text)
        if m:
            bias += w
            hits.append((m.group(0), "bearish"))
    # clip bias so it doesn't overpower the model
    bias = max(min(bias, 0.75), -0.75)
    return bias, hits

# ---------------- Scoring & labeling ----------------
def base_model_score(headline: str) -> Tuple[float, Dict]:
    """
    Convert model logits to signed score in [-1, 1], weighted by confidence.
    +1 = positive, -1 = negative, 0 = neutral.
    """
    inputs = tokenizer(headline, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    # Try to map via id2label (robust)
    # Expect labels like: {0:'negative',1:'neutral',2:'positive'} (case-insensitive)
    idx = int(torch.argmax(probs).item())
    conf = float(probs[idx].item())  # 0..1
    raw_label = id2label.get(idx, "").lower()

    if "pos" in raw_label:
        score = +1.0 * conf
        label = "Bullish"
    elif "neg" in raw_label:
        score = -1.0 * conf
        label = "Bearish"
    else:
        score = 0.0
        label = "Neutral"

    return score, {"label": label, "confidence": conf, "probs": probs.tolist()}

def label_from_combined(score: float) -> Tuple[str, str]:
    # Slight margins so we don't overcall slight noise
    if score > 0.25:
        if score > 0.9:   return "Bullish", "Strong"
        if score > 0.6:   return "Bullish", "Moderate"
        return "Bullish", "Slight"
    elif score < -0.25:
        if score < -0.9:  return "Bearish", "Strong"
        if score < -0.6:  return "Bearish", "Moderate"
        return "Bearish", "Slight"
    return "Neutral", "Balanced"

def combine_scores(model_score: float, bias: float) -> float:
    # Model leads; rules nudge.
    return model_score + bias

# ---------------- Highlight utility ----------------
def highlight_hits(text: str, hits: List[Tuple[str, str]]) -> str:
    """
    Wrap matched phrases in green (bullish) / red (bearish) marks.
    Uses HTML in markdown (safe in Streamlit).
    """
    out = text
    # sort by length desc to avoid nested replacements
    sorted_hits = sorted(hits, key=lambda x: len(x[0]), reverse=True)
    for phrase, kind in sorted_hits:
        color = "#0a8f08" if kind == "bullish" else "#b00020"
        safe = re.escape(phrase)
        out = re.sub(safe, f"<mark style='background:{color}22;border-radius:4px;padding:0 2px'>{phrase}</mark>", out, flags=re.I)
    return out

# ---------------- UI ----------------
with st.form("analyze"):
    headline = st.text_input("Headline", placeholder="e.g., Trilogy Metals stock explodes on 10% stake and permits approved")
    submitted = st.form_submit_button("Analyze")

if submitted and headline.strip():
    h = headline.strip()

    # 1) Base model
    base_score, meta = base_model_score(h)

    # 2) Rule bias (trader phrasing)
    bias, hits = rule_boost(h)

    # 3) Combine + label
    combined = combine_scores(base_score, bias)
    sentiment, strength = label_from_combined(combined)
    confidence_pct = int(round(min(1.0, abs(combined)) * 100))
    sector = detect_sector(h)

    # ----------- Output ----------
    st.subheader("📊 Result")
    st.markdown(f"**Sentiment:** {sentiment}")
    st.markdown(f"**Strength:** {strength}")
    st.markdown(f"**Confidence:** {confidence_pct}%")
    st.markdown(f"**Sector:** {sector}")

    # Headline with highlights
    if hits:
        st.markdown("**Headline (key phrases highlighted):**", unsafe_allow_html=True)
        st.markdown(highlight_hits(h, hits), unsafe_allow_html=True)
    else:
        st.markdown(f"**Headline:** {h}")

    # Reasoning
    if sentiment == "Bullish":
        st.success("Market tone appears optimistic 📈")
        st.write(
            "Language indicates positive momentum or supportive catalysts. "
            "Terms like *soars, surges, beats, raises guidance, approval, stake, permits* "
            "tend to draw risk-on flows, particularly if price/volume confirm."
        )
        st.write("**Suggested lens:** Look for continuation on volume, VWAP pullbacks, and relative strength vs. peers.")
    elif sentiment == "Bearish":
        st.error("Market tone appears cautious 📉")
        st.write(
            "Language signals uncertainty, risk, or negative revisions. "
            "Phrases like *plunges, tanks, cuts guidance, miss, probe, investigation, recall* "
            "often trigger de-risking or rotation to defensives."
        )
        st.write("**Suggested lens:** Watch for LH/LL structure, VWAP rejections, and respect oversold snapbacks on extremes.")
    else:
        st.info("Market tone appears neutral ⚖️")
        st.write(
            "Tone looks balanced or procedural — not clearly risk-on/off. "
            "Neutral headlines often need follow-through or additional catalysts before traders commit."
        )
        st.write("**Suggested lens:** Let price lead; trade ranges and confirmation signals.")

    # Transparency
    st.caption(
        "ℹ️ Confidence reflects a blend of the model’s probability and rule-based phrase matches. "
        "Rules provide small nudges so trader-style wording (e.g., *explodes/tanks*) is respected, "
        "but the model remains primary."
    )
    st.caption("Educational tool — not financial advice.")

# --------------- Notes for you ---------------
# pip install -q transformers torch streamlit
# If you want CSV/batch + Pro-gating later, reuse the earlier Pro-gate pattern.
