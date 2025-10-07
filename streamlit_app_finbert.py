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
    r"\bpartnership\b": 0.18, r"\bbuyback(s)?\b": 0.22, r"\bdividend (hike|increase|raised
