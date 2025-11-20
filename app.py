import streamlit as st
import pandas as pd
import re
import math
from collections import Counter, defaultdict
from datetime import datetime

st.set_page_config(page_title="Email Analytics AI", layout="wide")

# ----------------------
# Utilities / Lexicons
# ----------------------
def normalize_subject(s):
    if not isinstance(s, str):
        return ""
    x = s.lower().strip()
    x = re.sub(r"^(re:|fwd:|fw:|rv:|sv:)\s*", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def tokenize(text):
    if not isinstance(text, str):
        return []
    x = text.lower()
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    return [t for t in x.split() if t]

def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def default_sentiment_lexicon():
    df = load_csv("data/lexicon_sentiment.csv")
    if df.empty:
        return {"positive": set(["great","good","excellent","love","satisfied","happy","fast","helpful","thank" ]),
                "negative": set(["bad","terrible","hate","angry","upset","slow","unacceptable","issue","problem","refund","complaint","escalate"]) }
    pos = set(df[df["label"].str.lower()=="positive"]["token"].astype(str).str.lower())
    neg = set(df[df["label"].str.lower()=="negative"]["token"].astype(str).str.lower())
    return {"positive": pos, "negative": neg}

def default_emotion_lexicon():
    df = load_csv("data/lexicon_emotion.csv")
    if df.empty:
        return {
            "joy": set(["great","glad","pleased","love","happy","thank","appreciate"]),
            "anger": set(["angry","furious","mad","outrage","annoyed","unacceptable"]),
            "sadness": set(["sad","disappointed","regret","sorry","unhappy"]),
            "fear": set(["worried","concerned","afraid","risk","danger"]),
            "surprise": set(["surprised","unexpected","shock","sudden"]),
            "disgust": set(["disgust","gross","nasty"]),
        }
    out = {}
    for k, g in df.groupby("label"):
        out[str(k).lower()] = set(g["token"].astype(str).str.lower())
    return out

def urgency_tokens():
    return set(["urgent","asap","immediately","now","today","delay","deadline","priority","escalate","critical"])

# ----------------------
# Simple detectors
# ----------------------
def detect_sentiment(tokens, lex):
    pos = sum(1 for t in tokens if t in lex["positive"])
    neg = sum(1 for t in tokens if t in lex["negative"])
    if neg > pos and neg > 0:
        return "negative", neg - pos
    if pos > neg and pos > 0:
        return "positive", pos - neg
    return "neutral", 0

def detect_emotion(tokens, emo_lex):
    scores = {k: sum(1 for t in tokens if t in v) for k, v in emo_lex.items()}
    if not scores:
        return "neutral", 0
    label = max(scores, key=lambda k: scores[k])
    return label if scores[label] > 0 else "neutral", scores.get(label, 0)

def detect_intent(tokens):
    intents = {
        "complaint": set(["complaint","issue","problem","unacceptable","refund","angry","escalate"]),
        "inquiry": set(["how","what","when","where","why","help","support","question"]),
        "purchase": set(["buy","purchase","order","quote","pricing","invoice"]),
        "churn_risk": set(["cancel","termination","switch","refund","unsatisfied","leave"]),
        "feedback": set(["feedback","suggestion","recommend","review"]),
    }
    scores = {k: sum(1 for t in tokens if t in v) for k, v in intents.items()}
    label = max(scores, key=lambda k: scores[k])
    return label if scores[label] > 0 else "inquiry"

def compute_tone(text):
    if not isinstance(text, str) or not text:
        return {"shouting": 0.0, "exclamations": 0, "questions": 0, "avg_word_len": 0.0}
    upper = sum(1 for c in text if c.isupper())
    letters = sum(1 for c in text if c.isalpha())
    shouting = (upper / letters) if letters else 0.0
    exclam = text.count("!")
    ques = text.count("?")
    tokens = tokenize(text)
    avgw = sum(len(t) for t in tokens) / len(tokens) if tokens else 0.0
    return {"shouting": shouting, "exclamations": exclam, "questions": ques, "avg_word_len": avgw}

def detect_compliance(text, rules_df):
    if rules_df is None or rules_df.empty:
        rules = [
            {"name":"pii_email","pattern": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"},
            {"name":"pii_phone","pattern": r"\b(?:\+?\d[\d\s-]{7,})\b"},
            {"name":"profanity","pattern": r"\b(damn|hell|shit|crap)\b"},
        ]
    else:
        rules = [{"name": str(r["name"]), "pattern": str(r["pattern"])} for _, r in rules_df.iterrows()]
    hits = []
    for r in rules:
        if re.search(r["pattern"], text or "", flags=re.IGNORECASE):
            hits.append(r["name"])
    return hits

# ----------------------
# Naive Bayes (small trainer)
# ----------------------
class NaiveBayes:
    def __init__(self):
        self.class_counts = defaultdict(int)
        self.token_counts = defaultdict(lambda: defaultdict(int))
        self.vocab = set()
        self.total_tokens = defaultdict(int)
        self.trained = False

    def train(self, df, text_col, label_col):
        for _, row in df.iterrows():
            label = str(row[label_col]).lower()
            toks = tokenize(str(row[text_col]))
            self.class_counts[label] += 1
            self.total_tokens[label] += len(toks)
            for t in toks:
                self.vocab.add(t)
                self.token_counts[label][t] += 1
        self.trained = True

    def predict(self, text):
        if not self.trained:
            return None, 0.0
        toks = tokenize(text)
        V = len(self.vocab) if self.vocab else 1
        scores = {}
        total_docs = sum(self.class_counts.values()) or 1
        for c in self.class_counts:
            logp = math.log((self.class_counts[c] + 1) / (total_docs + len(self.class_counts)))
            for t in toks:
                tc = self.token_counts[c].get(t, 0)
                logp += math.log((tc + 1) / (self.total_tokens[c] + V))
            scores[c] = logp
        label = max(scores, key=lambda k: scores[k])
        return label, scores[label]

# ----------------------
# Scoring, routing & helpers
# ----------------------
def risk_score(sentiment_label, sentiment_strength, emotion_label, tone, urgency_score, compliance_hits):
    score = 0
    if sentiment_label == "negative":
        score += 30 + min(20, sentiment_strength * 5)
    if emotion_label in ["anger","fear","sadness"]:
        score += 20
    score += min(20, int(tone.get("shouting",0.0)*100))
    score += min(15, tone.get("exclamations",0)*3)
    score += min(15, urgency_score*5)
    score += min(30, len(compliance_hits)*10)
    return min(100, score)

def recommend_response(intent, sentiment_label):
    if intent == "complaint" or sentiment_label == "negative":
        return "Acknowledge issue, apologize, provide immediate resolution steps and timeline."
    if intent == "inquiry":
        return "Answer the question clearly, include links and next steps."
    if intent == "purchase":
        return "Share pricing, benefits, and a guided checkout or sales contact."
    if intent == "churn_risk":
        return "Offer retention incentive, schedule a call, address pain points explicitly."
    return "Thank for feedback and outline any follow-up actions."

def route_owner(intent, risk):
    if risk >= 70:
        return "Escalation Desk"
    if intent in ["purchase"]:
        return "Sales"
    if intent in ["complaint","churn_risk"]:
        return "Support"
    return "Customer Success"

def link_threads(df):
    df = df.copy()
    if "thread_id" in df.columns and df["thread_id"].notnull().any():
        df["thread_key"] = df["thread_id"].astype(str)
        return df
    subj = df.get("subject", pd.Series([""]*len(df)))
    df["norm_subject"] = subj.apply(normalize_subject)
    participants = df.get("sender", pd.Series([""]*len(df))).astype(str) + "|" + df.get("recipient", pd.Series([""]*len(df))).astype(str)
    df["thread_key"] = df["norm_subject"].astype(str) + "|" + participants.astype(str)
    return df

def analyze_emails(df, models, lex_sent, lex_emo, rules_df):
    out = []
    urg = urgency_tokens()
    for _, r in df.iterrows():
        body = str(r.get("body",""))
        toks = tokenize(body)
        s_label, s_strength = detect_sentiment(toks, lex_sent)
        if models.get("sentiment"):
            lab, _ = models["sentiment"].predict(body)
            if lab:
                s_label = lab
        e_label, _ = detect_emotion(toks, lex_emo)
        if models.get("emotion"):
            lab, _ = models["emotion"].predict(body)
            if lab:
                e_label = lab
        i_label = detect_intent(toks)
        if models.get("intent"):
            lab, _ = models["intent"].predict(body)
            if lab:
                i_label = lab
        tone = compute_tone(body)
        u_score = sum(1 for t in toks if t in urg)
        comp = detect_compliance(body, rules_df)
        risk = risk_score(s_label, s_strength, e_label, tone, u_score, comp)
        reco = recommend_response(i_label, s_label)
        owner = route_owner(i_label, risk)
        out.append({
            "id": r.get("id",""),
            "timestamp": r.get("timestamp",""),
            "sender": r.get("sender",""),
            "recipient": r.get("recipient",""),
            "subject": r.get("subject",""),
            "thread_key": r.get("thread_key",""),
            "sentiment": s_label,
            "emotion": e_label,
            "intent": i_label,
            "risk": risk,
            "owner": owner,
            "recommendation": reco,
            "compliance_hits": ",".join(comp),
            "exclamations": tone.get("exclamations",0),
            "questions": tone.get("questions",0),
            "shouting": round(tone.get("shouting",0.0),3),
        })
    return pd.DataFrame(out)

# ----------------------
# UI - ensure safe session_state use
# ----------------------
def ui_layout():
    st.title("Next-Gen Email Analytics Platform")
    st.caption("NLP+LLMs-inspired interpretation using Python, Streamlit, and pandas")
    tab = st.sidebar.radio("Sections", ["Data", "Train", "Analyze", "Dashboard", "Export"])
    return tab

def ui_data():
    st.subheader("Data Sources")

    # Initialize session_state keys if missing (safe)
    if "emails" not in st.session_state:
        st.session_state["emails"] = None
    if "rules" not in st.session_state:
        st.session_state["rules"] = None

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("Emails CSV")
        up = st.file_uploader("Upload emails.csv", type=["csv"], key="uploader_emails")
        if up is not None:
            try:
                df = pd.read_csv(up)
                st.session_state["emails"] = df
                st.success("Emails loaded successfully")
            except Exception as e:
                st.error(f"Failed to read uploaded emails CSV: {e}")

    with c2:
        st.markdown("Compliance Rules CSV")
        up2 = st.file_uploader("Upload compliance_rules.csv", type=["csv"], key="uploader_rules")
        if up2 is not None:
            try:
                df2 = pd.read_csv(up2)
                st.session_state["rules"] = df2
                st.success("Compliance rules loaded")
            except Exception as e:
                st.error(f"Failed to read rules CSV: {e}")

    if st.session_state["emails"] is None:
        st.info("Loading example emails if none uploaded")
        st.session_state["emails"] = pd.DataFrame([
            {"id":1,"timestamp":"2025-11-01","sender":"alice@example.com","recipient":"support@company.com","subject":"Order refund","body":"I am angry and want a refund ASAP. This is unacceptable!"},
            {"id":2,"timestamp":"2025-11-02","sender":"bob@example.com","recipient":"sales@company.com","subject":"Pricing inquiry","body":"Hi, what is the price for enterprise plan?"},
            {"id":3,"timestamp":"2025-11-03","sender":"carol@example.com","recipient":"success@company.com","subject":"Great support","body":"Thank you for the great help, very happy with the resolution."},
        ])

    if st.session_state["rules"] is None:
        st.info("Loading default compliance rules if none uploaded")
        st.session_state["rules"] = load_csv("data/compliance_rules.csv")

    st.dataframe(st.session_state["emails"], use_container_width=True)

def ui_train():
    st.subheader("Train Classifiers")
    c1, c2, c3 = st.columns(3)
    if "models" not in st.session_state:
        st.session_state["models"] = {}

    with c1:
        st.markdown("Sentiment Model")
        up = st.file_uploader("Upload sentiment training CSV", type=["csv"], key="train_sent")
        if st.button("Train Sentiment"):
            df = None
            try:
                df = pd.read_csv(up) if up is not None else load_csv("data/sentiment_training.csv")
            except Exception as e:
                st.error(f"Failed to load training CSV: {e}")
            if df is not None and not df.empty:
                m = NaiveBayes()
                m.train(df, "text", "label")
                st.session_state["models"]["sentiment"] = m
                st.success("Sentiment model trained")
            else:
                st.warning("No training data provided")

    with c2:
        st.markdown("Emotion Model")
        up2 = st.file_uploader("Upload emotion training CSV", type=["csv"], key="train_emo")
        if st.button("Train Emotion"):
            df = None
            try:
                df = pd.read_csv(up2) if up2 is not None else load_csv("data/emotion_training.csv")
            except Exception as e:
                st.error(f"Failed to load training CSV: {e}")
            if df is not None and not df.empty:
                m = NaiveBayes()
                m.train(df, "text", "label")
                st.session_state["models"]["emotion"] = m
                st.success("Emotion model trained")
            else:
                st.warning("No training data provided")

    with c3:
        st.markdown("Intent Model")
        up3 = st.file_uploader("Upload intent training CSV", type=["csv"], key="train_intent")
        if st.button("Train Intent"):
            df = None
            try:
                df = pd.read_csv(up3) if up3 is not None else load_csv("data/intent_training.csv")
            except Exception as e:
                st.error(f"Failed to load training CSV: {e}")
            if df is not None and not df.empty:
                m = NaiveBayes()
                m.train(df, "text", "label")
                st.session_state["models"]["intent"] = m
                st.success("Intent model trained")
            else:
                st.warning("No training data provided")

def ui_analyze():
    st.subheader("Analyze Emails")
    emails = st.session_state.get("emails", pd.DataFrame())
    rules = st.session_state.get("rules", pd.DataFrame())
    lex_sent = default_sentiment_lexicon()
    lex_emo = default_emotion_lexicon()

    if emails is None or emails.empty:
        st.warning("No emails available - upload or create sample data in Data tab")
        return

    linked = link_threads(emails)
    st.session_state["linked_emails"] = linked
    try:
        res = analyze_emails(linked, st.session_state.get("models", {}), lex_sent, lex_emo, rules)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return
    st.session_state["analysis"] = res
    st.dataframe(res, use_container_width=True)

def ui_dashboard():
    st.subheader("Dashboards")
    res = st.session_state.get("analysis", pd.DataFrame())
    if res is None or res.empty:
        st.warning("Run Analyze first")
        return
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("High-Risk Emails", int((res["risk"]>=70).sum()))
        st.metric("Compliance Flags", int(res["compliance_hits"].astype(str).apply(lambda x: 0 if x=="" else len(x.split(","))).sum()))
    with c2:
        st.metric("Negative Sentiment", int((res["sentiment"]=="negative").sum()))
        st.metric("Complaints", int((res["intent"]=="complaint").sum()))
    with c3:
        st.metric("Avg Risk", round(res["risk"].mean(),1))
        st.metric("Positive Sentiment", int((res["sentiment"]=="positive").sum()))
    st.bar_chart(res.groupby("intent")["risk"].mean())
    st.bar_chart(res.groupby("sentiment")["risk"].mean())
    st.bar_chart(res.groupby("owner")["risk"].mean())

def ui_export():
    st.subheader("Export")
    res = st.session_state.get("analysis", pd.DataFrame())
    if res is None or res.empty:
        st.warning("Nothing to export")
        return
    csv = res.to_csv(index=False).encode("utf-8")
    st.download_button("Download Analysis CSV", csv, "email_analysis.csv", "text/csv")

def _has_streamlit_ctx():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False

# ----------------------
# App entry (only if running in Streamlit)
# ----------------------
if _has_streamlit_ctx():
    tab = ui_layout()
    if tab == "Data":
        ui_data()
    elif tab == "Train":
        ui_train()
    elif tab == "Analyze":
        ui_analyze()
    elif tab == "Dashboard":
        ui_dashboard()
    else:
        ui_export()
