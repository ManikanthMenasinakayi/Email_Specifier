import streamlit as st
import pandas as pd
import json
import time
from typing import Literal
from google import genai
from google.genai import types

# --- 1. Configuration and Initialization ---

st.set_page_config(layout="wide", page_title="Insight Mail: LLM Demo")

# NOTE: For security, use the environment variable (recommended).
# The client automatically looks for the GEMINI_API_KEY environment variable.
try:
    # Initialize the Gemini Client explicitly with the hardcoded API key.
    # NOTE: This method is NOT recommended for production environments.
    # The 'api_key' parameter is used instead of relying on environment variables.
    client = genai.Client(api_key="AIzaSyBZbUAT_332xHCfY9Hc19HGvLKmWG8pQLs")
    LLM_READY = True
except Exception as e:
    # Keep the error message for debugging other potential issues (like network failure)
    st.error(f"🚨 Failed to initialize Gemini Client with hardcoded key. Error: {e}")
    client = None
    LLM_READY = False

# --- 2. LLM Output Schema (The Structured Intelligence) ---
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "emotion_tag": {
            "type": "string",
            "description": "The dominant specific emotion (Frustration, Urgency, Appreciation, Confusion, Anger, Neutral)."
        },
        "business_intent": {
            "type": "string",
            "description": "The primary business purpose (Churn Risk, Technical Issue, Billing Dispute, Feature Request, PII Submission/Admin, Positive Feedback)."
        },
        "compliance_flag": {
            "type": "string",
            "description": "Policy risk: 'None', 'PII Found (CC/Phone/Address)', or 'SLA Promise Made'."
        },
        "summary_sentence": {
            "type": "string",
            "description": "A single, concise sentence summarizing the customer's core problem and request."
        }
    },
    "required": ["emotion_tag", "business_intent", "compliance_flag", "summary_sentence"]
}

# --- 3. Core Logic: LLM Call and Scoring ---

def get_insights_from_llm(email_text: str, customer_tier: str) -> dict:
    """Uses OpenAI GPT to output strict JSON."""
    try:
        system_instruction = (
            "You are 'Insight Mail' AI. Output ONLY valid JSON following this schema:\n"
            f"{json.dumps(OUTPUT_SCHEMA, indent=2)}\n\n"
            "No explanation. No prose. JSON only."
        )

        user_prompt = f"""
        Customer Tier: {customer_tier}
        Email Text: {email_text}

        Return JSON only.
        """

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"}
        )

        json_output = response.choices[0].message.content
        return json.loads(json_output)

    except Exception as e:
        st.error(f"OpenAI API Error: {e}")
        return {}

def calculate_priority_score(insights: dict, tier: str) -> int:
    """Calculates a weighted Priority Score (0-100) based on LLM output and business rules."""
    score = 0
    
    # 1. Base Score from Emotion/Intent
    if insights.get('emotion_tag') in ['Frustration', 'Anger']:
        score += 35
    elif insights.get('business_intent') in ['Churn Risk', 'Technical Issue']:
        score += 50 

    # 2. Tier Multiplier (High-Value Customer Weighting)
    if tier == 'Platinum':
        score *= 1.8
    elif tier == 'Gold':
        score *= 1.4

    # 3. Compliance Risk Multiplier (Audit Urgency)
    if 'PII Found' in insights.get('compliance_flag', ''):
        score *= 1.25 
    elif 'SLA Promise Made' in insights.get('compliance_flag', ''):
        score *= 1.15

    return min(int(score), 100)

# --- 4. Streamlit UI Functions ---

def render_results(insights, priority_score):
    """Displays the structured output in a clean, professional format."""
    
    # Determine the priority level text
    if priority_score >= 85:
        priority_label = "🔥 CRITICAL - Immediate Escalation"
        st.error(f"Priority Level: {priority_label}")
    elif priority_score >= 65:
        priority_label = "⚠️ HIGH - Requires Rapid Response"
        st.warning(f"Priority Level: {priority_label}")
    else:
        priority_label = "💡 MEDIUM / LOW - Routine Handling"
        st.info(f"Priority Level: {priority_label}")

    st.markdown("---")
    st.subheader("Actionable Intelligence")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Priority Score",
            value=f"{priority_score}%",
            delta_color="off",
            help="Weighted risk score combining emotion, intent, and customer tier."
        )
        st.metric(
            label="Dominant Emotion",
            value=insights.get('emotion_tag', 'N/A'),
            delta_color="off"
        )

    with col2:
        st.metric(
            label="Business Intent",
            value=insights.get('business_intent', 'N/A'),
            delta_color="off",
            help="The underlying reason for contact (e.g., Churn Risk, Bug Report)."
        )
        # Highlight compliance flag
        compliance_risk = insights.get('compliance_flag', 'None')
        if compliance_risk != 'None':
            st.error(f"Compliance Flag: {compliance_risk}")
        else:
            st.success("Compliance Flag: None")

    st.markdown("---")
    st.markdown(f"**LLM Summary:** *{insights.get('summary_sentence', 'Analysis could not generate a summary.')}*")


def main():
    st.title("📧 Insight Mail: Real-Time Email Intelligence Demo")
    st.caption("Paste an email below to see its instant Priority Score and risk tags.")

    if not LLM_READY:
        st.stop()

    st.markdown("---")

    # --- Input Section ---
    col_input, col_tier = st.columns([3, 1])
    
    with col_input:
        email_text = st.text_area(
            "Paste Customer Email Text Here:",
            height=250,
            value="I am absolutely furious! My Platinum account has been down for 8 hours, and your agent promised me a call back 4 hours ago. This is unacceptable and I am seriously considering canceling. You can reach me on (555) 123-4567 to resolve this immediately.",
            key="email_input"
        )
    
    with col_tier:
        customer_tier = st.selectbox(
            "Select Customer Tier:",
            options=["Platinum", "Gold", "Silver", "Bronze"],
            index=0,
            key="tier_select"
        )

    # --- Run Button ---
    if st.button("Analyze Email (Get Structured Intelligence)", type="primary"):
        if not email_text.strip():
            st.warning("Please enter some email text to analyze.")
        else:
            with st.spinner("Calling Gemini API for deep contextual analysis..."):
                # 1. Get structured tags from LLM
                insights = get_insights_from_llm(email_text, customer_tier)
                
                if insights and insights.get('emotion_tag'):
                    # 2. Calculate final business score
                    priority_score = calculate_priority_score(insights, customer_tier)
                    
                    # 3. Display the results
                    render_results(insights, priority_score)
                else:
                    st.error("Analysis failed. Could not get structured insights from the LLM. Please try again.")

if __name__ == "__main__":
    main()