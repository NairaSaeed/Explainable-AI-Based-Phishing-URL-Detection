"""
gui/app.py
==========
Phishing URL Detection System — Gradio Dashboard
Paper: "Explainable AI-Based Phishing URL Detection"

Architecture:
  Auto mode  → try Full-feature RF first; fallback to URL-only RF on failure.
  The GUI delegates ALL prediction logic to src/predictor.py.
  Rule-based signal overrides are explicitly absent from this file.
"""

from __future__ import annotations

import os
import sys
import traceback

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gradio as gr
from src.predictor import predict_url, get_registry

# ── Verdict formatters ─────────────────────────────────────────────────────
_VERDICT_FMT = {
    "PHISHING": ("🔴 PHISHING", "🔴"),
    "REVIEW":   ("🟡 REVIEW",   "🟡"),
    "SAFE":     ("🟢 SAFE",     "🟢"),
}
_RISK_FMT = {
    "HIGH":   "HIGH 🔴",
    "MEDIUM": "MEDIUM 🟠",
    "REVIEW": "REVIEW 🟡",
    "LOW":    "LOW 🟢",
}


def _format_result(res: dict) -> str:
    verdict, risk_icon = _VERDICT_FMT.get(res["final_result"], ("❓", "❓"))
    risk = _RISK_FMT.get(res["risk_level"], res["risk_level"])

    signals = res.get("detected_signals", [])
    signals_text = ("\n".join(f"  • {s}" for s in signals)
                    if signals else "  None detected.")

    # Show a limitation note when URL-only fallback is used due to blocked fetch
    fallback_note = ""
    fetch = res.get("page_fetch_status", "")
    if "url_only" in res.get("prediction_mode", "").lower() or "fallback" in res.get("prediction_mode", "").lower():
        if any(x in fetch for x in ("blocked", "timeout", "error", "ssl")):
            fallback_note = (
                "\n  ⚠ URL-only fallback active: the page could not be fetched.\n"
                "    Prediction is based on URL lexical features only.\n"
                "    URL-only models have limited generalization to unseen URL patterns."
            )

    return (
        f"{'━'*52}\n"
        f"  RESULT:  {verdict}\n"
        f"{'━'*52}\n\n"
        f"  Risk Level:              {risk}\n"
        f"  Phishing Risk Score:     {res['probability_phishing']:.2%}\n"
        f"  Legitimate Probability:  {res['probability_legitimate']:.2%}\n"
        f"  Final Confidence:        {res['confidence']:.2%}\n\n"
        f"  (Note: A non-zero phishing risk score is normal. The final verdict depends on the decision threshold:\n"
        f"   • Risk >= 75%  → PHISHING\n"
        f"   • 45% to 74%   → REVIEW\n"
        f"   • Below 45%    → SAFE)\n\n"
        f"  Domain:                  {res['domain']}\n\n"
        f"  Selected Model:          {res['selected_model']}\n"
        f"  Model Accuracy (val):    {res['model_accuracy']:.4f}\n"
        f"  Model F1 (val):          {res['model_f1']:.4f}\n"
        f"  Prediction Mode:         {res['prediction_mode']}\n"
        f"  Page Fetch Status:       {res['page_fetch_status']}\n"
        f"  Feature Set Used:        {res['feature_set_used']}\n"
        f"  Content Coverage:        {res['content_coverage']:.0%}\n"
        f"{fallback_note}\n\n"
        f"  Contextual Signals:\n"
        f"{signals_text}\n\n"
        f"  Summary:  {res['summary']}\n\n"
        f"  ⚠ These signals are shown for analyst interpretation only and do not override the trained ML model verdict."
    )


def _format_shap(res: dict) -> str:
    lines = res.get("shap_explanation", [])
    if not lines:
        return "Explanation unavailable for this model/run."
    return "\n".join(lines)


def _format_lime(res: dict) -> str:
    lines = res.get("lime_explanation", [])
    if not lines:
        return "Explanation unavailable for this model/run."
    return "\n".join(lines)


def _format_diag(res: dict) -> str:
    d = res["diagnostics"]
    missing = res["missing_features"]
    extra   = res["extra_features"]

    top_vals = "\n".join(
        f"    {fname:<34} {val}"
        for fname, val in d.get("top_extracted", [])
    )

    return (
        f"{'─'*52}\n"
        f"  MODEL DIAGNOSTICS\n"
        f"{'─'*52}\n"
        f"  Model file:              {d.get('model_file')}\n"
        f"  model.classes_:          {d.get('model_classes')}\n"
        f"  Phishing class index:    {d.get('phishing_class_index')}\n"
        f"  Decision threshold:      {d.get('decision_threshold')}\n\n"
        f"  Expected features:       {res['expected_feature_count']}\n"
        f"  Extracted features:      {res['extracted_feature_count']}\n"
        f"  Missing ({len(missing)}):            "
        f"{missing[:8]}{'…' if len(missing) > 8 else ''}\n"
        f"  Extra ({len(extra)}):              "
        f"{extra[:5]}{'…' if len(extra) > 5 else ''}\n"
        f"  Feature vector shape:    {d.get('vec_shape')}\n\n"
        f"  Top extracted feature values:\n{top_vals}"
    )


# ── Gradio callback ────────────────────────────────────────────────────────
def analyze(url_raw: str, model_choice: str):
    if not url_raw or not url_raw.strip():
        msg = "⚠️  Please enter a URL to analyze."
        return msg, "", "", ""
    try:
        res = predict_url(url_raw, model_choice)
        return (
            _format_result(res),
            _format_shap(res),
            _format_lime(res),
            _format_diag(res),
        )
    except Exception:
        err = traceback.format_exc()
        return f"❌ Unexpected error:\n{err}", "", "", err


# ── Build Gradio UI ────────────────────────────────────────────────────────
def build_app(model_choices: list[str]) -> gr.Blocks:
    css = """
    body, .gradio-container { background:#0e1117 !important; color:#e6edf3; }
    .gr-block, .gr-box { background:#161b22 !important;
                         border:1px solid #30363d; border-radius:8px; }
    h1 { color:#58a6ff; font-size:1.8rem; }
    .gr-button-primary { background:#1f6feb !important; border:none;
                         color:white; font-weight:600; }
    .gr-button-primary:hover { background:#388bfd !important; }
    label { color:#8b949e !important; font-size:0.85rem; }
    textarea, input[type=text] { background:#21262d !important;
                                 color:#e6edf3 !important;
                                 border:1px solid #30363d !important;
                                 border-radius:6px; }
    """

    with gr.Blocks(title="Phishing URL Detection System", css=css) as demo:

        gr.Markdown("# 🛡️ Phishing URL Detection System")
        gr.Markdown(
            "<div style='color:#8b949e;font-size:0.95rem;margin-top:-12px'>"
            "Explainable AI using SHAP &amp; LIME &nbsp;|&nbsp; "
            "Full-feature &amp; URL-only modes &nbsp;|&nbsp; "
            "Paper: <em>Explainable AI-Based Phishing URL Detection</em>"
            "</div>"
        )

        with gr.Row():
            url_input = gr.Textbox(
                label="🔗 Enter URL to analyze",
                placeholder="https://example.com",
                scale=5,
            )
            model_select = gr.Dropdown(
                label="Model / Mode",
                choices=model_choices,
                value="Auto",
                scale=2,
            )
            btn = gr.Button("🔍 Analyze URL", variant="primary", scale=1)

        result_box = gr.Textbox(
            label="📊 Result Summary", lines=22, interactive=False
        )

        with gr.Row():
            shap_box = gr.Textbox(
                label="🔬 SHAP Explanation", lines=15, interactive=False
            )
            lime_box = gr.Textbox(
                label="🧪 LIME Explanation", lines=15, interactive=False
            )

        with gr.Accordion("🔧 Diagnostics (Advanced)", open=False):
            diag_box = gr.Textbox(
                label="Diagnostics", lines=20, interactive=False
            )

        _outputs = [result_box, shap_box, lime_box, diag_box]
        btn.click(analyze, inputs=[url_input, model_select], outputs=_outputs)
        url_input.submit(analyze, inputs=[url_input, model_select], outputs=_outputs)

    return demo


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading models and building explainers…")
    try:
        registry = get_registry()
    except FileNotFoundError as e:
        print(f"\n❌ {e}\n")
        sys.exit(1)

    model_choices = ["Auto"] + list(registry.keys())
    demo = build_app(model_choices)
    demo.launch(share=False)