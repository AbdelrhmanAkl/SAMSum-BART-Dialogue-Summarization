import html

import streamlit as st
import torch

from inference.inference import load_model, summarize_dialogue


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="SAMSum Dialogue Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# Design System (light, pearl-white)
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #F6F7FB;
        --surface: #FFFFFF;
        --ink: #172033;
        --muted: #667085;
        --line: #E6E9F2;
        --accent: #4F5BD5;
        --accent-soft: #EEF0FF;
        --ok: #1F8F6B;
        --shadow: 0 1px 2px rgba(23, 32, 51, 0.04), 0 8px 24px rgba(79, 91, 213, 0.06);
    }

    html, body, [class*="css"], .stApp, button, textarea, input {
        font-family: 'Manrope', system-ui, sans-serif !important;
    }

    .stApp {
        background:
            radial-gradient(900px 420px at 85% -10%, #E9ECFF 0%, rgba(233,236,255,0) 70%),
            var(--bg);
        color: var(--ink);
    }

    /* Chrome */
    footer, #MainMenu { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    .block-container {
        max-width: 1180px;
        padding-top: 2.4rem;
        padding-bottom: 3rem;
    }

    /* Hero */
    .pill {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent);
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.9rem;
        font-weight: 800;
        letter-spacing: -0.035em;
        line-height: 1.1;
        color: var(--ink);
        margin: 0 0 0.7rem 0;
    }
    .hero-subtitle {
        font-size: 1.08rem;
        color: var(--muted);
        max-width: 620px;
        line-height: 1.65;
        margin-bottom: 2.2rem;
    }

    /* Section titles */
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--ink);
        margin: 0 0 0.7rem 0.1rem;
    }

    /* Input */
    .stTextArea textarea {
        background: var(--surface) !important;
        border: 1px solid var(--line) !important;
        border-radius: 16px !important;
        padding: 1.1rem 1.2rem !important;
        font-size: 0.98rem !important;
        line-height: 1.7 !important;
        color: var(--ink) !important;
        box-shadow: var(--shadow);
    }
    .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 4px rgba(79, 91, 213, 0.12) !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 12px;
        height: 3rem;
        font-weight: 600;
        border: 1px solid var(--line);
        background: var(--surface);
        color: var(--ink);
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        border-color: var(--accent);
        color: var(--accent);
    }
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: var(--accent);
        border: none;
        color: #fff;
        box-shadow: 0 6px 18px rgba(79, 91, 213, 0.28);
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background: #4450C6;
        color: #fff;
    }

    /* Summary */
    .summary-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-left: 4px solid var(--accent);
        border-radius: 16px;
        padding: 1.5rem 1.6rem;
        font-size: 1.15rem;
        line-height: 1.75;
        font-weight: 500;
        color: var(--ink);
        box-shadow: var(--shadow);
    }
    .empty-card {
        border: 1.5px dashed #D3D8EA;
        border-radius: 16px;
        padding: 2.4rem 1.5rem;
        text-align: center;
        color: var(--muted);
        background: rgba(255, 255, 255, 0.6);
        line-height: 1.6;
    }

    /* Stat tiles */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.9rem;
        margin-top: 1rem;
    }
    .stat {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1rem 1.2rem;
    }
    .stat-label { font-size: 0.82rem; color: var(--muted); font-weight: 600; }
    .stat-value {
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: var(--ink);
        margin-top: 0.15rem;
    }

    /* Panels */
    .panel {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        height: 100%;
        box-shadow: var(--shadow);
    }
    .panel h4 { margin: 0 0 0.7rem 0; font-size: 1.05rem; font-weight: 700; }
    .panel p { color: var(--muted); line-height: 1.7; margin: 0 0 0.8rem 0; }
    .panel ul { margin: 0; padding-left: 1.1rem; color: var(--ink); line-height: 1.95; }
    .kv {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.65rem 0;
        border-bottom: 1px solid var(--line);
        font-size: 0.95rem;
    }
    .kv:last-child { border-bottom: none; }
    .kv span:first-child { color: var(--muted); }
    .kv span:last-child { font-weight: 600; text-align: right; }
    code {
        background: var(--accent-soft) !important;
        color: var(--accent) !important;
        border-radius: 6px;
        padding: 0.1rem 0.4rem !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: var(--ink); letter-spacing: -0.01em; }
    .side-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--muted);
        margin: 0.9rem 0 0.3rem 0;
    }
    .side-value {
        font-size: 0.92rem;
        font-weight: 600;
        padding: 0.5rem 0.75rem;
        background: var(--bg);
        border: 1px solid var(--line);
        border-radius: 10px;
        word-break: break-all;
    }

    hr { border-color: var(--line) !important; }
    .footer {
        text-align: center;
        color: var(--muted);
        font-size: 0.85rem;
        padding-top: 2.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Session State
# =========================================================

DEFAULT_DIALOGUE = """Hannah: Do you have Betty's number?
Amanda: I can't find it. Ask Larry.
Hannah: I don't know Larry.
Amanda: He's very nice.
Hannah: Okay, I'll text him."""

if "summary" not in st.session_state:
    st.session_state.summary = None

if "last_dialogue" not in st.session_state:
    st.session_state.last_dialogue = ""

if "dialogue" not in st.session_state:
    st.session_state.dialogue = DEFAULT_DIALOGUE


def clear_all():
    st.session_state.summary = None
    st.session_state.last_dialogue = ""
    st.session_state.dialogue = ""


# =========================================================
# Model Loading
# =========================================================

@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.markdown("### Model settings")

    st.markdown(
        '<div class="side-label">Model</div>'
        '<div class="side-value">facebook/bart-large-cnn</div>'
        '<div class="side-label">Task</div>'
        '<div class="side-value">Abstractive dialogue summarization</div>'
        '<div class="side-label">Dataset</div>'
        '<div class="side-value">knkarthick/samsum</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### Generation")

    num_beams = st.slider(
        "Beam search",
        min_value=1,
        max_value=8,
        value=4,
        step=1,
        help="Higher values can improve quality but increase inference time.",
    )

    max_summary_length = st.slider(
        "Maximum summary length",
        min_value=32,
        max_value=128,
        value=96,
        step=8,
    )

    min_summary_length = st.slider(
        "Minimum summary length",
        min_value=4,
        max_value=32,
        value=8,
        step=4,
    )

    st.divider()

    st.markdown("### Runtime")

    if torch.cuda.is_available():
        st.success(f"GPU: {torch.cuda.get_device_name(0)}")
        st.caption(f"CUDA {torch.version.cuda}")
    else:
        st.warning("Running on CPU")

    st.caption(
        f"Input up to 512 tokens. Output up to {max_summary_length} tokens. "
        "No-repeat n-gram size 3."
    )


# =========================================================
# Header
# =========================================================

st.markdown(
    """
    <div class="pill">Fine-tuned on SAMSum</div>
    <div class="hero-title">Turn any conversation<br>into a clear summary</div>
    <div class="hero-subtitle">
        Paste a chat or dialogue and get a short, readable summary
        from a fine-tuned BART Large CNN model.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Load Model
# =========================================================

try:
    with st.spinner("Loading summarization model..."):
        tokenizer, model, device = get_model()
except Exception as error:
    st.error("Unable to load the summarization model.")
    st.exception(error)
    st.stop()


# =========================================================
# Main Layout
# =========================================================

left_column, right_column = st.columns([1.15, 0.85], gap="large")


# ---------------------------------------------------------
# Dialogue Input
# ---------------------------------------------------------

with left_column:

    st.markdown('<div class="section-title">Dialogue</div>', unsafe_allow_html=True)

    dialogue = st.text_area(
        "Conversation to summarize",
        key="dialogue",
        height=320,
        placeholder=(
            "Alice: Are you coming to the meeting?\n"
            "Bob: Yes, I'll be there at 3 PM."
        ),
        label_visibility="collapsed",
    )

    input_words = len(dialogue.split()) if dialogue.strip() else 0
    st.caption(f"{input_words} words")

    button_col1, button_col2 = st.columns([3, 1])

    with button_col1:
        generate_clicked = st.button(
            "Generate summary",
            type="primary",
            use_container_width=True,
        )

    with button_col2:
        st.button("Clear", on_click=clear_all, use_container_width=True)


# =========================================================
# Generate Summary
# =========================================================

if generate_clicked:

    if not dialogue.strip():
        st.warning("Please enter a dialogue before generating a summary.")

    else:
        try:
            with st.spinner("Generating summary..."):
                summary = summarize_dialogue(
                    dialogue=dialogue,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    max_source_length=512,
                    max_summary_length=max_summary_length,
                    min_summary_length=min_summary_length,
                    num_beams=num_beams,
                )

            st.session_state.summary = summary
            st.session_state.last_dialogue = dialogue

        except Exception as error:
            st.error("An error occurred during summarization.")
            st.exception(error)


# ---------------------------------------------------------
# Summary Output
# ---------------------------------------------------------

with right_column:

    st.markdown('<div class="section-title">Summary</div>', unsafe_allow_html=True)

    if st.session_state.summary:

        safe_summary = html.escape(st.session_state.summary)

        summary_words = len(st.session_state.summary.split())
        original_words = len(st.session_state.last_dialogue.split())

        compression_ratio = (
            (1 - summary_words / original_words) * 100 if original_words > 0 else 0
        )

        st.markdown(
            f"""
            <div class="summary-card">{safe_summary}</div>
            <div class="stat-grid">
                <div class="stat">
                    <div class="stat-label">Summary words</div>
                    <div class="stat-value">{summary_words}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Shorter than original by</div>
                    <div class="stat-value">{compression_ratio:.0f}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            """
            <div class="empty-card">
                Your summary will appear here.<br>
                Add a dialogue and select <b>Generate summary</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# Model Performance
# =========================================================

st.markdown("<div style='height:2.2rem'></div>", unsafe_allow_html=True)
st.markdown('<div class="section-title">Model performance</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="stat-grid" style="margin-top:0">
        <div class="stat"><div class="stat-label">ROUGE-1</div><div class="stat-value">40.58</div></div>
        <div class="stat"><div class="stat-label">ROUGE-2</div><div class="stat-value">20.12</div></div>
        <div class="stat"><div class="stat-label">ROUGE-L</div><div class="stat-value">31.03</div></div>
        <div class="stat"><div class="stat-label">Test samples</div><div class="stat-value">819</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Project Information
# =========================================================

st.markdown("<div style='height:2.2rem'></div>", unsafe_allow_html=True)

about_col, technical_col = st.columns(2, gap="large")

with about_col:
    st.markdown(
        """
        <div class="panel">
            <h4>About this project</h4>
            <p>
                An abstractive dialogue summarization system built on a
                BART Large CNN model fine-tuned on the SAMSum dataset.
            </p>
            <ul>
                <li>14,731 training dialogues</li>
                <li>819 test dialogues</li>
                <li>Beam search generation</li>
                <li>GPU-accelerated inference</li>
                <li>ROUGE-based evaluation</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with technical_col:
    st.markdown(
        """
        <div class="panel">
            <h4>Technical details</h4>
            <div class="kv"><span>Base model</span><span><code>facebook/bart-large-cnn</code></span></div>
            <div class="kv"><span>Dataset</span><span><code>knkarthick/samsum</code></span></div>
            <div class="kv"><span>Input limit</span><span>512 tokens</span></div>
            <div class="kv"><span>Output limit</span><span>96 tokens</span></div>
            <div class="kv"><span>Beam search</span><span>4</span></div>
            <div class="kv"><span>No-repeat n-gram</span><span>3</span></div>
            <div class="kv"><span>Runtime</span><span>PyTorch + Transformers</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# Footer
# =========================================================

st.markdown(
    '<div class="footer">SAMSum Dialogue Summarizer. Fine-tuned transformer NLP application.</div>',
    unsafe_allow_html=True,
)
