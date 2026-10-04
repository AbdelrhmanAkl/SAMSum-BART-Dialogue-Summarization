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


def html_block(markup: str) -> str:
    """Flatten HTML so Markdown never mistakes indented lines for code blocks."""
    return "".join(line.strip() for line in markup.splitlines())


# =========================================================
# Design System: Bento + Soft Glass (warm pearl, light)
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #F7F5F1;
        --ink: #14142B;
        --ink-soft: #3B3B58;
        --muted: #6E6E86;
        --glass: rgba(255, 255, 255, 0.62);
        --glass-strong: rgba(255, 255, 255, 0.82);
        --stroke: rgba(255, 255, 255, 0.9);
        --line: rgba(20, 20, 43, 0.08);
        --violet: #6D5EF5;
        --rose: #E8719F;
        --peach: #F5A66B;
        --mint: #2FB68A;
        --shadow:
            0 1px 1px rgba(20, 20, 43, 0.03),
            0 10px 30px rgba(109, 94, 245, 0.07),
            0 30px 60px rgba(20, 20, 43, 0.04);
        --radius: 24px;
    }

    /* ---------- Fonts ---------- */
    .stApp, .stApp p, .stApp li, .stApp label, .stApp button,
    .stApp textarea, .stApp input, .stApp h1, .stApp h2, .stApp h3 {
        font-family: 'Manrope', system-ui, sans-serif !important;
    }
    [data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded {
        font-family: 'Material Symbols Rounded' !important;
    }

    /* ---------- Canvas: soft mesh gradient ---------- */
    .stApp {
        color: var(--ink);
        background:
            radial-gradient(700px 480px at 8% -5%, rgba(109, 94, 245, 0.20), transparent 70%),
            radial-gradient(640px 460px at 95% 5%, rgba(245, 166, 107, 0.22), transparent 70%),
            radial-gradient(720px 520px at 70% 105%, rgba(47, 182, 138, 0.14), transparent 70%),
            radial-gradient(600px 420px at 0% 95%, rgba(232, 113, 159, 0.14), transparent 70%),
            var(--bg);
        background-attachment: fixed;
    }

    footer, #MainMenu { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    .block-container {
        max-width: 1200px;
        padding-top: 2.6rem;
        padding-bottom: 3rem;
    }

    /* ---------- Hero ---------- */
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.4rem 0.95rem 0.4rem 0.8rem;
        border-radius: 999px;
        background: var(--glass-strong);
        border: 1px solid var(--stroke);
        box-shadow: 0 2px 10px rgba(20, 20, 43, 0.05);
        color: var(--ink-soft);
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 1.3rem;
        backdrop-filter: blur(12px);
    }
    .pill .dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: var(--mint);
        box-shadow: 0 0 0 0 rgba(47, 182, 138, 0.55);
        animation: pulse 2.2s infinite;
    }
    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(47, 182, 138, 0.55); }
        70%  { box-shadow: 0 0 0 9px rgba(47, 182, 138, 0); }
        100% { box-shadow: 0 0 0 0 rgba(47, 182, 138, 0); }
    }
    .stApp .hero-title {
        font-family: 'Instrument Serif', Georgia, serif !important;
        font-weight: 400;
        font-size: 4.3rem;
        line-height: 1.02;
        letter-spacing: -0.025em;
        color: var(--ink);
        margin: 0 0 1.1rem 0;
    }
    .stApp .hero-title em {
        font-style: italic;
        background: linear-gradient(95deg, var(--violet) 0%, var(--rose) 55%, var(--peach) 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-right: 0.08em;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: var(--muted);
        max-width: 560px;
        line-height: 1.7;
        margin-bottom: 2.2rem;
    }

    /* ---------- Glass cards (Streamlit containers via key) ---------- */
    .st-key-input_card, .st-key-summary_card {
        background: var(--glass);
        backdrop-filter: blur(22px) saturate(170%);
        -webkit-backdrop-filter: blur(22px) saturate(170%);
        border: 1px solid var(--stroke);
        border-radius: var(--radius);
        box-shadow: var(--shadow), inset 0 1px 0 rgba(255, 255, 255, 0.9);
        padding: 1.5rem 1.6rem 1.6rem 1.6rem;
    }
    .card-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.9rem;
    }
    .card-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: var(--ink);
        letter-spacing: -0.01em;
    }
    .card-tag {
        font-size: 0.74rem;
        font-weight: 700;
        color: var(--violet);
        background: rgba(109, 94, 245, 0.10);
        border-radius: 999px;
        padding: 0.2rem 0.65rem;
    }

    /* ---------- Input ---------- */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.75) !important;
        border: 1px solid var(--line) !important;
        border-radius: 16px !important;
        padding: 1.1rem 1.2rem !important;
        font-size: 0.98rem !important;
        line-height: 1.75 !important;
        color: var(--ink) !important;
        box-shadow: inset 0 1px 2px rgba(20, 20, 43, 0.03);
    }
    .stTextArea textarea:focus {
        border-color: var(--violet) !important;
        box-shadow: 0 0 0 4px rgba(109, 94, 245, 0.14) !important;
    }
    [data-testid="stCaptionContainer"] { color: var(--muted); }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border-radius: 14px;
        height: 3.1rem;
        font-weight: 700;
        letter-spacing: -0.005em;
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.8);
        color: var(--ink);
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }
    .stButton > button:hover {
        border-color: var(--violet);
        color: var(--violet);
        transform: translateY(-1px);
    }
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #1B1B3D 0%, #2E2A66 100%);
        border: none;
        color: #fff;
        box-shadow: 0 10px 24px rgba(27, 27, 61, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.18);
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background: linear-gradient(135deg, #2A2A63 0%, #4B3FD1 100%);
        color: #fff;
        box-shadow: 0 14px 30px rgba(75, 63, 209, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.2);
    }

    /* Example chips */
    .st-key-ex_1 button, .st-key-ex_2 button, .st-key-ex_3 button {
        height: 2.2rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 999px;
        color: var(--ink-soft);
        background: rgba(255, 255, 255, 0.7);
    }
    .chip-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: var(--muted);
        margin: 0.2rem 0 0.35rem 0.1rem;
    }

    /* ---------- Summary ---------- */
    .summary-text {
        font-size: 1.28rem;
        line-height: 1.7;
        font-weight: 600;
        letter-spacing: -0.012em;
        color: var(--ink);
        padding: 1.2rem 1.3rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(109, 94, 245, 0.08), rgba(232, 113, 159, 0.08));
        border: 1px solid rgba(109, 94, 245, 0.14);
        animation: fadeUp 0.5s ease both;
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .empty-state {
        border: 1.5px dashed rgba(20, 20, 43, 0.14);
        border-radius: 18px;
        padding: 2.8rem 1.5rem;
        text-align: center;
        color: var(--muted);
        line-height: 1.7;
        background: rgba(255, 255, 255, 0.45);
    }
    .empty-state .spark { font-size: 1.6rem; display: block; margin-bottom: 0.4rem; }

    .mini-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.8rem;
        margin-top: 0.9rem;
    }
    .mini {
        background: rgba(255, 255, 255, 0.75);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 0.9rem 1.1rem;
    }
    .mini .k { font-size: 0.78rem; color: var(--muted); font-weight: 600; }
    .mini .v {
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: var(--ink);
        margin-top: 0.1rem;
    }

    /* ---------- Bento grid ---------- */
    .section-label {
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
        margin: 2.6rem 0 0.9rem 0.2rem;
    }
    .bento {
        display: grid;
        grid-template-columns: repeat(12, 1fr);
        gap: 1rem;
    }
    .tile {
        background: var(--glass);
        backdrop-filter: blur(22px) saturate(170%);
        -webkit-backdrop-filter: blur(22px) saturate(170%);
        border: 1px solid var(--stroke);
        border-radius: var(--radius);
        box-shadow: var(--shadow), inset 0 1px 0 rgba(255, 255, 255, 0.9);
        padding: 1.4rem 1.5rem;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 150px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .tile:hover { transform: translateY(-3px); }
    .c3 { grid-column: span 3; }
    .c4 { grid-column: span 4; }
    .c5 { grid-column: span 5; }
    .c7 { grid-column: span 7; }
    .r2 { grid-row: span 2; }

    .tile .label { font-size: 0.82rem; font-weight: 700; color: var(--muted); }
    .tile .value {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        color: var(--ink);
        line-height: 1.05;
    }
    .tile .hint { font-size: 0.82rem; color: var(--muted); margin-top: 0.3rem; }
    .bar {
        height: 7px;
        border-radius: 999px;
        background: rgba(20, 20, 43, 0.07);
        overflow: hidden;
        margin-top: 0.9rem;
    }
    .bar > span {
        display: block;
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--violet), var(--rose));
    }

    /* Featured (dark) tile */
    .tile.feature {
        background:
            radial-gradient(420px 280px at 100% 0%, rgba(232, 113, 159, 0.45), transparent 70%),
            radial-gradient(380px 260px at 0% 100%, rgba(109, 94, 245, 0.55), transparent 70%),
            #14142B;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 24px 50px rgba(20, 20, 43, 0.28);
    }
    .tile.feature .label { color: rgba(255, 255, 255, 0.7); }
    .tile.feature .value {
        color: #fff;
        font-family: 'Instrument Serif', Georgia, serif !important;
        font-weight: 400;
        font-size: 6rem;
        letter-spacing: -0.03em;
    }
    .tile.feature .hint { color: rgba(255, 255, 255, 0.65); font-size: 0.9rem; line-height: 1.6; }
    .tile.feature .bar { background: rgba(255, 255, 255, 0.16); }
    .tile.feature .bar > span { background: linear-gradient(90deg, #fff, #F5C5DA); }

    /* Info tiles */
    .tile.info { justify-content: flex-start; min-height: 0; }
    .tile.info h4 {
        margin: 0 0 0.8rem 0;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: var(--ink);
    }
    .tile.info p { color: var(--muted); line-height: 1.7; margin: 0 0 0.9rem 0; font-size: 0.95rem; }
    .tile.info ul { margin: 0; padding: 0; list-style: none; }
    .tile.info li {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        padding: 0.42rem 0;
        color: var(--ink-soft);
        font-size: 0.95rem;
        font-weight: 500;
    }
    .tile.info li::before {
        content: "";
        width: 7px; height: 7px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--violet), var(--rose));
        flex-shrink: 0;
    }
    .kv {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        padding: 0.62rem 0;
        border-bottom: 1px solid var(--line);
        font-size: 0.93rem;
    }
    .kv:last-child { border-bottom: none; }
    .kv span:first-child { color: var(--muted); }
    .kv span:last-child { font-weight: 700; text-align: right; color: var(--ink); }
    code {
        background: rgba(109, 94, 245, 0.10) !important;
        color: var(--violet) !important;
        border-radius: 8px;
        padding: 0.12rem 0.5rem !important;
        font-size: 0.82rem !important;
    }

    @media (max-width: 900px) {
        .bento > * { grid-column: span 12 !important; grid-row: auto !important; }
        .stApp .hero-title { font-size: 3rem; }
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.72);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] h3 {
        color: var(--ink);
        letter-spacing: -0.01em;
        font-weight: 800;
    }
    .side-label {
        font-size: 0.76rem;
        font-weight: 700;
        color: var(--muted);
        margin: 0.9rem 0 0.3rem 0;
    }
    .side-value {
        font-size: 0.9rem;
        font-weight: 600;
        padding: 0.55rem 0.8rem;
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid var(--line);
        border-radius: 12px;
        word-break: break-all;
        color: var(--ink);
    }

    hr { border-color: var(--line) !important; }
    .footer {
        text-align: center;
        color: var(--muted);
        font-size: 0.85rem;
        padding-top: 3rem;
    }
    .footer b { color: var(--ink-soft); }
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

EXAMPLES = {
    "ex_1": (
        "Plans",
        "Jake: Are we still on for dinner tonight?\n"
        "Mia: Yes! Which place did you book?\n"
        "Jake: The Italian one near the station, 8 PM.\n"
        "Mia: Perfect, I'll bring my sister too.\n"
        "Jake: Great, I'll change the booking to three people.",
    ),
    "ex_2": (
        "Work",
        "Omar: The client moved the deadline to Friday.\n"
        "Lena: That's two days earlier than planned.\n"
        "Omar: I know. Can you finish the report by Thursday noon?\n"
        "Lena: Yes, but I need the final numbers from Sam today.\n"
        "Omar: I'll message him right now.",
    ),
    "ex_3": (
        "Shopping",
        "Tom: I'm at the store. Do we need anything besides milk?\n"
        "Ana: Eggs and bread, please.\n"
        "Tom: Got it. What about coffee?\n"
        "Ana: We still have plenty, skip it.\n"
        "Tom: Okay, milk, eggs and bread then.",
    ),
}

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


def load_example(key: str):
    st.session_state.dialogue = EXAMPLES[key][1]
    st.session_state.summary = None
    st.session_state.last_dialogue = ""


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
    html_block(
        """
        <div class="pill"><span class="dot"></span>Fine-tuned on SAMSum &middot; BART Large CNN</div>
        <div class="hero-title">Turn any conversation<br>into a <em>clear summary</em></div>
        <div class="hero-subtitle">
            Paste a chat or dialogue and get a short, readable summary
            from a fine-tuned BART Large CNN model.
        </div>
        """
    ),
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

    with st.container(key="input_card"):

        st.markdown(
            '<div class="card-head"><span class="card-title">Dialogue</span>'
            '<span class="card-tag">Input</span></div>',
            unsafe_allow_html=True,
        )

        dialogue = st.text_area(
            "Conversation to summarize",
            key="dialogue",
            height=300,
            placeholder=(
                "Alice: Are you coming to the meeting?\n"
                "Bob: Yes, I'll be there at 3 PM."
            ),
            label_visibility="collapsed",
        )

        input_words = len(dialogue.split()) if dialogue.strip() else 0
        st.caption(f"{input_words} words")

        st.markdown('<div class="chip-label">Try an example</div>', unsafe_allow_html=True)
        chip_cols = st.columns(3)
        for col, (key, (label, _)) in zip(chip_cols, EXAMPLES.items()):
            with col:
                st.button(
                    label,
                    key=key,
                    on_click=load_example,
                    args=(key,),
                    use_container_width=True,
                )

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

    with st.container(key="summary_card"):

        st.markdown(
            '<div class="card-head"><span class="card-title">Summary</span>'
            '<span class="card-tag">Output</span></div>',
            unsafe_allow_html=True,
        )

        if st.session_state.summary:

            safe_summary = html.escape(st.session_state.summary)

            summary_words = len(st.session_state.summary.split())
            original_words = len(st.session_state.last_dialogue.split())

            compression_ratio = (
                max(0.0, (1 - summary_words / original_words) * 100)
                if original_words > 0
                else 0
            )

            st.markdown(
                html_block(
                    f"""
                    <div class="summary-text">{safe_summary}</div>
                    <div class="mini-grid">
                        <div class="mini"><div class="k">Summary words</div><div class="v">{summary_words}</div></div>
                        <div class="mini"><div class="k">Shorter than original by</div><div class="v">{compression_ratio:.0f}%</div></div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

        else:
            st.markdown(
                html_block(
                    """
                    <div class="empty-state">
                        <span class="spark">✨</span>
                        Your summary will appear here.<br>
                        Add a dialogue and select <b>Generate summary</b>.
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )


# =========================================================
# Model Performance (Bento)
# =========================================================

st.markdown('<div class="section-label">Model performance</div>', unsafe_allow_html=True)

st.markdown(
    html_block(
        """
        <div class="bento">
            <div class="tile feature c5 r2">
                <div>
                    <div class="label">ROUGE-1</div>
                    <div class="hint">Unigram overlap with the reference summaries on the SAMSum test set.</div>
                </div>
                <div>
                    <div class="value">40.58</div>
                    <div class="bar"><span style="width:40.58%"></span></div>
                </div>
            </div>
            <div class="tile c3">
                <div class="label">ROUGE-2</div>
                <div><div class="value">20.12</div><div class="bar"><span style="width:20.12%"></span></div></div>
            </div>
            <div class="tile c4">
                <div class="label">ROUGE-L</div>
                <div><div class="value">31.03</div><div class="bar"><span style="width:31.03%"></span></div></div>
            </div>
            <div class="tile c3">
                <div class="label">Test samples</div>
                <div><div class="value">819</div><div class="hint">held-out dialogues</div></div>
            </div>
            <div class="tile c4">
                <div class="label">Training samples</div>
                <div><div class="value">14,731</div><div class="hint">fine-tuning dialogues</div></div>
            </div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)


# =========================================================
# Project Information (Bento)
# =========================================================

st.markdown('<div class="section-label">Project</div>', unsafe_allow_html=True)

st.markdown(
    html_block(
        f"""
        <div class="bento">
            <div class="tile info c5">
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
            <div class="tile info c7">
                <h4>Technical details</h4>
                <div class="kv"><span>Base model</span><span><code>facebook/bart-large-cnn</code></span></div>
                <div class="kv"><span>Dataset</span><span><code>knkarthick/samsum</code></span></div>
                <div class="kv"><span>Input limit</span><span>512 tokens</span></div>
                <div class="kv"><span>Output limit</span><span>{max_summary_length} tokens</span></div>
                <div class="kv"><span>Beam search</span><span>{num_beams}</span></div>
                <div class="kv"><span>No-repeat n-gram</span><span>3</span></div>
                <div class="kv"><span>Runtime</span><span>PyTorch + Transformers</span></div>
            </div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)


# =========================================================
# Footer
# =========================================================

st.markdown(
    '<div class="footer"><b>SAMSum Dialogue Summarizer</b> &middot; '
    "Fine-tuned transformer NLP application.</div>",
    unsafe_allow_html=True,
)
