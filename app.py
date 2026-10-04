import html
import streamlit as st
import torch

from inference.inference import load_model, summarize_dialogue


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="SAMSum BART Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# Custom CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Header */
    .hero-title {
        font-size: 2.7rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }

    /* Cards */
    .info-card {
        padding: 1rem 1.1rem;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 12px;
        background: rgba(128, 128, 128, 0.04);
        margin-bottom: 0.8rem;
    }

    .summary-card {
        padding: 1.35rem;
        border: 1px solid rgba(128, 128, 128, 0.3);
        border-radius: 14px;
        background: rgba(128, 128, 128, 0.06);
        font-size: 1.08rem;
        line-height: 1.8;
        margin-top: 0.5rem;
    }

    .metric-card {
        padding: 0.8rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }

    /* Section spacing */
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.85rem;
        padding-top: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Session State
# =========================================================

if "summary" not in st.session_state:
    st.session_state.summary = None

if "last_dialogue" not in st.session_state:
    st.session_state.last_dialogue = ""


# =========================================================
# Model Loading
# =========================================================

@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


# =========================================================
# Header
# =========================================================

st.markdown(
    '<div class="hero-title">📝 SAMSum BART Dialogue Summarizer</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-subtitle">
        Professional abstractive dialogue summarization powered by a
        fine-tuned BART Large CNN model.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.header("⚙️ Model Configuration")

    st.markdown("### Model")

    st.code(
        "facebook/bart-large-cnn",
        language="text",
    )

    st.markdown("### Task")

    st.write("Abstractive Dialogue Summarization")

    st.markdown("### Dataset")

    st.code(
        "knkarthick/samsum",
        language="text",
    )

    st.divider()

    st.markdown("### 🎛️ Generation Settings")

    num_beams = st.slider(
        "Beam Search",
        min_value=1,
        max_value=8,
        value=4,
        step=1,
        help="Higher values can improve generation quality but increase inference time.",
    )

    max_summary_length = st.slider(
        "Maximum Summary Length",
        min_value=32,
        max_value=128,
        value=96,
        step=8,
    )

    min_summary_length = st.slider(
        "Minimum Summary Length",
        min_value=4,
        max_value=32,
        value=8,
        step=4,
    )

    st.divider()

    st.markdown("### 🖥️ Runtime")

    if torch.cuda.is_available():

        st.success("GPU Available")

        st.caption(
            torch.cuda.get_device_name(0)
        )

        st.caption(
            f"CUDA: {torch.version.cuda}"
        )

    else:

        st.warning("Running on CPU")

    st.divider()

    st.markdown("### 📌 Model Limits")

    st.caption("Maximum input length: 512 tokens")
    st.caption("Maximum output length: 96 tokens")
    st.caption("No-repeat n-gram size: 3")


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

left_column, right_column = st.columns(
    [1.15, 0.85],
    gap="large",
)


# =========================================================
# Dialogue Input
# =========================================================

with left_column:

    st.markdown(
        '<div class="section-title">💬 Dialogue</div>',
        unsafe_allow_html=True,
    )

    default_dialogue = """Hannah: Do you have Betty's number?
Amanda: I can't find it. Ask Larry.
Hannah: I don't know Larry.
Amanda: He's very nice.
Hannah: Okay, I'll text him."""

    dialogue = st.text_area(
        "Enter the conversation you want to summarize:",
        value=default_dialogue,
        height=330,
        placeholder=(
            "Example:\n"
            "Alice: Are you coming to the meeting?\n"
            "Bob: Yes, I'll be there at 3 PM."
        ),
        label_visibility="collapsed",
    )

    input_words = len(dialogue.split()) if dialogue.strip() else 0

    st.caption(
        f"Input: **{input_words} words**"
    )

    button_col1, button_col2 = st.columns(
        [3, 1]
    )

    with button_col1:

        generate_clicked = st.button(
            "✨ Generate Summary",
            type="primary",
            use_container_width=True,
        )

    with button_col2:

        clear_clicked = st.button(
            "🗑️ Clear",
            use_container_width=True,
        )

    if clear_clicked:

        st.session_state.summary = None

        st.rerun()


# =========================================================
# Generate Summary
# =========================================================

if generate_clicked:

    if not dialogue.strip():

        st.warning(
            "Please enter a dialogue before generating a summary."
        )

    else:

        try:

            with st.spinner(
                "Generating abstractive summary..."
            ):

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

            st.error(
                "An error occurred during summarization."
            )

            st.exception(error)


# =========================================================
# Summary Output
# =========================================================

with right_column:

    st.markdown(
        '<div class="section-title">📄 Generated Summary</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.summary:

        safe_summary = html.escape(
            st.session_state.summary
        )

        st.markdown(
            f"""
            <div class="summary-card">
                {safe_summary}
            </div>
            """,
            unsafe_allow_html=True,
        )

        summary_words = len(
            st.session_state.summary.split()
        )

        original_words = len(
            st.session_state.last_dialogue.split()
        )

        if original_words > 0:

            compression_ratio = (
                1 - (summary_words / original_words)
            ) * 100

        else:

            compression_ratio = 0

        st.markdown("")

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:

            st.metric(
                "Summary Words",
                summary_words,
            )

        with metric_col2:

            st.metric(
                "Compression",
                f"{compression_ratio:.1f}%",
            )

        st.success(
            "Summary generated successfully."
        )

    else:

        st.info(
            "Your generated summary will appear here."
        )


# =========================================================
# Model Performance
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">📊 Model Performance</div>',
    unsafe_allow_html=True,
)

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:

    st.metric(
        "ROUGE-1",
        "40.58",
    )

with metric2:

    st.metric(
        "ROUGE-2",
        "20.12",
    )

with metric3:

    st.metric(
        "ROUGE-L",
        "31.03",
    )

with metric4:

    st.metric(
        "Test Samples",
        "819",
    )


# =========================================================
# Project Information
# =========================================================

st.divider()

about_col, technical_col = st.columns(
    2,
    gap="large",
)


# ---------------------------------------------------------
# About
# ---------------------------------------------------------

with about_col:

    with st.expander(
        "ℹ️ About This Project",
        expanded=True,
    ):

        st.markdown(
            """
            This project implements a professional
            **abstractive dialogue summarization system**
            using a fine-tuned BART Large CNN model.
            The model was fine-tuned on the **SAMSum**
            dialogue summarization dataset.
            ### Key Highlights
            - Fine-tuned `facebook/bart-large-cnn`
            - 14,731 training dialogues
            - 819 test dialogues
            - Beam search generation
            - GPU-accelerated inference
            - ROUGE-based evaluation
            - Local production-style inference pipeline
            """
        )


# ---------------------------------------------------------
# Technical Details
# ---------------------------------------------------------

with technical_col:

    with st.expander(
        "🔧 Technical Details",
        expanded=True,
    ):

        st.markdown(
            """
            **Base Model**
            `facebook/bart-large-cnn`
            **Dataset**
            `knkarthick/samsum`
            **Input Limit**
            512 tokens
            **Output Limit**
            96 tokens
            **Beam Search**
            4
            **No Repeat N-Gram**
            3
            **Runtime**
            PyTorch + Hugging Face Transformers
            """
        )


# =========================================================
# Footer
# =========================================================

st.markdown(
    """
    <div class="footer">
        SAMSum BART Dialogue Summarization System
        <br>
        Fine-tuned Transformer-based NLP Application
    </div>
    """,
    unsafe_allow_html=True,
)