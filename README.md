# SAMSum BART Dialogue Summarization

> **Professional Abstractive Dialogue Summarization System** built with BART Large CNN, SAMSum, PyTorch, Hugging Face Transformers, and Streamlit.

<p align="center">

**End-to-End NLP Project • Fine-Tuned Transformer • Hugging Face • Streamlit**

</p>

---

## 🚀 Project Overview

**SAMSum BART Dialogue Summarization** is an end-to-end abstractive dialogue summarization system designed to transform multi-speaker conversations into concise, fluent, and context-aware summaries.

The project fine-tunes the pretrained **`facebook/bart-large-cnn`** sequence-to-sequence Transformer model on the **SAMSum** dialogue summarization dataset.

The final system is fully deployed:

* **Source Code:** GitHub
* **Fine-Tuned Model:** Hugging Face Hub
* **Interactive Application:** Streamlit
* **Inference:** Real-time abstractive summarization

The application allows users to enter a conversation, configure generation parameters, and generate a summary using the fine-tuned BART model.

---

## 🌐 Demo & Resources

<p align="center">

<a href="https://samsum-bart-dialogue-summarization.streamlit.app/">
  <img src="https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo"/>
</a>

<a href="https://huggingface.co/AbdelrahmanAkl/SAMSum-BART-Dialogue-Summarization">
  <img src="https://img.shields.io/badge/🤗_Hugging_Face-Model-FFD21F?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face"/>
</a>

<a href="https://github.com/AbdelrhmanAkl/SAMSum-BART-Dialogue-Summarization">
  <img src="https://img.shields.io/badge/💻_GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/>
</a>

</p>
---

## ✨ Key Features

* Abstractive dialogue summarization
* Fine-tuned BART Large CNN
* SAMSum dialogue summarization dataset
* Transformer-based sequence-to-sequence architecture
* Beam Search generation
* Configurable summary length
* Minimum summary length control
* No-repeat n-gram constraint
* GPU-accelerated training
* FP16 mixed-precision training
* Gradient accumulation
* Gradient checkpointing
* Early stopping
* Best-model checkpoint selection
* ROUGE evaluation
* Baseline comparison
* Qualitative evaluation
* Error analysis
* Hugging Face model hosting
* Streamlit interactive interface
* Production-style inference pipeline

---

# 🧠 Model

### Base Model

```text
facebook/bart-large-cnn
```

### Fine-Tuned Model

```text
AbdelrahmanAkl/SAMSum-BART-Dialogue-Summarization
```

The model is fine-tuned specifically for dialogue summarization using the SAMSum dataset.

### Task

```text
Dialogue → Abstractive Summary
```

Example:

**Input**

```text
Hannah: Do you have Betty's number?
Amanda: I can't find it. Ask Larry.
Hannah: I don't know Larry.
Amanda: He's very nice.
Hannah: Okay, I'll text him.
```

**Generated Summary**

```text
Hannah will text Larry about Betty's number.
```

---

# 📊 Dataset

The project uses the **SAMSum** dataset, a benchmark dataset for abstractive dialogue summarization.

Each example contains:

* `dialogue` — multi-speaker conversation
* `summary` — human-written reference summary

### Dataset Split

| Split      |    Samples |
| ---------- | ---------: |
| Training   |     14,731 |
| Validation |        818 |
| Test       |        819 |
| **Total**  | **16,368** |

---

# 🔬 Data Preparation

Tokenization was performed using the BART tokenizer.

### Input Configuration

| Parameter             |      Value |
| --------------------- | ---------: |
| Maximum source length | 512 tokens |
| Maximum target length |  96 tokens |
| Truncation            |    Enabled |
| Dynamic padding       |    Enabled |
| Label padding         |     `-100` |
| Padding multiple      |          8 |

The source length of 512 tokens was selected after analyzing the dataset distribution.

Only a very small percentage of dialogues exceeded the selected source limit.

---

# ⚙️ Training Configuration

The model was fine-tuned using PyTorch and Hugging Face Transformers.

| Parameter                    |                     Value |
| ---------------------------- | ------------------------: |
| Base Model                   | `facebook/bart-large-cnn` |
| GPU                          |           NVIDIA Tesla T4 |
| Batch Size / GPU             |                         4 |
| Gradient Accumulation        |                         4 |
| Effective Batch Size         |                        16 |
| Learning Rate                |                    `5e-5` |
| Weight Decay                 |                    `0.01` |
| Epochs                       |                         3 |
| Warmup Steps                 |                       500 |
| Optimizer                    |                     AdamW |
| Precision                    |                      FP16 |
| Gradient Checkpointing       |                   Enabled |
| Evaluation                   |               Every Epoch |
| Generation During Evaluation |                   Enabled |
| Number of Beams              |                         4 |
| Generation Max Length        |                        96 |
| Early Stopping               |                   Enabled |
| Best Model Selection         |                   ROUGE-L |

---

# 📈 Evaluation Results

Evaluation was performed on the held-out SAMSum test set containing **819 examples**.

### Fine-Tuned Model

| Metric     |     Score |
| ---------- | --------: |
| ROUGE-1    | **40.58** |
| ROUGE-2    | **20.12** |
| ROUGE-L    | **31.03** |
| ROUGE-Lsum | **31.05** |

The results demonstrate a substantial improvement over the original pretrained BART model on the same test set.

---

# 🆚 Baseline Comparison

The fine-tuned model was compared against the original pretrained `facebook/bart-large-cnn`.

| Metric     | Baseline | Fine-Tuned | Improvement |
| ---------- | -------: | ---------: | ----------: |
| ROUGE-1    |    31.03 |  **40.58** |   **+9.55** |
| ROUGE-2    |    10.36 |  **20.12** |   **+9.76** |
| ROUGE-L    |    23.34 |  **31.02** |   **+7.67** |
| ROUGE-Lsum |    23.34 |  **31.05** |   **+7.71** |

### Key Observation

The fine-tuned model achieved a particularly strong improvement in **ROUGE-2**, indicating better overlap with reference summaries at the bigram level.

This confirms that domain-specific fine-tuning substantially improved the pretrained model's performance on the SAMSum dialogue summarization task.

---

# 🧪 Qualitative Evaluation

In addition to automatic metrics, several generated summaries were manually inspected.

The model generally demonstrated:

* Good dialogue understanding
* Good preservation of key information
* Strong grammatical fluency
* Effective compression of conversations
* Reasonable handling of multiple speakers

### Observed Weaknesses

The main limitations identified during qualitative and quantitative analysis were:

* Over-generation
* Occasional attribution errors
* Occasional unsupported details

These observations are important because ROUGE scores alone do not fully capture factuality or summary conciseness.

---

# 🔍 Error Analysis

Prediction length analysis was performed on the complete test set.

| Statistic     | Reference | Prediction |
| ------------- | --------: | ---------: |
| Mean Length   |     20.02 |      43.90 |
| Median Length |        18 |         44 |

The average prediction was approximately **2.89× longer** than the reference summary.

Approximately **73.4%** of test examples had predictions at least 1.8× the reference length.

However, this should **not** be interpreted as a hallucination rate.

The analysis indicates that the primary issue is **over-generation**, since SAMSum reference summaries can be highly compressed compared with the original dialogue.

No retraining was performed specifically for this issue; the current model was retained as the final portfolio model.

---

# 🏗️ System Architecture

```text
                         ┌──────────────────────┐
                         │      User Input      │
                         │      Dialogue        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Streamlit App     │
                         │       app.py         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Inference Pipeline  │
                         │   inference.py       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                  ┌────────────────────────────────────┐
                  │        Hugging Face Hub             │
                  │ Fine-Tuned BART Large CNN Model     │
                  └──────────────────┬─────────────────┘
                                     │
                                     ▼
                         ┌──────────────────────┐
                         │  Generated Summary   │
                         └──────────────────────┘
```

---

# 🔄 Deployment Architecture

The project separates the application source code from the large model artifacts.

```text
                 GitHub
                   │
                   │
             Source Code
                   │
                   ▼
              Streamlit
                   │
                   │
                   ▼
          inference.py
                   │
                   │
                   ▼
          Hugging Face Hub
                   │
                   │
                   ▼
       Fine-Tuned BART Model
```

The large model weights are **not stored inside the GitHub repository**.

Instead, the Streamlit application loads the fine-tuned model directly from Hugging Face.

This keeps the source repository lightweight while maintaining a deployable production-style architecture.

---

# 📁 Project Structure

```text
SAMSum-BART-Dialogue-Summarization/
│
├── inference/
│   └── inference.py
│
├── model/
│   └── Fine-tuned model files
│
├── notebooks/
│   └── QLoRA_SAMSum_Dialogue_Summarization.ipynb
│
├── app.py
├── metadata.json
├── project_summary.json
├── requirements.txt
├── .gitignore
└── README.md
```

> The large model directory is excluded from Git tracking and hosted separately on Hugging Face Hub.

---

# 🖥️ Streamlit Application

The Streamlit interface provides:

### Input

Users can enter a multi-speaker dialogue directly into the application.

### Generation Controls

* Beam Search
* Maximum Summary Length
* Minimum Summary Length

### Model Information

The application also displays:

* Model architecture
* Dataset
* Evaluation metrics
* Runtime device
* CUDA information when available
* Project information

---

# ⚡ Local Installation

Clone the repository:

```bash
git clone https://github.com/AbdelrhmanAkl/SAMSum-BART-Dialogue-Summarization.git
cd SAMSum-BART-Dialogue-Summarization
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

The fine-tuned model is automatically downloaded from Hugging Face when required.

---

# 🧩 Direct Inference

The inference module can also be executed independently:

```bash
python inference/inference.py
```

The inference pipeline automatically:

1. Loads the tokenizer
2. Loads the fine-tuned model from Hugging Face
3. Detects CUDA when available
4. Tokenizes the dialogue
5. Generates the summary
6. Decodes the generated sequence
7. Returns the final cleaned summary

---

# 🛠️ Generation Strategy

The final inference configuration uses:

```text
max_source_length = 512
max_summary_length = 96
min_summary_length = 8
num_beams = 4
early_stopping = True
no_repeat_ngram_size = 3
length_penalty = 1.0
```

This configuration provides a balance between summary quality, fluency, and generation control.

---

# ⚠️ Limitations

Despite the strong improvement over the pretrained baseline, the system has several limitations.

### 1. Over-generation

Generated summaries are generally longer than the human-written references.

### 2. Attribution Errors

In some cases, information may be associated with the wrong speaker.

### 3. Unsupported Details

The model may occasionally generate details that are not explicitly supported by the dialogue.

### 4. Model Size

BART Large is relatively large for CPU-based deployment and requires significant memory during loading and inference.

### 5. Dataset Scope

The model is optimized for the conversational style represented in SAMSum and may not generalize equally well to every type of real-world dialogue.

---

# 🚀 Future Improvements

Potential improvements include:

* Better length control
* Length-aware decoding
* Factuality evaluation
* Speaker attribution evaluation
* Hallucination detection
* Coverage-aware generation
* ROUGE + BERTScore evaluation
* Human evaluation
* Quantization for faster CPU inference
* Knowledge distillation
* Parameter-efficient fine-tuning
* Improved deployment optimization
* Larger and more diverse dialogue datasets

---

# 🧰 Technology Stack

### Machine Learning

* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* BART
* Seq2Seq Learning

### NLP

* Abstractive Summarization
* Dialogue Understanding
* Transformer Models
* Text Generation
* ROUGE Evaluation

### Deployment

* Streamlit
* Hugging Face Hub

### Development

* Python
* Jupyter Notebook
* Git
* GitHub

---

# 📚 Project Workflow

```text
1. Dataset Exploration
        ↓
2. Token Analysis
        ↓
3. Data Preparation
        ↓
4. BART Fine-Tuning
        ↓
5. Validation Evaluation
        ↓
6. Test Evaluation
        ↓
7. Baseline Comparison
        ↓
8. Qualitative Evaluation
        ↓
9. Error Analysis
        ↓
10. Model Export
        ↓
11. Hugging Face Hub
        ↓
12. Streamlit Deployment
        ↓
13. Production-Style Inference
```

---

# 📌 Reproducibility

The project configuration, inference pipeline, model metadata, and training notebook are included in the repository.

The trained model is hosted separately on Hugging Face because of its large file size.

---

# 📜 License

The project uses the Apache 2.0 licensed BART model and follows the licensing terms of the underlying pretrained model and dataset.

Please review the original dataset and model licenses before using the system for commercial applications.

---

# 👨‍💻 Author

**Abdelrahman Akl**

AI Engineer | Machine Learning | NLP | Deep Learning | LLMs

This project was developed as a professional end-to-end NLP portfolio project demonstrating the complete lifecycle of a Transformer-based machine learning application — from dataset preparation and model fine-tuning to evaluation, model hosting, and live deployment.

---

## ⭐ Project Summary

**SAMSum BART Dialogue Summarization** demonstrates how a pretrained Transformer can be adapted to a specialized NLP task and transformed into a complete deployable application.

The final solution combines:

```text
Fine-Tuned Transformer
        +
Systematic Evaluation
        +
Error Analysis
        +
Hugging Face Model Hosting
        +
Streamlit Deployment
        =
End-to-End NLP Application
```

If you find this project useful, consider giving the repository a ⭐.
