import re

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ---------------------------------------------------------
# Hugging Face Model
# ---------------------------------------------------------

MODEL_ID = "AbdelrahmanAkl/SAMSum-BART-Dialogue-Summarization"


def load_model():
    """Load the fine-tuned BART model from Hugging Face."""

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID
    )

    # Explicit BART generation configuration
    model.generation_config.decoder_start_token_id = 2
    model.generation_config.bos_token_id = 0
    model.generation_config.eos_token_id = 2
    model.generation_config.pad_token_id = 1
    model.generation_config.forced_bos_token_id = 0

    model.to(device)
    model.eval()

    return tokenizer, model, device


def summarize_dialogue(
    dialogue,
    tokenizer,
    model,
    device,
    max_source_length=512,
    max_summary_length=96,
    min_summary_length=8,
    num_beams=4,
):
    """Generate a concise summary for a dialogue."""

    if dialogue is None:
        raise ValueError("Dialogue cannot be None.")

    if not isinstance(dialogue, str):
        dialogue = str(dialogue)

    dialogue = dialogue.strip()

    if not dialogue:
        raise ValueError("Dialogue cannot be empty.")

    inputs = tokenizer(
        dialogue,
        max_length=max_source_length,
        truncation=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        generated_ids = model.generate(
            **inputs,
            max_length=max_summary_length,
            min_length=min_summary_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
        )

    summary = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return re.sub(
        r"\s+",
        " ",
        summary,
    ).strip()


if __name__ == "__main__":

    tokenizer, model, device = load_model()

    sample_dialogue = """
    Hannah: Do you have Betty's number?
    Amanda: I can't find it. Ask Larry.
    Hannah: I don't know Larry.
    Amanda: He's very nice.
    Hannah: Okay, I'll text him.
    """

    summary = summarize_dialogue(
        sample_dialogue,
        tokenizer,
        model,
        device,
    )

    print("\nGenerated Summary:")
    print(summary)