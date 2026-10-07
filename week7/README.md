# Week 7: Build Your Own Tiny LLM

[Open the Week 7 notebook in Google Colab](https://colab.research.google.com/github/rahulsharma-rs/HI-634-FoundationsOfAI/blob/main/week7/colab/build_your_own_tiny_llm.ipynb)

This lesson builds a small, character-level GPT-style transformer in PyTorch. It follows text from tokenization and next-character examples through masked attention, training, generation, full fine-tuning, and LoRA. The last section optionally downloads and fine-tunes the public `HuggingFaceTB/SmolLM2-135M` model.

## Files and versions

| File | Purpose |
| --- | --- |
| [`colab/build_your_own_tiny_llm.ipynb`](colab/build_your_own_tiny_llm.ipynb) | Current Week 7 notebook, copied from the Drive file `Build_Your_Own_Tiny_LLM-lattest.ipynb`. Run this one. |
| [`archive/build_your_own_tiny_llm_previous.ipynb`](archive/build_your_own_tiny_llm_previous.ipynb) | Earlier Drive version, kept for comparison. |

The two notebooks have the same 57 cells and the same main lesson. Only the final Hugging Face fine-tuning cell differs: the current version uses 8 epochs and a `1e-4` learning rate; the earlier one uses 3 epochs and `5e-5`. Both files retain their original code and saved outputs.

## Open and run in Google Colab

1. Open the [current notebook in Colab](https://colab.research.google.com/github/rahulsharma-rs/HI-634-FoundationsOfAI/blob/main/week7/colab/build_your_own_tiny_llm.ipynb). Sign in to your Google account if asked. You can also open [Google Colab](https://colab.research.google.com/), choose **File > Open notebook > GitHub**, and paste `https://github.com/rahulsharma-rs/HI-634-FoundationsOfAI/blob/main/week7/colab/build_your_own_tiny_llm.ipynb`.
2. Choose **File > Save a copy in Drive** so your edits and notes are saved to your account. Running the GitHub copy does not change the repository.
3. Choose **Runtime > Change runtime type** and select a **GPU** hardware accelerator if one is available. A T4 is enough for this lesson. Save the setting, connect, and run the first code cell. It should print `Running on: CUDA`. CPU also works for the main lesson, but its training loop uses 600 steps instead of 3,000 and takes longer.
4. To run the optional Part 6 in the same session, insert a new code cell near the top of your copy and run this once before the lesson:

   ```python
   %pip -q install tiktoken transformers
   ```

   Colab normally supplies PyTorch and Matplotlib. The notebook can install `tiktoken` for Part 1 itself; installing it up front avoids a pause during the run. If Colab asks for a runtime restart after installation, restart, reconnect, and rerun from the first cell.
5. Run the cells **from top to bottom** using each cell's play button. Do not jump straight to training: later cells use `text`, the vocabulary, `model`, and helper functions defined earlier. The dataset cell should report about **1,115,394 characters**, and the vocabulary cell should report **65 characters**. If the dataset cell says `Download failed, using a small built-in text instead`, reconnect and rerun that cell until the Tiny Shakespeare download succeeds before continuing. The fallback lacks characters used later in the notebook.
6. Continue through Parts 1-5. The main training cell prints training and validation loss every 10% of its steps. Expect the loss to decrease; exact values and generated text vary by runtime and random sampling. After training, inspect the generated Shakespeare-like text and compare the Q&A answers before and after full and LoRA fine-tuning.
7. Part 6 is optional. It downloads `HuggingFaceTB/SmolLM2-135M` from Hugging Face and then runs an 8-epoch demonstration on three repeated Q&A examples. It needs internet access, additional memory, and more time. Skip cells under **Part 6 (Bonus)** if you only need the from-scratch lesson. Once your setup cell has run and the source download works, **Runtime > Run all** is another way to execute the full notebook.
8. The notebook saves `tiny_gpt_pretrained.pt` in Colab's temporary runtime. To keep that checkpoint, run the following in a new cell before disconnecting:

   ```python
   from google.colab import files
   files.download("tiny_gpt_pretrained.pt")
   ```

   Keep your notebook copy in Drive as well; the checkpoint alone does not contain the code or later fine-tuned models.

## What to inspect in each part

| Part | Main code and output | What it demonstrates |
| --- | --- | --- |
| 0: Setup | `torch.cuda.is_available()` selects CUDA or CPU. | The notebook changes its main training step count with the device. |
| 1: Input | Tiny Shakespeare is downloaded; characters are mapped to integer IDs; the text is split 90/10; shifted input-target batches are built. | Next-token learning uses the next character as the target, with no manual labels. |
| 2: Architecture | An attention matrix gets a causal mask; `TinyGPT` uses token and position embeddings, four transformer blocks, and a prediction head. | Future characters are hidden from attention. The model has **824,897 parameters** with the downloaded corpus. |
| 3: Training | AdamW minimizes cross-entropy; a loss plot compares training and validation batches. | The supplied notebook's CPU run went from about **4.32/4.32** to **1.91/2.00** training/validation loss after 600 steps. These are examples, not required targets. |
| 4: Generation | Prompts are extended character by character; temperature and top-k change sampling. | Different settings change the style and variability of generated text. |
| 5: Fine-tuning | Ten authored Q&A examples are used for full fine-tuning and a hand-built LoRA adapter. | The model can learn the answer format, but the tiny dataset mostly tests memorization. The saved LoRA run has **51,200 trainable parameters**, about **5.8%** of its adapter-wrapped model. |
| 6: Bonus | Hugging Face `transformers` loads SmolLM2-135M and fine-tunes it on three repeated examples. | It contrasts the hand-built teaching model with a pretrained model; the tiny example set cannot establish generalization. |

## Practical notes

- No separate dataset file is needed. Part 1 downloads [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt). The Q&A examples are embedded in the notebook. Part 6 downloads the [SmolLM2-135M model](https://huggingface.co/HuggingFaceTB/SmolLM2-135M).
- If `Running on: CPU` appears, check the runtime setting. GPU availability varies by Colab account and current capacity; CPU is supported for Parts 0-5.
- If a cell raises `ModuleNotFoundError: No module named 'transformers'`, run the installation cell above and retry that cell. If the model download fails, retry Part 6 when internet access is available.
- The character tokenizer can only encode characters found in its training text. If a custom prompt or Q&A example raises `KeyError`, choose characters already present in the printed vocabulary or rebuild the vocabulary and model for your new text.
- Colab runtime files are temporary. Save the Drive copy and download any checkpoint you need before ending the session.
- This is a teaching exercise in language modeling, not a clinical model or a measure of real-world answer quality. The examples contain no patient records.

The Week 7 notebook is Colab-first because it downloads public text and an optional model and can take longer than this repository's offline notebook check. The repository-wide `scripts/execute_notebooks.py` intentionally covers the offline notebooks in `week*/notebooks/`; this notebook lives under `week7/colab/`.

## Sources

- [Original current notebook on Google Drive](https://drive.google.com/file/d/1k0BgN1xIej-YQniB_It5gQYa3KbphyIL/view)
- [Original earlier notebook on Google Drive](https://drive.google.com/file/d/1bI1O6UDMyy6dPtWVw4_Qznx6abIg72_O/view)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html) and [runtime information](https://research.google.com/colaboratory/runtime-version-faq.html)
- [Hugging Face Transformers installation](https://huggingface.co/docs/transformers/installation)
