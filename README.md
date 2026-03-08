# Why Is My LLM Slow?

Companion code repository for the book *Why Is My LLM Slow?* — a hands-on guide
to LLM inference optimization.

## Quickstart

All practicals run on **Google Colab's free T4 GPU**. No local GPU required.

**First-time setup (do this once before Chapter 0):**

1. Open [`notebooks/00_first_time_setup.ipynb`](notebooks/00_first_time_setup.ipynb)
   in Colab by clicking **Open in Colab** below
2. Select **Runtime → Change runtime type → T4 GPU**
3. Run all cells in order

The setup notebook clones this repo into your Google Drive, installs all packages,
and downloads the LLaMA-3.2-1B weights. After it finishes, all chapter notebooks
are ready in your Drive — no re-setup needed between sessions.

**Starting each chapter:**
1. Open the chapter notebook from your Drive in Colab
2. Run the three cells at the top (mount Drive, run setup, HF login)
3. See `✓  Environment ready.` — continue with the chapter

Full instructions are in **Appendix A** of the book.

## Repository structure

```
notebooks/
  00_first_time_setup.ipynb   ← run this once before anything else
  chapter_00_*.ipynb
  chapter_01_*.ipynb
  ...
code/
  colab_setup.py              ← installs and verifies the environment
```

## Requirements

- A Google account (for Colab and Drive — both free)
- A free [Hugging Face](https://huggingface.co) account
- Meta's LLaMA license accepted at [huggingface.co/meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)

## Packages installed by `colab_setup.py`

| Package | Version |
|---|---|
| torch | 2.10 (CUDA 12.6) |
| triton | 3.x |
| bitsandbytes | 0.45+ |
| transformers | 4.47+ |
| accelerate | latest |
