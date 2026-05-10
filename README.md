# 🧠 BabyMind

> A developmental AI that starts with zero knowledge and learns from scratch — just like a human baby.

BabyMind is a research project exploring whether an AI can be taught language the same way a child learns it — starting from nothing, and growing only from what it is explicitly taught. No pre-trained weights. No internet data. Just you, teaching it word by word.

---

## 💡 The Idea

Most AIs (ChatGPT, Claude, Gemini) are trained on billions of words from the internet before you ever talk to them. They already "know" everything.

BabyMind is the opposite.

It starts as pure random noise — no language, no words, no concepts. You feed it a curriculum, like a parent teaching a child, and it learns **only** from that. This project starts with basic Hindi and English, mimicking how an Indian child would pick up language from scratch.

This is inspired by research in **developmental AI** and **tabula rasa learning**.

---

## 🚀 How It Works

1. You write training data in `input.txt` — simple words, then sentences, then concepts
2. The model (nanoGPT) trains on **only** that data
3. You chat with it via `chat.py` and observe what it has learned
4. You expand `input.txt`, retrain, and watch it grow

---

## 📂 Project Structure

```
BabyMind/
├── input.txt          # The AI's entire knowledge — you control this
├── chat.py            # Chat with the trained model
├── train.py           # Training script (from nanoGPT)
├── model.py           # GPT model architecture
├── sample.py          # Raw text generation
├── config/            # Training configs
└── data/
    └── shakespeare_char/
        ├── input.txt  # Copy of your training data goes here
        └── prepare.py # Tokenizer prep script
```

---

## 🛠️ Setup

### Requirements
- Python 3.11+
- PyTorch (with CUDA recommended)
- Windows / Linux / Mac

### Install dependencies
```bash
git clone https://github.com/YOURUSERNAME/BabyMind
cd BabyMind
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

---

## 🏋️ Training

**Step 1 — Put your training data in place:**
```bash
copy input.txt data\shakespeare_char\input.txt
```

**Step 2 — Prepare the data:**
```bash
python data/shakespeare_char/prepare.py
```

**Step 3 — Train:**
```bash
python train.py config/train_shakespeare_char.py --compile=False --always_save_checkpoint=True
```

You'll see the loss dropping as BabyMind learns:
```
iter 0:   loss 3.43   ← knows nothing
iter 100: loss 1.74   ← starting to see patterns
iter 500: loss 0.07   ← memorizing what you taught it
```

---

## 💬 Chat

```bash
python chat.py
```

```
==================================================
  BabyMind — Chat Mode
  Your AI has learned from what you taught it.
  Type anything and see what it continues.
  Type 'quit' to exit.
==================================================

You: paani
AI: paani matlab water
    water matlab paani

You: i am
AI: i am hungry
    mujhe bhookh lagi hai

You: mama
AI: mama loves me
    mama mujhse pyaar karti hai
```

---

## 📖 Training Curriculum (Baby → Adult)

The `input.txt` is structured as a developmental curriculum:

| Stage | What it learns |
|-------|---------------|
| 🍼 Newborn | Sounds, syllables — a aa i ii ka kha |
| 👶 Baby | mama, papa, paani, roti — basic survival words |
| 🧒 Toddler | Hindi-English mappings — paani matlab water |
| 👦 Child | Simple sentences — i am hungry, mujhe bhookh lagi hai |
| 🧑 Teen | Colours, numbers, emotions, actions |
| 🧑‍💻 Adult | Past/future tense, school, relationships |

To advance a stage — add more lines to `input.txt` and retrain.

---

## 🔬 Research Goal

This project is a simulation of **developmental AI** — the idea that intelligence might emerge from grounded, staged learning rather than massive pre-training. Key questions being explored:

- Can a model learn language structure from a tiny, curated curriculum?
- How does output quality change as the curriculum grows?
- What is the minimum data needed for basic bilingual understanding?

---

## ⚠️ Limitations

- BabyMind is a **character-level** model — it predicts one character at a time
- It **completes/continues** text, it does not truly "understand" or "reply"
- Small dataset = overfitting — it mostly reproduces training patterns
- To improve generalization, the curriculum must grow significantly

---

## 🙏 Credits

- Model architecture: [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
- Training curriculum: hand-crafted Hindi-English developmental data
- Research concept & implementation: Aman

---

## 📜 License

MIT License — free to use, modify, and build on.