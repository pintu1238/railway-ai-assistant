# 🚂 Railway AI Conversational Assistant

An intelligent AI-powered chatbot for UK railway queries, built with a fine-tuned T5-small model and served via a FastAPI backend with a modern chat UI.

---

## 📁 Project Structure

```
railway-ai-assistant/
├── generate_dataset.py       # Synthetic dataset generator (1000 examples)
├── fine_tuning.py            # T5-small fine-tuning script (local GPU)
├── data/
│   ├── dataset.csv           # Full training dataset (1000 rows)
│   └── dataset_sample.csv    # Sample (100 rows)
├── railway_model/            # Saved fine-tuned model weights (after training)
├── app/
│   ├── main.py               # FastAPI application
│   ├── model_utils.py        # Model loading and inference logic
│   ├── templates/
│   │   └── index.html        # Chat UI
│   └── static/
│       └── style.css         # Styling
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone and create virtual environment
```bash
git clone <repo-url>
cd railway-ai-assistant
python -m venv railway
railway\Scripts\activate        # Windows
# source railway/bin/activate   # Mac/Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate dataset
```bash
python generate_dataset.py
```
Generates `data/dataset.csv` with **1000 diverse railway conversations** across 13 scenario types — no API key required.

### 4. Fine-tune the model
```bash
python fine_tuning.py
```
- Model: `t5-small` (~60M parameters)
- Training: 5 epochs, mixed precision (fp16 on GPU)
- Output: saved to `railway_model/`
- Time: ~15–20 min on GPU, ~1–2 hrs on CPU

### 5. Run the API
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Open: **http://localhost:8000**

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Chat UI |
| `POST` | `/chat` | Send message, get response |
| `GET`  | `/health` | Server health check |
| `GET`  | `/api/info` | Model and capability info |

### POST /chat — Request
```json
{
  "message": "Train to Manchester tomorrow with bike",
  "history": [
    {"role": "user", "content": "I need a train"},
    {"role": "assistant", "content": "Where would you like to go?"}
  ]
}
```

### POST /chat — Response
```json
{
  "response": "Got it — searching for trains to Manchester tomorrow with bike space.",
  "state": {
    "destination": "Manchester",
    "time": "tomorrow",
    "constraints": ["bike"],
    "priority": null
  },
  "model": "t5-small-railway-finetuned"
}
```

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Base model | `t5-small` (HuggingFace) |
| Parameters | 60.5M |
| Architecture | Encoder-Decoder (Seq2Seq) |
| Task | Conversational response generation |
| Training data | 1000 synthetic UK railway conversations |
| Epochs | 5 |
| Optimizer | AdamW (lr=3e-4) |
| Mixed precision | fp16 (GPU) |

---

## 📊 Dataset Details

| Scenario Type | Count |
|---------------|-------|
| Single-turn destination only | 100 |
| Destination + time | 120 |
| Destination + priority | 80 |
| Multi-turn with constraint | 150 |
| Mid-flow destination change | 100 |
| Mid-flow time change | 80 |
| Urgent queries | 70 |
| Overnight/sleeper | 50 |
| Platform queries | 50 |
| Return journeys | 60 |
| Multi-constraint | 80 |
| Full 3-turn conversations | 100 |
| Budget/cheapest queries | 60 |
| **Total** | **1000** |

---

## 💬 Example Conversations

```
User: Train to Manchester tomorrow morning
Bot:  Got it — searching for trains to Manchester tomorrow morning.
      Do you have any special requirements?

User: I have a bike
Bot:  Searching for trains to Manchester tomorrow morning with bike
      accommodation. Here are your options...

User: Actually, make it Leeds instead
Bot:  No problem — switching to Leeds instead of Manchester.
      Still travelling tomorrow morning?
```

---

## 🛠️ Tech Stack

- **Model**: HuggingFace Transformers — T5-small
- **Training**: PyTorch + mixed precision (fp16)
- **API**: FastAPI + Uvicorn
- **UI**: Vanilla HTML/CSS/JS (dark theme)
- **Dataset**: Rule-based synthetic generation (no external API)

---

## 📋 Requirements

```
transformers>=4.35.0
torch>=2.0.0
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
pandas>=2.0.0
scikit-learn>=1.3.0
sentencepiece>=0.1.99
python-dotenv>=1.0.0
jinja2>=3.1.0
python-multipart>=0.0.6
```