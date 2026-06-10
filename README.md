#  Railway AI Conversational Assistant

An intelligent AI-powered chatbot for UK railway queries, built with a fine-tuned T5-small model and served via a FastAPI backend with a modern chat UI.

---

##  install the requirements file.



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
##  Example Conversations

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
