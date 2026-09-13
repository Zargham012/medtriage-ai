# MedTriage AI — Quick Start (5-7 hour build)

Everything here is 100% free: Groq API free tier, Streamlit Community Cloud,
open-source Python libraries, public MIT-BIH data.

## 0. Get your free Groq API key (2 minutes)
Go to https://console.groq.com/keys → sign up free → create a key.
Copy it — you'll need it below.

## 1. Local test run (fastest path — do this FIRST)
```bash
pip install -r requirements.txt
export GROQ_API_KEY="your_key_here"      # Mac/Linux
# set GROQ_API_KEY=your_key_here          # Windows CMD
streamlit run app.py
```
This runs immediately using the **heuristic ECG fallback** (no training needed) —
so you have a fully working end-to-end demo within minutes: ECG input → RAG
retrieval → Groq-generated triage report.

**Do this step first, before anything else.** A working ugly app beats a broken
polished one. Get this running, then improve it.

## 2. (Optional, if you have 20-30 spare minutes) Train a real ECG model
Open Google Colab, upload `ecg_model.py` and `train_ecg_colab.py`, then run:
```python
!pip install wfdb torch numpy scipy -q
!python train_ecg_colab.py
```
This downloads a small free MIT-BIH subset, trains a tiny CNN (~5-10 min on
Colab CPU), and saves `ecg_cnn.pt`. Download that file and place it in the
same folder as `app.py`. The app will automatically detect it and switch
from heuristic mode to the trained model — no code changes needed.

**If you're short on time, skip this step entirely.** The heuristic fallback
in `ecg_model.py` already produces a working demo.

## 3. Deploy for free (do this early, not last)
### Option A: Streamlit Community Cloud (recommended, easiest)
1. Push this folder to a public GitHub repo.
2. Go to https://share.streamlit.io → "New app" → connect your repo → select `app.py`.
3. In the app settings, add a secret:
   ```
   GROQ_API_KEY = "your_key_here"
   ```
4. Deploy. You'll get a public URL in ~2 minutes.

### Option B: Hugging Face Spaces
1. Create a new Space → SDK: Streamlit.
2. Upload all files.
3. In Space Settings → "Repository secrets" → add `GROQ_API_KEY`.
4. It builds and deploys automatically.

## File overview
| File | Purpose |
|---|---|
| `app.py` | Main Streamlit UI — run this |
| `ecg_model.py` | ECG classifier (trained CNN if available, else fast heuristic) |
| `train_ecg_colab.py` | Optional: train real CNN on free MIT-BIH data in Colab |
| `rag_pipeline.py` | Embeds + retrieves guideline snippets (ChromaDB, free) |
| `groq_pipeline.py` | Builds grounded prompt, calls Groq API for the triage report |
| `guidelines.txt` | Curated clinical guideline excerpts used for RAG grounding |
| `requirements.txt` | All free/open-source dependencies |

## Suggested time allocation (5-7 hrs, nothing cut)
1. **Hour 1** — Run steps 1 & 3 above: get the skeleton deployed live immediately.
2. **Hour 2** — Test with 3-4 different symptom/ECG combos, tune the Groq prompt
   in `groq_pipeline.py` if the reports need adjusting.
3. **Hour 3** (optional) — Train the real ECG model in Colab (step 2), swap it in.
4. **Hour 4** — Polish the UI (colors for urgency levels, layout), add 2-3 more
   guideline snippets to `guidelines.txt` if you want richer RAG coverage.
5. **Hours 5-7** — Buffer for debugging, redeploy, and prepare your demo pitch.

## Demo script tip
When presenting: show one **low-urgency** case and one **high-urgency** case
side by side so judges see the triage logic actually differentiating — that's
more convincing than a single happy-path demo.
