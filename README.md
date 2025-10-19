<<<<<<< HEAD
# voice
=======
# Voice Grader (Streamlit)
Hands‑free grading: speak a student name and feedback; the app transcribes short chunks and maps them to a rubric in real time. Teacher edits override AI. Optional n8n webhooks fetch roster/rubric and save results to Airtable.

## Local dev
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
streamlit run app.py
```

Optional env vars (or `.streamlit/secrets.toml`):
`N8N_ROSTER_URL`, `N8N_RUBRIC_URL`, `N8N_SAVE_GRADE_URL`

## Deploy on Streamlit Community Cloud
1) Create a GitHub repo and push this code.
2) Go to https://streamlit.io/cloud → New app → select repo → Deploy.
3) Add Secrets (OPENAI_API_KEY plus any n8n URLs).

## Create & push GitHub repo
```bash
git init
git add .
git commit -m "Initial commit: Voice Grader demo"
git branch -M main
git remote add origin https://github.com/<your-username>/voice-grader.git
git push -u origin main
```
>>>>>>> 7e7d5ab (Initial commit: Voice Grader Streamlit app)
