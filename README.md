### Prerequisites
- **Python 3.11 or 3.12** (Python 3.15 does not work yet, because `faiss-cpu` has no build for it)
- **Git**
- **[Ollama](https://ollama.com)**, installed and running
- About 3 GB of free disk space (the model, the embedding model and the packages)

### Setup (once)
```powershell
git clone https://github.com/<owner>/documentManager.git
cd documentManager
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
ollama pull llama3.2:3b
```
On macOS or Linux, use `python3.12 -m venv .venv` and `source .venv/bin/activate`.

### Start the app
```powershell
.venv\Scripts\activate
streamlit run app.py
```
A browser tab opens at http://localhost:8501. Stop the app with Ctrl+C.

### Check it with a PDF
1. Upload a **text-based** PDF (for example `data/pdfs/auas_student_handbook.pdf`). Wait for "Indexed: <file name>". The first run also downloads the embedding model (about 90 MB).
2. Ask a question about the document. The answer cites the page, and the **Sources** box shows the passages used.
3. Ask a question the document cannot answer. The app should reply "Not found in the document." with no sources.

Example questions for the handbook:
- *How many times can I attempt an exam?* → three attempts (page 12)
- *What is the semester contribution for winter 2026/27?* → 342.50 euros (page 7)
- *What is the phone number of the International Office?* → Not found in the document.

### Run the evaluation
With Ollama running:
```powershell
python -c "from eval.run_eval import run_evaluation; from docucheck.pipeline import ask; run_evaluation(ask)"
```
Results are written to `eval/results/results.csv`. Note that this uses the index built from the last uploaded PDF, so rebuild it from the handbook first:
```powershell
python -m docucheck.retriever build data\pdfs\auas_student_handbook.pdf
```

### Troubleshooting
- **Package install fails on `faiss-cpu`:** check `python --version`; it must be 3.11 or 3.12.
- **Cannot connect to Ollama:** start the Ollama app and check `ollama list`.
- **Embedding model download stuck at 0 bytes (Windows):** run `$env:HF_HUB_DISABLE_XET = "1"` and try again.
- **Scanned PDFs are not supported** (text-based PDFs only).
