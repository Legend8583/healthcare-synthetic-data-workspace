# Healthcare Synthetic Data Copilot

Healthcare Synthetic Data Copilot is a compact Streamlit app for a public healthcare synthetic-data workflow. It walks teams through a transparent six-step process:

1. Upload & Learn
2. Data Hygiene Review
3. Metadata Controls
4. Generate Synthetic Data
5. Validate Fidelity
6. Analysis Readiness & Downstream Use Cases

The app is intentionally heuristic and easy to edit. It is designed for education, prototyping, and architectural clarity rather than production-grade differential privacy.

## Repo Structure

```text
.
├── app.py
├── requirements.txt
├── README.md
├── sample_data.csv
└── src
    ├── __init__.py
    ├── explainer.py
    ├── generator.py
    ├── hygiene_advisor.py
    ├── metadata_builder.py
    ├── profiler.py
    └── validator.py
```

## What The Demo Shows

- A bundled emergency-room style CSV that loads automatically for first-run use
- Transparent transformation from source data to editable metadata to synthetic output
- Data hygiene concerns such as missingness, identifiers, and extreme wait-time outliers
- Metadata controls with a privacy-vs-fidelity slider and a wait-time scenario adjustment
- Lightweight fidelity and privacy checks for downstream decision support
- A final step that explains why the synthetic dataset is useful for further analysis
- An optional in-app AI chatbox powered by the OpenAI API when `OPENAI_API_KEY` is configured

## Local Run

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Optional: enable the in-app AI copilot chat:

   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL that Streamlit prints in your terminal.

The bundled sample data lives at `sample_data.csv`, so you can launch the app immediately without preparing your own file.

## Editing Guide

- Update `src/profiler.py` to change schema discovery and dataset profiling behavior.
- Update `src/hygiene_advisor.py` to tune quality heuristics and recommendations.
- Update `src/metadata_builder.py` to change default field strategies or exposed metadata controls.
- Update `src/generator.py` to adjust the synthetic sampling logic.
- Update `src/validator.py` to tighten fidelity or privacy checks.
- Update `src/explainer.py` to change the analysis-readiness summary and downstream use-case framing.
- Update `src/chat_assistant.py` to change the in-app AI copilot behavior or OpenAI prompt.
