# Project Guide: Recommending Usability Improvements with MLLMs

A plain-language guide for a first-semester MS (AI) student. Everything below is based on the files in this repository.

## 1. What is this project about?

**Paper:** *Recommending Usability Improvements with Multimodal Large Language Models* (Lubos, Felfernig, Garber, Le, Henrich; Graz University of Technology), accepted at **FSE 2026** (ACM Foundations of Software Engineering, a top software engineering conference). arXiv: 2604.25420, DOI 10.1145/3797121.

**The problem:** Checking whether an app's user interface is easy to use ("usability") normally needs usability experts or user testing. That is expensive, and small teams often can't afford it.

**Their idea:** Give a multimodal LLM (one that understands video and text) two things:
1. a **screen recording** of someone doing a task in an app, and
2. a short **text description** of the app and the task.

The model then acts like a usability expert and returns a list of usability problems, an explanation, and a fix for each. It judges the app against **Nielsen's 10 usability heuristics**, which are standard UI design rules (for example "visibility of system status" and "error prevention").

**The pipeline (4 steps, all in `evaluation/LLM_Usability_Evaluation.ipynb`):**

| Step | What happens |
|---|---|
| 1. Setup | Load task descriptions, the heuristics, and the Gemini model |
| 2. Heuristic evaluation | For each task video, ask the model once per heuristic (10 times) to find issues |
| 3. Aggregation | Many issues are duplicates, so embed them with `all-MiniLM-L6-v2` (sentence-transformers) and cluster them by cosine similarity (threshold 0.7) |
| 4. Severity ranking | The model re-ranks the merged issues by overall impact, so developers see the most important ones first |

**Evaluation:** Software engineers rated the top-ranked recommendations (is the issue clearly stated, is it plausible, is the fix complete, would you have found it yourself?).

**Model used:** `gemini-2.0-flash-001`. The README says this model may no longer be available to new users, so you may need `gemini-2.5-flash` or newer.

## 2. Where is the dataset? Was a model trained?

**No model was trained.** This is the most important thing to understand. The project uses an existing, pre-trained model (Gemini) through an API. It is a *prompting pipeline*, not a training project, so there are no training data or weights.

The "data" is an **evaluation dataset** that the authors collected themselves. It is all in this repo:

| Folder | Contents |
|---|---|
| `data/EventHelpR/` | 6 screen-recording videos (.mp4) of 2 personas (organizer, participant) using a group decision-making app, plus `task_description.json` |
| `data/KnowledgeCheckR/` | 6 screen-recording videos of 2 personas (teacher, student) using a quiz app, plus `task_description.json` |
| `data/*/reports/` | The Gemini outputs: raw issues (`report_...json`), the evaluation (`evaluation_...json`), the ranked issues (`ranking_report_...json`) |
| `user_study/responses/` | 12 CSV files of anonymized ratings from the human study (about 4 to 20 participants per task) |
| `user_study/survey/survey_structure.md` | The survey questions |

The dataset is **small**: 2 apps, 12 task videos. These are the authors' own apps, so it is a case study, not a large benchmark. The whole repo is about 69 MB.

## 3. Is it a real research project? Could it be an MS paper topic?

**Yes, it is real.** It is peer-reviewed at a top venue, the code and data are public on GitHub and archived on Zenodo with a DOI, and the repo has the MIT license. It also earned the Available, Functional and Reusable artifact badges (see `STATUS.md`).

**For your course assignment:** it is a good choice. You can run it, see the output, and explain it. The "Results_Review" notebook works with **no API key**, so you can show outputs for free.

**For an MS thesis or paper:** this repo is a *starting point*, not a finished thesis. The paper is already published, so you can't resubmit it. A thesis would need your own new contribution. Ideas that are realistic:

- **Newer models:** the paper used Gemini 2.0 Flash. Compare it with Gemini 2.5, GPT, Claude or open models (Qwen-VL, LLaVA). Does a newer model find better issues?
- **Different or more apps:** test it on apps you build or choose, including mobile apps, and see whether it generalizes. The repo is built for this (see "Reuse" in `README.md`).
- **Other heuristics:** swap Nielsen's heuristics for accessibility rules (WCAG) or other guidelines by editing `nielsen_heuristics.json`.
- **Better aggregation or ranking:** replace the embedding clustering with something smarter, or evaluate different thresholds.
- **Prompt ablations:** does one prompt per heuristic beat one prompt for all heuristics? How much does the app context text matter?
- **Real usability experts as ground truth:** the paper used engineers as raters. Comparing against experts would be a stronger study.
- **Cost and consistency:** run the same video several times and measure how stable the answers are.

Check these ideas with your professor, and read the paper's "Limitations" and "Future work" sections, because that is where good thesis topics usually come from.

## 4. Can you modify it or add to it?

**Yes.** The MIT license lets you use, change and redistribute the code, including for academic work. Two conditions: keep the license notice, and **cite the paper** (BibTeX is at the bottom of `README.md`). Using it without credit would be plagiarism, which matters even more in an academic setting.

Easy modifications, with no coding changes beyond one variable:

1. **Change the model:** edit `GENAI_MODEL` in the notebook.
2. **Add your own app:** create `data/MyApp/`, add a `task_description.json` and screen recordings, then set `DATA_PATH`.
3. **Change heuristics:** edit `evaluation/nielsen_heuristics.json`.
4. **Change prompts:** edit the prompt strings in the notebook.
5. **Change the clustering threshold:** default 0.7, in `aggregate_issues_by_task_and_similarity`.
6. **Change the embedding model:** default `all-MiniLM-L6-v2`.

Harder: use a non-Gemini model. The code calls the Google API, so you would need to rewrite the model-calling functions for OpenAI, Anthropic or a local model. Video input also works differently across providers (some need frames extracted with OpenCV).

## 5. What to show your professor

A sensible plan for the first assignment:

1. **Summarize the paper** in 1 page: problem, method, dataset, results, limitations.
2. **Run `Results_Review.ipynb`** (free, no API key) and screenshot the outputs: the issues found, the severity rankings, and the user study ratings.
3. **Run the full pipeline on one video** (needs a free Gemini API key from Google AI Studio). Pick a short video, such as `data/EventHelpR/organizer/1_registration_and_login/`, to keep the cost tiny. Show the real output you generated.
4. **Make one small modification** and compare. For example, run it with a newer Gemini model, or on a screen recording of an app you pick, and note what changed. This shows you understood it and aren't just running someone else's code.
5. **Honest discussion:** what worked, what the limitations are (small dataset, one model, LLM can hallucinate issues), and what you would do as future work.

## 6. How to run it (Windows)

```powershell
cd "M:\MasterComsats\Advanced AI\Research\MLLM-Usability-Improvements"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name=venv
jupyter notebook
```

Then open `evaluation/Results_Review.ipynb` (no key needed) or `evaluation/LLM_Usability_Evaluation.ipynb` (set `GENAI_API_KEY` first). Python 3.9+ is required. Don't commit your API key to git.

## 7. Things to be careful about

- **Not trained, not new data:** don't describe it in your report as "trained on a dataset". Say it prompts a pre-trained MLLM and was evaluated on a small custom dataset.
- **Model availability:** `gemini-2.0-flash-001` may return a 404. Use a newer model and mention that your results may differ from the paper's.
- **LLM outputs vary** between runs, so your output won't exactly match the stored reports.
- **API cost:** small, but running every video through 10 heuristic prompts adds up. Start with one video.
- **Generalization:** results come from 2 web apps and a small number of raters. Don't over-claim.
- **Notebooks I did not execute:** this guide comes from reading the repo's documentation, file structure and data files. I haven't run the notebooks, so test them before the demo.
