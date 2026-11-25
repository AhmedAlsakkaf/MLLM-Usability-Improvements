"""
Extended usability evaluation pipeline.

Builds on the pipeline of Lubos et al. (FSE 2026) in LLM_Usability_Evaluation.ipynb
and adds:
  - any model through `MLLMClient` (one value switches the model),
  - a second criteria set for visual design (visual_design_criteria.json),
  - a configurable frame rate, so the model can see short animations,
  - optional grounding with colours and motion measured by OpenCV,
  - identification of a single animation or transition from a short clip.

Command line (run from the `evaluation/` directory):

    python extended_pipeline.py measure   ../data/EventHelpR/participant/1_access_event/access_event.mp4
    python extended_pipeline.py evaluate  --data ../data/EventHelpR/ --criteria visual --model gemini-3.8-flash
    python extended_pipeline.py aggregate --data ../data/EventHelpR/ --file evaluation_visual_gemini-3.8-flash_<timestamp>.json
    python extended_pipeline.py rank      --data ../data/EventHelpR/ --file report_visual_gemini-3.8-flash_<timestamp>.json
    python extended_pipeline.py animation my_clip.mp4 --model gemini-3.8-flash

Adding your own videos:

    python extended_pipeline.py add-recording C:/path/to/recording.mp4     # asks for app, persona and task
    python extended_pipeline.py animations                                 # every clip in ../data/animations/clips/
    python extended_pipeline.py ask my_clip.mp4 "Which transition is this and how do I build it?"

Easier: start the web app with start_app.bat in the repository root.
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime

from mllm_client import MLLMClient
import video_tools

HERE = os.path.dirname(os.path.abspath(__file__))
CRITERIA_FILES = {
    "nielsen": os.path.join(HERE, "nielsen_heuristics.json"),
    "visual": os.path.join(HERE, "visual_design_criteria.json"),
}
# Frames per second the model looks at. 1 is the Gemini default used by the original study.
DEFAULT_FPS = {"nielsen": 1, "visual": 5}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def store_json(data, data_path, name):
    os.makedirs(os.path.join(data_path, "reports"), exist_ok=True)
    path = os.path.join(data_path, "reports", name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Stored {path}")
    return path


# ------------------------------------------------------------------ evaluation
SYSTEM_PROMPTS = {
    "nielsen": """
You are a highly experienced usability expert.
Your task is to evaluate the software interface using Nielsen’s usability heuristics.
Use the provided user interaction recording and contextual descriptions to support a detailed, heuristic-based evaluation.
Assess each heuristic independently.
Base your evaluations strictly on the provided materials and context — do not make assumptions.
Provide structured findings with clearly described issues and actionable recommendations.
""",
    "visual": """
You are a highly experienced user interface and motion designer.
Your task is to evaluate the visual design of the software interface against the given design criterion.
Use the provided user interaction recording and contextual descriptions to support a detailed evaluation.
Assess each criterion independently.
Base your evaluations strictly on what is visible in the recording — do not make assumptions.
When you refer to a colour, name it and give an approximate hex value. When you refer to something that happens at a specific moment, give the timestamp in seconds.
Provide structured findings with clearly described issues and actionable recommendations.
""",
}


def get_evaluation_user_prompt(criterion_name, criterion_explanation, app_description, persona, task,
                               grounding=None):
    """Builds the per-criterion prompt. The wording follows the original notebook."""
    prompt = ""
    prompt += f"App Description:\n{app_description}\n"
    prompt += f"Persona:\n{persona}\n"
    prompt += f"User Task:\n{task}\n"
    if grounding:
        prompt += f"\n{grounding}\n"
    prompt += f"""
Heuristic Under Evaluation:
{criterion_name}

Heuristic Explanation:
{criterion_explanation}

Evaluation Instructions:
Evaluate the interface for the heuristic listed above. Follow these steps:
1. Identify any usability issues related to the heuristic and explain each issue.
2. Provide a practical, actionable recommendation to fix each issue.
3. Assign a severity level using:
   - No Issue
   - Minor Issue
   - Major Issue

Response Guideline:
Keep each issue’s analysis and recommendation concise (1–5 sentences). Avoid restating the heuristic definition.

Format your response as JSON as follows:
{{
  "heuristic": "{criterion_name}",
  "issues": [
    {{
      "description": "[Brief description of the usability issue]",
      "recommendation": "[Actionable suggestion for improvement]",
      "severity": "[No Issue | Minor Issue | Major Issue]"
    }}
    // Repeat for each issue
  ]
}}

If no issues are found:
{{
  "heuristic": "{criterion_name}",
  "issues": []
}}
Answer with JSON only.
"""
    return prompt


def run_usability_evaluation(data_path, client, criteria="nielsen", fps=None, ground=False,
                             only_task=None, on_progress=None):
    """
    Evaluates every task recording against every criterion of the chosen set.

    Args:
        data_path: Application folder containing task_description.json.
        client: An `MLLMClient`.
        criteria: "nielsen" (original 10 heuristics) or "visual" (visual design criteria).
        fps: Frames per second shown to the model (default: 1 for nielsen, 5 for visual).
        ground: If True, colours and motion measured with OpenCV are added to the prompt.
        only_task: Optional task id such as "EHR_P2_T1" to evaluate a single recording.
        on_progress: Optional function called as on_progress(done, total, id) before each request.

    Returns:
        list of model responses, one per task and criterion, each with an "id".
    """
    task_description = load_json(os.path.join(data_path, "task_description.json"))
    criteria_set = load_json(CRITERIA_FILES[criteria])
    fps = fps if fps is not None else DEFAULT_FPS[criteria]
    results = []
    n_tasks = sum(1 for p in task_description["personas"] for t in p["tasks"]
                  if not only_task or f"{task_description['app_id']}_{p['persona_id']}_{t['task_id']}" == only_task)
    total = n_tasks * len(criteria_set)

    for persona in task_description["personas"]:
        for task in persona["tasks"]:
            task_key = f"{task_description['app_id']}_{persona['persona_id']}_{task['task_id']}"
            if only_task and task_key != only_task:
                continue
            video_path = os.path.join(data_path, task["video_file"])
            grounding = video_tools.grounding_text(video_path) if ground else None
            for criterion_id, criterion in criteria_set.items():
                id = f"{task_key}_{criterion_id}"
                print(f"Processing {id}")
                if on_progress:
                    on_progress(len(results), total, id)
                user_prompt = get_evaluation_user_prompt(
                    criterion["name"], criterion["explanation"], task_description["description"],
                    persona["description"], task["description"], grounding)
                response = client.get_response_json(SYSTEM_PROMPTS[criteria], user_prompt,
                                                    video_path=video_path, fps=fps)
                if not isinstance(response, dict) or "issues" not in response:
                    print(f"  Skipped {id}: the model did not return the expected JSON.")
                    response = {"heuristic": criterion["name"], "issues": [], "raw_response": str(response)}
                response["id"] = id
                results.append(response)
    return results


# ----------------------------------------------------------------- aggregation
def extract_task_id(full_id):
    """'EHR_P1_T1_H4' -> 'EHR_P1_T1'"""
    match = re.match(r"^[A-Z]+_P\d+_T\d+", full_id)
    return match.group(0) if match else full_id


def aggregate_issues_by_task_and_similarity(data, threshold=0.7, embedding_model="all-MiniLM-L6-v2"):
    """Groups issues by task, then clusters them by cosine similarity (as in the original)."""
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer(embedding_model)

    task_groups = defaultdict(list)
    for entry in data:
        for issue in entry["issues"]:
            if issue.get("severity") == "No Issue":
                continue
            task_groups[extract_task_id(entry["id"])].append({"heuristic": entry["heuristic"], **issue})

    aggregated = {}
    for task_id, issues in task_groups.items():
        embeddings = model.encode([i["description"] for i in issues], convert_to_tensor=True)
        visited, clusters = set(), []
        for i in range(len(issues)):
            if i in visited:
                continue
            members = [i]
            visited.add(i)
            for j in range(i + 1, len(issues)):
                if j not in visited and util.cos_sim(embeddings[i], embeddings[j]).item() >= threshold:
                    members.append(j)
                    visited.add(j)
            clusters.append({
                "descriptions": [issues[m]["description"] for m in members],
                "recommendations": [issues[m]["recommendation"] for m in members],
                "severities": [issues[m]["severity"] for m in members],
                "heuristics": [issues[m]["heuristic"] for m in members],
            })
        aggregated[task_id] = clusters
    return aggregated


def majority_severity_vote(severities):
    """Majority vote; ties go to 'Major Issue' if present."""
    most_common = Counter(severities).most_common()
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        return "Major Issue" if "Major Issue" in [s[0] for s in most_common] else most_common[0][0]
    return most_common[0][0]


def summarize_with_llm(client, texts, prompt_type):
    joined = "\n".join(texts)
    if prompt_type == "description":
        prompt = f"""
Summarize the following usability issue descriptions into one clear, concise problem statement (1-5 sentences) that captures the main user pain point.
Avoid repeating examples.
Focus on the core usability problem.

Descriptions:
{joined}

Answer directly with the summary.
"""
    else:
        prompt = f"""
Summarize the following usability improvement recommendations into one actionable suggestion that resolves the identified issues.
Combine overlapping suggestions and avoid redundancy.

Recommendations:
{joined}

Answer directly with the summary.
"""
    return client.get_response_text(prompt).strip()


def finalize_aggregated_report(aggregated, client):
    """Summarises each cluster into one issue with a recommendation and a severity."""
    final = {}
    for task_id, clusters in aggregated.items():
        print(f"--- Finalizing report for task: {task_id} ({len(clusters)} clusters) ---")
        final[task_id] = []
        for cluster in clusters:
            many = len(cluster["descriptions"]) > 1
            final[task_id].append({
                "description": summarize_with_llm(client, cluster["descriptions"], "description") if many
                else cluster["descriptions"][0],
                "recommendation": summarize_with_llm(client, cluster["recommendations"], "recommendation") if many
                else cluster["recommendations"][0],
                "severity": majority_severity_vote(cluster["severities"]),
                "heuristics": sorted(set(cluster["heuristics"])),
            })
    return final


# --------------------------------------------------------------------- ranking
RANKING_SYSTEM_PROMPT = """
You are a highly experienced usability expert.
Your task is to re-rank identified usability issues by actual severity.
Consider the impact on task success and user frustration, frequency and likelihood of occurrence, and effort required to diagnose and recover.
Base your evaluations strictly on the provided materials and context — do not make assumptions.
Provide a ranked list from most to least severe.
"""


def get_ranking_user_prompt(issues):
    return f"""
Evaluate and re-rank the following usability issues by severity.
Each issue includes a description and a recommendation on how to fix the issue.

Format your response as JSON as follows:
[
    {{
      "id": "[Issue identifier]",
      "justification": "[Explanation of the issue severity.]"
    }}
    // Repeat for each issue
]

Answer with JSON only.

Usability issues to rank:
{json.dumps(issues, indent=2, ensure_ascii=False)}
"""


def rank_report(report, client):
    """Re-ranks the aggregated issues of each task by severity."""
    ranked_report = {}
    for task_id, issues in report.items():
        print(f"--- Ranking issues for task: {task_id} ---")
        to_rank = [{"id": f"I_{i}", "description": issue["description"],
                    "recommendation": issue["recommendation"]} for i, issue in enumerate(issues)]
        response = client.get_response_json(RANKING_SYSTEM_PROMPT, get_ranking_user_prompt(to_rank))
        ranked, seen = [], set()
        if isinstance(response, list):
            for item in response:
                match = re.fullmatch(r"I_(\d+)", str(item.get("id", ""))) if isinstance(item, dict) else None
                if match and int(match.group(1)) < len(issues) and int(match.group(1)) not in seen:
                    index = int(match.group(1))
                    seen.add(index)
                    ranked.append({**issues[index],
                                   "rank_justification": item.get("justification", "No justification provided.")})
        # Keep issues the model left out, so nothing is silently dropped.
        ranked += [{**issue, "rank_justification": "Not ranked by the model."}
                   for i, issue in enumerate(issues) if i not in seen]
        ranked_report[task_id] = ranked
    return ranked_report


# ------------------------------------------------------- animation identification
ANIMATION_SYSTEM_PROMPT = """
You are a senior motion designer and front-end engineer.
You are shown a short screen recording that contains a user interface animation or transition.
Describe only what is visible in the recording. If you are unsure, say so and lower your confidence.
"""


def get_animation_user_prompt(context=None, measured=None):
    taxonomy = load_json(os.path.join(HERE, "animation_taxonomy.json"))
    prompt = ""
    if context:
        prompt += f"Context given by the user:\n{context}\n\n"
    if measured:
        prompt += f"{measured}\n\n"
    prompt += f"""
Identify the main animation or transition in this recording.

Common names you may use (choose another name if none fits):
{", ".join(taxonomy["named_patterns"])}

Primitive effects: {", ".join(taxonomy["primitive_effects"])}
Purposes: {", ".join(taxonomy["purposes"])}

Format your response as JSON as follows:
{{
  "name": "[The common name designers and developers use for this animation or transition]",
  "alternative_names": ["[Other names it is known by, or similar effects it could be confused with]"],
  "primitive_effects": ["[One or more of the primitive effects listed above]"],
  "purpose": "[One of the purposes listed above]",
  "description": "[What moves or changes, from which state to which state, in 2-4 sentences]",
  "start_time_s": [When the animation starts, in seconds],
  "duration_ms": [Estimated duration in milliseconds],
  "easing": "[Estimated easing, e.g. linear, ease-out, ease-in-out, spring]",
  "colours": ["[Approximate hex values of the main colours involved]"],
  "how_to_implement": "[Which technique to use and why: CSS transition, CSS keyframes, Web Animations API, a library such as GSAP or Framer Motion, etc.]",
  "code_example": "[A short, self-contained HTML/CSS (and JS if needed) example that reproduces the effect]",
  "confidence": "[low | medium | high]"
}}
Answer with JSON only.
"""
    return prompt


def measure_for_prompt(video_path, start=None, end=None):
    """Returns (animated motion segments, the same as prompt text or None)."""
    segments = video_tools.detect_motion_segments(video_path, start=start, end=end)
    measured = [s for s in segments if s["kind"] == "animated"]
    if not measured:
        return measured, None
    text = "Measured screen changes (from the pixels):\n" + "\n".join(
        f"- {s['start']:.2f}s to {s['end']:.2f}s ({s['duration'] * 1000:.0f} ms), "
        f"up to {s['peak_area']:.0%} of the screen" for s in measured[:10])
    return measured, text


ASK_SYSTEM_PROMPT = """
You are a senior UI/UX designer, motion designer and front-end engineer.
You are shown a screen recording of a user interface. Answer the user's question about it.
Base your answer only on what is visible in the recording. If something cannot be seen or you are unsure, say so.
When you refer to a moment, give the timestamp in seconds. When you refer to a colour, give an approximate hex value.
When the user asks how to build something, include a short, self-contained code example.
Answer in Markdown.
"""


def ask_about_video(video_path, client, question, context=None, fps=10, start=None, end=None, ground=False):
    """
    Asks the model a free question about a clip or recording.

    Args:
        question: Anything, e.g. "Which transition is used when the card opens, and how do I build it in CSS?"
        context: Optional background, e.g. the website or what you clicked.
        fps, start, end: As in `identify_animation`.
        ground: If True, measured motion timings are added to the prompt.

    Returns:
        dict with the question, the model's Markdown answer under "answer", and the measurements.
    """
    measured, measured_text = measure_for_prompt(video_path, start, end) if ground else (None, None)
    prompt = ""
    if context:
        prompt += f"Context given by the user:\n{context}\n\n"
    if measured_text:
        prompt += f"{measured_text}\n\n"
    prompt += f"Question:\n{question}\n"
    answer = client.get_response_text(prompt, ASK_SYSTEM_PROMPT, video_path=video_path, fps=fps, start=start, end=end)
    return {"video": os.path.basename(video_path), "model_name": client.model, "fps": fps,
            "question": question, "context": context, "answer": answer, "measured": measured}


def identify_animation(video_path, client, context=None, fps=10, start=None, end=None, ground=True):
    """
    Asks the model to name and describe the animation in a clip and to explain
    how to build it.

    Args:
        video_path: A short recording of the animation (a few seconds works best).
        client: An `MLLMClient`.
        context: Optional text, e.g. "the menu on example.com when I click the burger icon".
        fps: Frames per second shown to the model. Use 10 or more: most UI animations
            last under half a second.
        start, end: Optional clip boundaries in seconds.
        ground: If True, measured motion segments are added to the prompt and returned
            next to the model's answer for comparison.

    Returns:
        dict with the model's answer under "model" and the measurements under "measured".
    """
    measured, measured_text = measure_for_prompt(video_path, start, end) if ground else (None, None)
    answer = client.get_response_json(ANIMATION_SYSTEM_PROMPT, get_animation_user_prompt(context, measured_text),
                                      video_path=video_path, fps=fps, start=start, end=end)
    return {"video": os.path.basename(video_path), "model_name": client.model, "fps": fps,
            "model": answer, "measured": measured}


# ------------------------------------------------------------ adding your videos
VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm", ".mkv", ".avi")
DATA_ROOT = os.path.join(os.path.dirname(HERE), "data")
ANIMATION_DIR = os.path.join(DATA_ROOT, "animations")


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def ask(value, question):
    """Returns `value` if given on the command line, otherwise asks in the terminal."""
    while not value:
        value = input(f"{question}: ").strip()
    return value


def add_recording(video, app=None, app_description=None, persona=None, persona_description=None,
                  task=None, task_description=None):
    """
    Registers a screen recording for usability evaluation.

    Copies the video to data/<App>/<persona>/<n>_<task>/ and adds the task to
    data/<App>/task_description.json (created if the app is new). Anything not
    passed as an argument is asked for in the terminal.

    Returns:
        The task id, e.g. "MYAPP_P1_T1", to use with `evaluate --task`.
    """
    import shutil
    if not os.path.isfile(video):
        raise FileNotFoundError(f"Video not found: {video}")
    app = ask(app, "Application name (e.g. MyShop)")
    app_dir = os.path.join(DATA_ROOT, re.sub(r"[^A-Za-z0-9]+", "", app))
    description_path = os.path.join(app_dir, "task_description.json")
    if os.path.exists(description_path):
        description = load_json(description_path)
    else:
        description = {
            "app_id": re.sub(r"[^A-Z]", "", app.upper())[:8] or "APP",
            "name": app,
            "description": ask(app_description, "What is the application? (1-2 sentences)"),
            "personas": [],
        }
    persona = ask(persona, "Who is the user in the video? (e.g. Customer)")
    entry = next((p for p in description["personas"] if p["name"].lower() == persona.lower()), None)
    if entry is None:
        entry = {"persona_id": f"P{len(description['personas']) + 1}", "name": persona,
                 "description": ask(persona_description, f"Describe the {persona} and their goal (1 sentence)"),
                 "tasks": []}
        description["personas"].append(entry)
    task = ask(task, "Short task name (e.g. Checkout)")
    number = len(entry["tasks"]) + 1
    relative = f"{slug(persona)}/{number}_{slug(task)}/{slug(task)}{os.path.splitext(video)[1].lower()}"
    target = os.path.join(app_dir, relative)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    os.makedirs(os.path.join(app_dir, "reports"), exist_ok=True)
    shutil.copy2(video, target)
    entry["tasks"].append({"task_id": f"T{number}", "name": task,
                           "description": ask(task_description, "What does the user do in the video, step by step?"),
                           "video_file": relative})
    with open(description_path, "w", encoding="utf-8") as f:
        json.dump(description, f, indent=4, ensure_ascii=False)
    task_key = f"{description['app_id']}_{entry['persona_id']}_T{number}"
    print(f"Copied the video to {target}")
    print(f"Registered task {task_key} in {description_path}")
    print(f"Next: python extended_pipeline.py evaluate --data ../data/{os.path.basename(app_dir)}/ "
          f"--criteria visual --task {task_key}")
    return task_key


def run_animation_folder(client, fps=10, redo=False):
    """
    Identifies the animation in every clip in data/animations/clips/.

    Results go to data/animations/results/<clip>__<model>.json. Clips that already
    have a result for this model are skipped unless `redo` is True. Every clip gets
    a row in data/animations/labels.csv, where you can write the correct name
    (column "label") and optional context for the model (column "context").
    """
    import csv
    clips_dir = os.path.join(ANIMATION_DIR, "clips")
    results_dir = os.path.join(ANIMATION_DIR, "results")
    labels_path = os.path.join(ANIMATION_DIR, "labels.csv")
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    clips = sorted(f for f in os.listdir(clips_dir) if f.lower().endswith(VIDEO_EXTENSIONS))
    if not clips:
        print(f"No clips found. Put short videos ({', '.join(VIDEO_EXTENSIONS)}) into {clips_dir}")
        return []

    labels = {}
    if os.path.exists(labels_path):
        with open(labels_path, newline="", encoding="utf-8") as f:
            labels = {row["clip"]: row for row in csv.DictReader(f)}
    # Drop rows of deleted clips unless someone already typed a label for them.
    labels = {k: v for k, v in labels.items() if k in clips or v.get("label")}
    for clip in clips:
        labels.setdefault(clip, {"clip": clip, "label": "", "context": ""})
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip", "label", "context"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(labels[c] for c in sorted(labels))

    results = []
    for clip in clips:
        out = os.path.join(results_dir, f"{os.path.splitext(clip)[0]}__{client.model}.json")
        if os.path.exists(out) and not redo:
            print(f"Skipping {clip} (already done)")
            continue
        print(f"Identifying animation in {clip}")
        result = identify_animation(os.path.join(clips_dir, clip), client,
                                    context=labels[clip].get("context") or None, fps=fps)
        result["label"] = labels[clip].get("label") or None
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        answer = result["model"]
        print(f"  -> {answer.get('name') if isinstance(answer, dict) else 'no valid answer'}   ({out})")
        results.append(result)
    return results


# ------------------------------------------------------------------------- CLI
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("measure", help="Measure colours and motion of a video (no API key needed)")
    p.add_argument("video")

    p = sub.add_parser("evaluate", help="Run the per-criterion evaluation")
    p.add_argument("--data", required=True)
    p.add_argument("--criteria", choices=list(CRITERIA_FILES), default="nielsen")
    p.add_argument("--model")
    p.add_argument("--fps", type=float)
    p.add_argument("--ground", action="store_true", help="Add measured colours and motion to the prompt")
    p.add_argument("--task", help="Evaluate one task only, e.g. EHR_P2_T1")

    for name, text in (("aggregate", "Merge similar issues of an evaluation file"),
                       ("rank", "Rank the issues of an aggregated report")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--data", required=True)
        p.add_argument("--file", required=True, help="File name inside <data>/reports/")
        p.add_argument("--model")

    p = sub.add_parser("animation", help="Name and describe the animation in a short clip")
    p.add_argument("video")
    p.add_argument("--model")
    p.add_argument("--context")
    p.add_argument("--fps", type=float, default=10)
    p.add_argument("--start", type=float)
    p.add_argument("--end", type=float)
    p.add_argument("--out", help="Write the result to this JSON file")

    p = sub.add_parser("ask", help="Ask your own question about a clip or recording")
    p.add_argument("video")
    p.add_argument("question")
    p.add_argument("--model")
    p.add_argument("--context")
    p.add_argument("--fps", type=float, default=10)
    p.add_argument("--start", type=float)
    p.add_argument("--end", type=float)
    p.add_argument("--ground", action="store_true", help="Add measured motion timings to the prompt")

    p = sub.add_parser("add-recording", help="Register a screen recording for usability evaluation")
    p.add_argument("video")
    p.add_argument("--app")
    p.add_argument("--app-description")
    p.add_argument("--persona")
    p.add_argument("--persona-description")
    p.add_argument("--task")
    p.add_argument("--task-description")

    p = sub.add_parser("animations", help="Identify the animation in every clip in data/animations/clips/")
    p.add_argument("--model")
    p.add_argument("--fps", type=float, default=10)
    p.add_argument("--redo", action="store_true", help="Process clips again even if a result exists")

    args = parser.parse_args()

    if args.command == "measure":
        print(json.dumps({"info": video_tools.video_info(args.video),
                          "colours": video_tools.colour_report(args.video),
                          "motion": video_tools.motion_summary(args.video)}, indent=2))
        return
    if args.command == "add-recording":
        add_recording(args.video, args.app, args.app_description, args.persona, args.persona_description,
                      args.task, args.task_description)
        return

    client = MLLMClient(args.model)
    if args.command == "animations":
        run_animation_folder(client, args.fps, args.redo)
        return
    if args.command == "ask":
        result = ask_about_video(args.video, client, args.question, args.context, args.fps,
                                 args.start, args.end, args.ground)
        print(result["answer"])
        return
    if args.command == "evaluate":
        results = run_usability_evaluation(args.data, client, args.criteria, args.fps, args.ground, args.task)
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        suffix = "_grounded" if args.ground else ""
        store_json(results, args.data, f"evaluation_{args.criteria}{suffix}_{client.model}_{stamp}.json")
    elif args.command == "aggregate":
        evaluation = load_json(os.path.join(args.data, "reports", args.file))
        report = finalize_aggregated_report(aggregate_issues_by_task_and_similarity(evaluation), client)
        store_json(report, args.data, re.sub(r"^evaluation_", "report_", args.file))
    elif args.command == "rank":
        report = load_json(os.path.join(args.data, "reports", args.file))
        store_json(rank_report(report, client), args.data, re.sub(r"^report_", "ranking_report_", args.file))
    elif args.command == "animation":
        result = identify_animation(args.video, client, args.context, args.fps, args.start, args.end)
        text = json.dumps(result, indent=2, ensure_ascii=False)
        print(text)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text)


if __name__ == "__main__":
    main()
