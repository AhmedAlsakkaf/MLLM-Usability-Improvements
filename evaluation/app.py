"""
Local web app for the extended usability study.

Start it by double-clicking start_app.bat in the repository root, or run
from the repository root:

    python -m streamlit run evaluation/app.py

It opens in your browser at http://localhost:8501. Nothing leaves your
computer except the requests to the model you choose.
"""

import glob
import html
import importlib.util
import json
import os
import re
import sys
import tempfile
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import extended_pipeline as ep
import video_tools
from mllm_client import MLLMClient, REPO_ROOT, load_env

DATA = os.path.join(REPO_ROOT, "data")
CLIPS = os.path.join(ep.ANIMATION_DIR, "clips")
RESULTS = os.path.join(ep.ANIMATION_DIR, "results")
UPLOAD_TYPES = [e.lstrip(".") for e in ep.VIDEO_EXTENSIONS]
MODELS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite",
          "gemini-4-hydrogen", "gemini-4-argon", "gemini-4-carbon", "gemini-3.1-pro-preview"]
PRESET_QUESTIONS = [
    "(write your own)",
    "What is the name of this animation or transition, and what are other names for it?",
    "Describe this animation step by step: what moves, from where to where, how long it takes and which easing it uses.",
    "How can I build this exact animation? Give HTML and CSS (and JavaScript only if needed).",
    "Which colours are used in this animation? Give hex values and say where each one appears.",
    "Is this animation good for usability? What would you change and why?",
    "Which library or technique (CSS transitions, keyframes, GSAP, Framer Motion, Lottie) fits best here, and why?",
]

st.set_page_config(page_title="UI Analysis Lab", layout="wide")
load_env()

ACCENT = "#f5b22b"
INK = "#141414"

# The "marker": a hand-drawn highlighter stroke in the accent colour, slightly
# uneven at the ends and swiped in from the left. Used for the active nav item
# and for one word in each page headline.
st.markdown(f"""
<style>
:root {{ --accent: {ACCENT}; --ink: {INK}; }}

/* Inter reads best slightly tightened; inputs don't inherit it, so name them. */
html, body, p, span, label, li, input, textarea, button, [data-baseweb] {{ letter-spacing: -0.018em; }}

/* Labels: small and quiet, so they never compete with what you type. */
[data-testid="stWidgetLabel"] p {{ font-size: .8rem; color: #ecebe78c; font-weight: 500; letter-spacing: -0.005em; }}
/* Placeholders: faint, clearly not real content. */
input::placeholder, textarea::placeholder {{ color: #ecebe740 !important; opacity: 1; }}
[data-baseweb="select"] [aria-selected="false"] + div, [data-baseweb="select"] div[class*="placeholder"] {{ color: #ecebe740 !important; }}
::selection {{ background: var(--accent); color: var(--ink); }}

[data-testid="stAppDeployButton"], [data-testid="stDecoration"] {{ display: none; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 3.2rem; max-width: 1180px; }}

@keyframes swipe {{ from {{ background-size: 0% 100%; }} to {{ background-size: 100% 100%; }} }}
.marker, [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
  background-image: linear-gradient(100deg, transparent 0.6%, var(--accent) 2.2%,
                    #f5b22bee 6%, var(--accent) 92%, #f5b22bcc 96.5%, transparent 99%);
  background-repeat: no-repeat; background-size: 100% 100%;
  animation: swipe .45s cubic-bezier(.65,.05,.36,1) both;
  color: var(--ink);
  border-radius: .3em .7em .35em .6em;
  box-decoration-break: clone; -webkit-box-decoration-break: clone;
}}
.marker {{ padding: 0 .18em; margin: 0 -.05em; }}

.headline {{
  font-size: clamp(2.2rem, 4.4vw, 3.4rem); font-weight: 500; line-height: 1.08;
  letter-spacing: -0.045em; margin: 0 0 1.6rem; color: var(--text-color, #ecebe7);
}}

/* ---- left navigation ---- */
[data-testid="stSidebar"] {{ border-right: 1px solid #ffffff0d; }}
.brand {{ font-size: 1.05rem; font-weight: 600; letter-spacing: -0.035em; margin: .2rem 0 1.4rem; }}
.brand b {{ color: var(--accent); }}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: .15rem; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
  padding: .32rem .7rem; margin: 0; width: fit-content; cursor: pointer;
  transition: color .15s, transform .15s;
}}
[data-testid="stSidebar"] [role="radiogroup"] label > div > div:not([data-testid]) {{ display: none; }}
[data-testid="stSidebar"] [role="radiogroup"] label p {{ font-size: 1.05rem; letter-spacing: -0.025em; }}
[data-testid="stSidebar"] [role="radiogroup"] label:not(:has(input:checked)):hover {{
  color: var(--accent); transform: translateX(3px);
}}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {{ color: var(--ink); font-weight: 500; }}
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(iframe[height="0"]) {{ display: none; }}
.muted {{ opacity: .55; font-size: .8rem; }}
.dot {{ display: inline-block; width: .5em; height: .5em; border-radius: 50%; margin-right: .45em; }}

/* ---- controls ---- */
button[kind="primary"], [data-testid="stBaseButton-primary"] {{ color: var(--ink) !important; font-weight: 600; }}
button[kind="primary"] p {{ color: var(--ink) !important; }}
[data-testid="stExpander"] details {{ border-color: #ffffff14; }}
h3 {{ letter-spacing: -0.035em; }}

/* ---- motion ----
   Streamlit only mounts elements that are new, so this plays when you switch
   pages or when a result appears, not on every slider move. */
@keyframes rise {{
  from {{ opacity: 0; transform: translateY(12px); filter: blur(4px); }}
  to   {{ opacity: 1; transform: none; filter: none; }}
}}
[data-testid="stMain"] [data-testid="stVerticalBlock"] > * {{
  animation: rise .55s cubic-bezier(.2,.75,.25,1) both;
}}
[data-testid="stMain"] [data-testid="stVerticalBlock"] > *:nth-child(2) {{ animation-delay: .05s; }}
[data-testid="stMain"] [data-testid="stVerticalBlock"] > *:nth-child(3) {{ animation-delay: .10s; }}
[data-testid="stMain"] [data-testid="stVerticalBlock"] > *:nth-child(4) {{ animation-delay: .15s; }}
[data-testid="stMain"] [data-testid="stVerticalBlock"] > *:nth-child(5) {{ animation-delay: .20s; }}
[data-testid="stMain"] [data-testid="stVerticalBlock"] > *:nth-child(n+6) {{ animation-delay: .25s; }}
.headline .marker {{ animation-delay: .3s; animation-duration: .55s; }}

/* While Streamlit reruns, waiting content dims softly instead of flashing. */
[data-stale="true"] {{ opacity: .55 !important; transition: opacity .35s ease; }}

button {{ transition: transform .15s ease, box-shadow .2s ease, background-color .2s ease !important; }}
button:hover {{ transform: translateY(-1px); }}
button:active {{ transform: translateY(0) scale(.98); }}
button[kind="primary"]:hover {{ box-shadow: 0 6px 22px -8px #f5b22b99; }}
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div {{ transition: border-color .2s ease, box-shadow .2s ease; }}
[data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within {{ box-shadow: 0 0 0 3px #f5b22b26; }}

/* ---- phones ---- */
@media (max-width: 640px) {{
  .block-container {{ padding: 2.8rem 1rem 3rem; }}
  [data-testid="stHorizontalBlock"] {{ gap: .9rem !important; }}
  /* Stats sit two per row instead of one huge number per row. */
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {{ flex-wrap: wrap; }}
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="stColumn"] {{
    flex: 1 1 calc(50% - .9rem) !important; min-width: calc(50% - .9rem) !important; width: auto !important;
  }}
  [data-testid="stMetricValue"] {{ font-size: 1.5rem; }}
  [data-testid="stSidebar"] [role="radiogroup"] label {{ padding: .45rem .8rem; }}  /* bigger tap targets */
}}

@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ animation: none !important; transition: none !important; }}
}}
</style>
""", unsafe_allow_html=True)


def headline(before, marked, after=""):
    """Page title with one word under the accent marker."""
    st.markdown(f"<div class='headline'>{html.escape(before)} <span class='marker'>{html.escape(marked)}</span>"
                f"{html.escape(after)}</div>", unsafe_allow_html=True)


# Optional password gate for public deployments: set APP_PASSWORD in the
# server's environment. Locally (no APP_PASSWORD) the app opens directly.
_app_password = os.environ.get("APP_PASSWORD", "")
if _app_password and not st.session_state.get("authenticated"):
    headline("UI Analysis", "Lab")
    entered = st.text_input("Password", type="password")
    if entered:
        if entered == _app_password:
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("Wrong password.")
    st.stop()


# ------------------------------------------------------------------- helpers
def save_upload(uploaded, folder):
    os.makedirs(folder, exist_ok=True)
    name = re.sub(r"[^\w.\-]+", "_", uploaded.name)
    path = os.path.join(folder, name)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path


def show_video(path):
    try:
        st.video(path)
    except Exception:
        st.caption("This video format cannot be previewed in the browser, but the model can still read it.")


@st.cache_data(show_spinner=False)
def cached_info(path, mtime):
    return video_tools.video_info(path)


@st.cache_data(show_spinner=False)
def cached_measure(path, mtime):
    return video_tools.colour_report(path), video_tools.motion_summary(path)


def mtime(path):
    return os.path.getmtime(path)


def swatches(colours):
    """Renders a row of colour chips. `colours` is a list of hex strings or palette dicts."""
    chips = []
    for c in colours:
        hex_value = c["hex"] if isinstance(c, dict) else str(c)
        if not re.fullmatch(r"#[0-9A-Fa-f]{6}", hex_value.strip()):
            continue
        note = ""
        if isinstance(c, dict):
            note = f"{c['share']:.0%}"
            if "contrast_with_background" in c:
                note += f" · {c['contrast_with_background']}:1"
        chips.append(
            f"<div style='display:inline-block;margin:0 12px 8px 0;text-align:center;font-size:12px'>"
            f"<div style='width:64px;height:40px;border-radius:6px;border:1px solid #8884;background:{hex_value}'></div>"
            f"<div>{html.escape(hex_value)}</div><div style='opacity:.7'>{html.escape(note)}</div></div>")
    if chips:
        st.markdown("".join(chips), unsafe_allow_html=True)


def severity_label(severity):
    return {"Major Issue": ":red[**Major**]", "Minor Issue": ":orange[**Minor**]"}.get(severity, f"**{severity}**")


def store_result(result, prefix):
    os.makedirs(RESULTS, exist_ok=True)
    stem = os.path.splitext(result["video"])[0]
    path = os.path.join(RESULTS, f"{stem}__{prefix}__{result['model_name']}_{datetime.now():%Y%m%d%H%M%S}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return path


def render_measured(measured):
    if measured:
        with st.expander(f"Measured motion in the clip ({len(measured)} animated changes)"):
            st.dataframe(pd.DataFrame(measured)[["start", "end", "duration", "peak_area"]].rename(columns={
                "start": "start (s)", "end": "end (s)", "duration": "duration (s)", "peak_area": "screen share"}),
                hide_index=True)


def render_animation(result):
    answer = result.get("model")
    if not isinstance(answer, dict):
        st.warning("The model did not return the expected format. Its raw answer:")
        st.text(answer)
        return
    st.markdown(f"### {answer.get('name', 'Unnamed animation')}")
    if answer.get("alternative_names"):
        st.caption("Also called: " + ", ".join(map(str, answer["alternative_names"])))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", f"{answer.get('duration_ms', '?')} ms")
    c2.metric("Easing", str(answer.get("easing", "?")))
    c3.metric("Starts at", f"{answer.get('start_time_s', '?')} s")
    c4.metric("Confidence", str(answer.get("confidence", "?")))
    st.write(answer.get("description", ""))
    effects = ", ".join(map(str, answer.get("primitive_effects", [])))
    st.markdown(f"**Effects:** {effects}  \n**Purpose:** {answer.get('purpose', '?')}")
    if answer.get("colours"):
        st.markdown("**Colours**")
        swatches(answer["colours"])
    st.markdown("**How to build it**")
    st.write(answer.get("how_to_implement", ""))
    code = answer.get("code_example")
    if code:
        st.code(code, language="html")
        if "<" in code:
            with st.expander("Live preview of the generated code"):
                components.html(code, height=380, scrolling=True)
    render_measured(result.get("measured"))


def render_answer(result):
    st.markdown(f"**Question:** {result.get('question', '')}")
    st.markdown(result.get("answer", ""))
    render_measured(result.get("measured"))


def result_label(path):
    """'03 Nov 15:30 · record2 · identify' instead of a long folder path."""
    name = os.path.splitext(os.path.basename(path))[0]
    parts = name.split("__")
    if len(parts) >= 3:
        name = f"{parts[0]} · {parts[1]}"
    else:
        app_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        name = f"{app_name} · {name.split('_gemini')[0].replace('_', ' ')}"
    return f"{datetime.fromtimestamp(mtime(path)):%d %b %H:%M} · {name}"


def get_client(model):
    if not model:
        st.error("Choose a model in the sidebar first.")
        st.stop()
    return MLLMClient(model)


# ------------------------------------------------------------------- sidebar
PAGES = ["Ask", "Evaluate", "Add recording", "Measure", "Results"]

with st.sidebar:
    st.markdown("<div class='brand'>ui analysis <b>lab</b></div>", unsafe_allow_html=True)
    page = st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")
    # On phones the sidebar covers the page, so close it after a page is picked.
    if st.session_state.get("_shown_page", page) != page:
        # A unique comment per change stops Streamlit reusing the old iframe (which would not re-run).
        st.session_state["_nav_count"] = st.session_state.get("_nav_count", 0) + 1
        components.html(f"<!-- nav {st.session_state['_nav_count']} -->" + """<script>
            const w = window.parent;
            if (w.matchMedia("(max-width: 768px)").matches) {
                const b = w.document.querySelector('[data-testid="stSidebarCollapseButton"] button');
                if (b) b.click();
            }</script>""", height=0)
    st.session_state["_shown_page"] = page

    st.markdown("<div style='height:2.5rem'></div>", unsafe_allow_html=True)
    default_model = os.environ.get("MLLM_MODEL") or MODELS[0]
    options = MODELS if default_model in MODELS else [default_model] + MODELS
    choice = st.selectbox("Model", options + ["Other model name..."], index=options.index(default_model))
    model = st.text_input("Model name") if choice == "Other model name..." else choice

    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if model.startswith("gemini"):
        key_ok = bool(gemini_key) and gemini_key != "YOUR_API_KEY_HERE"
        key_note = "key ready" if key_ok else "GEMINI_API_KEY missing"
    else:
        key_ok = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL"))
        key_note = "key ready" if key_ok else "OPENAI_API_KEY missing"
    st.markdown(f"<span class='muted'><span class='dot' style='background:{'#7bc47f' if key_ok else '#e5534b'}'>"
                f"</span>{key_note}</span>", unsafe_allow_html=True)

    example_env = os.path.join(REPO_ROOT, ".env.example")
    if os.path.exists(example_env):
        with open(example_env, encoding="utf-8") as f:
            if re.search(r"^GEMINI_API_KEY=(?!YOUR_API_KEY_HERE)\S+", f.read(), re.M):
                st.warning(".env.example contains a real API key. Keep the real key only in .env.")

# -------------------------------------------------------- page: ask about clip
if page == "Ask":
    headline("Ask about a", "clip")
    uploaded = st.file_uploader("Upload a clip", type=UPLOAD_TYPES, key="clip_upload")
    if uploaded is not None:
        upload_id = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("last_clip_upload") != upload_id:
            path = save_upload(uploaded, CLIPS)
            st.session_state["last_clip_upload"] = upload_id
            st.session_state["clip_choice"] = os.path.basename(path)

    os.makedirs(CLIPS, exist_ok=True)
    clips = sorted((f for f in os.listdir(CLIPS) if f.lower().endswith(ep.VIDEO_EXTENSIONS)),
                   key=lambda f: -mtime(os.path.join(CLIPS, f)))
    if not clips:
        st.info("No clips yet. Drop one above.")
    else:
        if st.session_state.get("clip_choice") not in clips:
            st.session_state["clip_choice"] = clips[0]
        clip = st.selectbox("Clip", clips, key="clip_choice")
        clip_path = os.path.join(CLIPS, clip)
        info = cached_info(clip_path, mtime(clip_path))

        left, right = st.columns([1, 1], gap="large")
        with left:
            show_video(clip_path)
            st.caption(f"{info['duration']} s · {info['width']}×{info['height']} px · {info['fps']} fps")
            start = end = None
            if st.checkbox("Only look at part of the clip"):
                start, end = st.slider("Part to analyse (seconds)", 0.0, float(info["duration"]),
                                       (0.0, float(info["duration"])), step=0.1)
            fps = st.slider("Frames per second the model sees", 1, 24, 10,
                            help="Most UI animations last under half a second. 10 or more is recommended; "
                                 "higher costs more.")
            ground = st.checkbox("Give the model the measured motion timings", value=True,
                                 help="Adds the times when the screen measurably moves, so the model guesses less.")

        with right:
            mode = st.segmented_control("Mode", ["Identify the animation", "Ask my own question"],
                                        default="Identify the animation", label_visibility="collapsed",
                                        key="ask_mode") or "Identify the animation"
            context = st.text_area("Context", height=80,
                                   placeholder="e.g. The menu on example.com when I click the burger icon.")
            question = ""
            if mode == "Ask my own question":
                preset = st.selectbox("Start from an example", PRESET_QUESTIONS)
                question = st.text_area("Your question", height=140,
                                        value="" if preset == PRESET_QUESTIONS[0] else preset,
                                        placeholder="e.g. What is the transition when the card expands, "
                                                    "and how do I build it in CSS?")
            go = st.button("Ask →", type="primary", use_container_width=True)

        if go:
            if mode == "Ask my own question" and not question.strip():
                st.error("Write a question first.")
            else:
                client = get_client(model)
                with st.spinner(f"Asking {client.model}…"):
                    try:
                        if mode == "Identify the animation":
                            result = ep.identify_animation(clip_path, client, context or None, fps, start, end, ground)
                            result["kind"] = "identify"
                        else:
                            result = ep.ask_about_video(clip_path, client, question.strip(), context or None,
                                                        fps, start, end, ground)
                            result["kind"] = "ask"
                    except Exception as error:
                        st.error(f"The request failed: {error}")
                        st.stop()
                saved = store_result(result, result["kind"])
                st.session_state["last_answer"] = result
                st.toast(f"Saved to {os.path.relpath(saved, REPO_ROOT)}")

        result = st.session_state.get("last_answer")
        if result and result.get("video") == clip:
            st.divider()
            if result.get("kind") == "identify":
                render_animation(result)
            else:
                render_answer(result)

# -------------------------------------------------- page: usability evaluation
elif page == "Evaluate":
    headline("Evaluate", "usability")
    apps = sorted(d for d in os.listdir(DATA) if os.path.exists(os.path.join(DATA, d, "task_description.json")))
    if not apps:
        st.info("No applications yet. Add one under Add recording.")
    else:
        app = st.selectbox("Application", apps)
        app_path = os.path.join(DATA, app)
        description = ep.load_json(os.path.join(app_path, "task_description.json"))
        tasks = [(f"{description['app_id']}_{p['persona_id']}_{t['task_id']}", p, t)
                 for p in description["personas"] for t in p["tasks"]]
        index = st.selectbox("Task", range(len(tasks)),
                             format_func=lambda i: f"{tasks[i][0]} · {tasks[i][1]['name']} · {tasks[i][2]['name']}")
        task_key, persona, task = tasks[index]
        video_path = os.path.join(app_path, task["video_file"])

        left, right = st.columns([1, 1], gap="large")
        with left:
            show_video(video_path)
        with right:
            with st.expander("Task details"):
                st.markdown(f"**App:** {description['description']}  \n**User:** {persona['description']}  \n"
                            f"**Task:** {task['description']}")
            criteria = st.segmented_control("Criteria", ["visual", "nielsen"], default="visual", key="criteria",
                                format_func=lambda c: {"visual": "Visual design",
                                                       "nielsen": "Nielsen heuristics"}[c]) or "visual"
            eval_fps = st.slider("Frames per second", 1, 15, ep.DEFAULT_FPS[criteria],
                                 key=f"eval_fps_{criteria}",
                                 help="Frames per second the model sees. The original study used 1.")
            eval_ground = st.checkbox("Add measured colours and motion", value=False)
            n_calls = len(ep.load_json(ep.CRITERIA_FILES[criteria]))
            run = st.button(f"Run evaluation · {n_calls} requests", type="primary", use_container_width=True)

        if run:
            client = get_client(model)
            bar = st.progress(0.0, text="Starting...")
            try:
                if eval_ground:
                    bar.progress(0.0, text="Measuring colours and motion...")
                evaluation = ep.run_usability_evaluation(
                    app_path, client, criteria, eval_fps, eval_ground, only_task=task_key,
                    on_progress=lambda done, total, id: bar.progress(done / total, text=f"{id} ({done + 1}/{total})"))
            except Exception as error:
                st.error(f"The evaluation failed: {error}")
                st.stop()
            bar.progress(1.0, text="Done")
            name = (f"evaluation_{criteria}{'_grounded' if eval_ground else ''}_{client.model}_"
                    f"{datetime.now():%Y%m%d%H%M%S}.json")
            ep.store_json(evaluation, app_path, name)
            st.session_state["evaluation"] = {"task": task_key, "app": app, "file": name, "data": evaluation}
            st.toast(f"Saved to data/{app}/reports/{name}")

        stored = st.session_state.get("evaluation")
        if stored and stored["task"] == task_key and stored["app"] == app:
            st.divider()
            issues = [i for e in stored["data"] for i in e["issues"] if i.get("severity") != "No Issue"]
            majors = sum(1 for i in issues if i.get("severity") == "Major Issue")
            st.subheader(f"{len(issues)} issues found, {majors} major")
            for entry in stored["data"]:
                found = [i for i in entry["issues"] if i.get("severity") != "No Issue"]
                with st.expander(f"{entry['heuristic']} · {len(found)} issues", expanded=bool(found)):
                    if not found:
                        st.write("No issues reported.")
                    for issue in found:
                        st.markdown(f"{severity_label(issue.get('severity', ''))} {issue.get('description', '')}  \n"
                                    f"*Fix:* {issue.get('recommendation', '')}")

            st.markdown("### Merge & rank")
            if importlib.util.find_spec("sentence_transformers") is None:
                st.info("Needs `pip install sentence-transformers`.")
            elif st.button("Merge and rank"):
                client = get_client(model)
                with st.spinner("Merging similar issues and ranking them..."):
                    report = ep.finalize_aggregated_report(
                        ep.aggregate_issues_by_task_and_similarity(stored["data"]), client)
                    ranked = ep.rank_report(report, client)
                report_name = stored["file"].replace("evaluation_", "report_", 1)
                ep.store_json(report, app_path, report_name)
                ep.store_json(ranked, app_path, "ranking_" + report_name)
                for n, issue in enumerate(ranked.get(task_key, []), 1):
                    st.markdown(f"**{n}.** {severity_label(issue['severity'])} {issue['description']}  \n"
                                f"*Fix:* {issue['recommendation']}  \n*Why this rank:* {issue['rank_justification']}")

# ------------------------------------------------------ page: add a recording
elif page == "Add recording":
    headline("Add a", "recording")
    st.caption("One screen recording of one task.")
    apps = sorted(d for d in os.listdir(DATA) if os.path.exists(os.path.join(DATA, d, "task_description.json")))
    recording = st.file_uploader("Screen recording", type=UPLOAD_TYPES, key="recording_upload")
    app_choice = st.selectbox("Application", apps, index=None,
                              placeholder="New application (or pick an existing one)")
    new_app = app_choice is None
    app_name = st.text_input("Application name", placeholder="e.g. MyShop") if new_app else app_choice
    app_description = st.text_area("What is the application?", height=70) if new_app else None

    personas = []
    if not new_app:
        personas = [p["name"] for p in
                    ep.load_json(os.path.join(DATA, app_choice, "task_description.json"))["personas"]]
    persona_choice = st.selectbox("Who is the user in the video?", personas, index=None,
                                  placeholder="New user type (or pick an existing one)")
    new_persona = persona_choice is None
    persona_name = st.text_input("User type", placeholder="e.g. Customer") if new_persona else persona_choice
    persona_description = (st.text_area("Who are they and what do they want?", height=70)
                           if new_persona else None)
    task_name = st.text_input("Short task name", placeholder="e.g. Checkout")
    task_description = st.text_area("What happens in the video, step by step?", height=90)

    if st.button("Add recording", type="primary"):
        required = {"recording": recording, "application name": app_name, "user type": persona_name,
                    "task name": task_name, "task description": task_description}
        if new_app:
            required["application description"] = app_description
        if new_persona:
            required["user description"] = persona_description
        missing = [k for k, v in required.items() if not v]
        if missing:
            st.error("Please fill in: " + ", ".join(missing))
        else:
            with tempfile.TemporaryDirectory() as tmp:
                temp_path = save_upload(recording, tmp)
                task_key = ep.add_recording(temp_path, app_name, app_description, persona_name,
                                            persona_description, task_name, task_description)
            st.success(f"Added as {task_key}. Find it under Evaluate.")

# ---------------------------------------------------------- page: measurement
elif page == "Measure":
    headline("Measure the", "pixels")
    st.caption("Colours, contrast and motion, straight from the video. No AI, free.")
    videos = sorted(os.path.relpath(p, REPO_ROOT) for ext in ep.VIDEO_EXTENSIONS
                    for p in glob.glob(os.path.join(DATA, "**", f"*{ext}"), recursive=True))
    if not videos:
        st.info("No videos found under data/.")
    else:
        video_rel = st.selectbox("Video", videos)
        video_abs = os.path.join(REPO_ROOT, video_rel)
        if st.button("Measure", type="primary"):
            with st.spinner("Measuring…"):
                colours, motion = cached_measure(video_abs, mtime(video_abs))
            st.session_state["measure"] = (video_rel, colours, motion)
        measured = st.session_state.get("measure")
        if measured and measured[0] == video_rel:
            _, colours, motion = measured
            st.markdown(f"### Colours · bg {colours['background']}")
            st.caption("Screen share · contrast. WCAG: 4.5:1 text, 3:1 large text and buttons.")
            swatches(colours["palette"])
            st.markdown("### Motion")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Screen changes", motion["screen_changes"])
            c2.metric("Instant", motion["instant"])
            c3.metric("Animated", motion["animated"])
            c4.metric("Median animation", f"{motion['median_animation_duration'] or 0:.2f} s")
            if motion["segments"]:
                st.dataframe(pd.DataFrame(motion["segments"])[["start", "end", "duration", "peak_area", "kind"]],
                             hide_index=True)

# -------------------------------------------------------------- page: results
elif page == "Results":
    headline("Saved", "results")
    files = glob.glob(os.path.join(RESULTS, "*.json")) + glob.glob(os.path.join(DATA, "*", "reports", "*.json"))
    files = sorted(files, key=mtime, reverse=True)
    if not files:
        st.info("No saved results yet.")
    else:
        chosen = st.selectbox("Result", files, label_visibility="collapsed",
                              format_func=result_label)
        data = ep.load_json(chosen)
        if isinstance(data, dict) and data.get("kind") == "ask":
            render_answer(data)
        elif isinstance(data, dict) and "model" in data and "video" in data:
            render_animation(data)
        with st.expander("Raw JSON"):
            st.json(data)
