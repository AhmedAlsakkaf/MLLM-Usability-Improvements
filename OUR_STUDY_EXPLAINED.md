# Our Study, Explained Simply

Read this before you meet your supervisor. The formal version is the project report; this is the same thing in plain words.

## 1. The story in five sentences

1. A published paper (Lubos et al., FSE 2026) showed that an AI model can watch a screen recording of someone using an app and list the usability problems it sees.
2. That paper only checked *usability rules* (Nielsen's 10 heuristics), used *one* AI model (Gemini 2.0 Flash, now switched off by Google), and let the model see only *1 frame per second* of the video.
3. We took their code and added three things: you can switch the AI model, the AI also judges *visual design* (colours, fonts, layout, animations), and the AI can *name an animation* from a short clip and explain how to build it.
4. We also added a tool that measures colours and motion directly from the video pixels, without any AI, so we can check whether the AI is telling the truth.
5. Our research question in one line: **can these AI models judge how an interface looks and moves, not only how usable it is, and how accurate are they?**

## 2. Words you will need

| Word | Meaning |
|---|---|
| MLLM | Multimodal large language model. An AI that understands text plus images or video. Gemini is one. |
| Usability | How easy an app is to use. |
| Nielsen's heuristics | 10 classic rules for usable interfaces, e.g. "show the user what is happening". |
| Prompt | The instruction text we send to the AI. |
| Pipeline | A fixed series of steps. Ours: video in, list of ranked problems out. |
| FPS | Frames per second. How many still pictures of the video the AI looks at each second. |
| WCAG contrast | An accessibility standard. Text needs a contrast ratio of at least 4.5:1 against its background, large text and buttons at least 3:1. |
| Grounding | Giving the AI measured facts (real colours, real animation times) inside the prompt so it guesses less. |
| Ground truth | The correct answer we compare the AI against. |
| Baseline | The old result we compare new results with (the paper's stored Gemini 2.0 output). |

**Important:** we do not train any model. We send prompts to models that already exist and study their answers. Never say "we trained a model".

## 3. What the original paper did

1. Take a screen recording and a short description of the app, the user and the task.
2. Ask the AI 10 times, once per Nielsen rule: "what problems do you see for this rule?"
3. Many answers repeat each other, so similar ones are merged.
4. The AI sorts the merged problems from most to least serious.
5. Software engineers rated the top 3 problems per task. They found them clear and believable (about 4.0 to 4.3 out of 5).

Their data: 2 web apps, 12 recordings, 95 survey responses.

## 4. What we added, and why

| What we added | Why it matters | File |
|---|---|---|
| **Model switching** | The paper's model no longer exists. Now one setting changes the model, so we can compare several. | `evaluation/mllm_client.py` |
| **Visual design criteria** (colour and contrast, typography, layout, consistency, motion, interactive states) | The old reports almost never mention how things look or move. | `evaluation/visual_design_criteria.json` |
| **Higher frame rate** (5 to 10 FPS instead of 1) | Most animations last under half a second. At 1 FPS the AI cannot see them. | `evaluation/extended_pipeline.py` |
| **Pixel measurements** (real colours, contrast, when the screen moves) | Gives us a way to check the AI, and facts to feed it. | `evaluation/video_tools.py` |
| **Animation identification** | Give a short clip; the AI names the animation, describes it, estimates its timing and writes example code. | `evaluation/extended_pipeline.py` |

## 5. What we have found so far

These are early results. Say so when you present them.

- **The old reports ignore visual design.** Of 206 problems the paper's model reported, at most 33 mention colour, fonts, layout or motion.
- **The recordings are full of motion the old setup could not see.** Our tool found 418 screen changes in the 12 recordings. 298 of them are animated and shorter than one second.
- **Our extension runs on a real model.** We ran the visual design evaluation with `gemini-3.8-flash` on one 21-second recording (a participant opening an event in EventHelpR). It reported 15 visual design problems, 2 of them rated major. Examples:
  - a "last modified" text in dark blue on a dark blue banner, almost unreadable;
  - the "enter your event key" field sits far down the landing page, below marketing content;
  - the event page appears with a hard cut and content "pops in" with no transition.
- **Animation identification runs too.** On a 3.5-second clip of that recording the model answered "smooth scroll" and gave CSS code for it.

What we have **not** done yet: checked by eye whether those 15 problems are really in the video, run the other 11 recordings, compared several models, or collected our own clips. One run of one video proves the tool works. It does not prove the AI is accurate.

## 6. How to add your videos

You never upload anything by hand. You put a video in the project, and the code sends it to the model for you.

### Easiest: the app

Double-click `start_app.bat` in the project folder. A page opens in your browser (http://localhost:8501) with five tabs:

| Tab | What you do there |
|---|---|
| Ask about a clip | Upload or pick a clip, then either click **Identify the animation** (name, timing, easing, code) or **Ask my own question** and type anything. |
| Usability evaluation | Pick an app and a task, choose visual design or Nielsen criteria, press Run. |
| Add a recording | Upload a recording of an app and fill in a short form. |
| Measure (no AI) | Colours, contrast and motion from the pixels. Free. |
| Saved results | Every earlier answer, newest first. |

The model is chosen in the left sidebar. Close the black window to stop the app.

### The same from the command line

Open a terminal **in the `evaluation` folder** (`cd evaluation`) for all commands below. Running them from the project root gives "No such file or directory".

### A. A short animation clip (to ask "what is this animation?")

1. Record a few seconds of the animation with any screen recorder (Windows: `Win + Alt + R`, or the Snipping Tool's video mode). Shorter is better: 2 to 6 seconds.
2. Drop the file into `data/animations/clips/`.
3. Run:
   ```
   python extended_pipeline.py animations
   ```
4. Read the answer in `data/animations/results/`. Clips that already have an answer are skipped next time.
5. Optional: open `data/animations/labels.csv`. In the `label` column write the name *you* think is correct (this becomes our ground truth). In the `context` column you can give the model a hint, such as the website name.

### B. A full recording of someone using an app (for the usability and visual design evaluation)

1. Record yourself doing one task in the app, start to finish.
2. Run this with the path to your video. It asks you a few questions (app name, who the user is, what the task is):
   ```
   python extended_pipeline.py add-recording "C:\path\to\my_recording.mp4"
   ```
3. It copies the video into `data/<AppName>/`, writes the description file, and prints the exact command to run next, for example:
   ```
   python extended_pipeline.py evaluate --data ../data/MyShop/ --criteria visual --task MYSHOP_P1_T1
   ```
4. Results appear in `data/<AppName>/reports/`.

### Other useful commands

```
python extended_pipeline.py ask "..\data\animations\clips\record.mp4" "Which transition is this and how do I build it?"
python extended_pipeline.py measure "path\to\video.mp4"      # colours and motion, no AI, free
python extended_pipeline.py evaluate --data ../data/EventHelpR/ --criteria nielsen --task EHR_P2_T1   # the original 10 rules
python extended_pipeline.py evaluate ... --model gemini-4-hydrogen    # use another model
python extended_pipeline.py evaluate ... --ground             # also give the AI the measured facts
```

To change the default model, edit `MLLM_MODEL` in the `.env` file.

## 7. How to present it (about 5 minutes)

1. **Problem.** Usability testing needs experts and is expensive.
2. **Base paper.** An AI watches screen recordings and suggests fixes. Published at FSE 2026, code is public.
3. **Gap.** One model, usability rules only, 1 frame per second, and nobody checks the AI's claims against the video.
4. **Our idea.** Compare models, add visual design and motion, and check the AI against measurements.
5. **Evidence the gap is real.** 33 of 206 old issues are visual; 298 sub-second animations in the recordings.
6. **Demo.** Run `measure` on a video (free), then show the 15 visual issues from the real run.
7. **Plan.** Run all recordings on 2 to 3 models, collect and label animation clips, small rating study.
8. **Ask for guidance** (next section).

## 8. Questions to ask your supervisor

Their answers decide the scope, so write them down.

- Is this a **course project** or should it grow into a **thesis or paper**? (This changes how much evaluation is needed.)
- Is extending a published paper acceptable, and how should we cite it?
- Which of the three additions matters most to them? Should we do all three or go deep on one?
- How many models and how many videos are enough?
- Do we need **human raters** (a survey like the original paper), and who could they be?
- What should be submitted, in which format, and when? (Report template, page limit, deadline, demo.)
- Is there budget for API costs, or should we stay on the free tier?

## 9. Questions they may ask you

| Question | Honest answer |
|---|---|
| Did you train a model? | No. We prompt existing models and analyse the output. |
| What is new compared with the paper? | Model comparison, visual design and motion criteria, measurements to check the AI, animation naming. |
| How do you know the AI is right? | That is the research. We compare its colours and timings with pixel measurements, and its animation names with our own labels. Human raters judge the rest. |
| Can the AI really see colours and animations? | Partly. Published benchmarks (ColorBench, AniMINT) say models are weak at colour and at understanding what an animation is for. We measure how weak, on real interfaces. |
| Why not just use the paper's model? | Google shut it down. This is also why being able to switch models matters. |
| Is the dataset big enough? | No, it is small: 2 apps and 12 videos from the paper. We plan to add our own. |
| Are results repeatable? | Not exactly. The AI gives slightly different answers each run, so we run everything three times. |
| Has someone done this already? | Parts of it. AniMINT (ACL 2026) tests animation understanding on isolated clips. We have not found work that combines usability evaluation of recordings with visual design and pixel checks. We still need to read those papers fully. |

## 10. Keep these safe

- Your API key belongs only in the `.env` file. Never paste it into a notebook, the report, or `.env.example`.
- Cite the original paper in everything. Their code is MIT licensed, so reuse is allowed with credit.
