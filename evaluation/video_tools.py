"""
Objective measurements on screen recordings with OpenCV (no model, no API key).

These serve two purposes in the extended study:
  1. Grounding: measured colours and motion segments can be added to the prompt.
  2. Checking: the model's statements about colour and animation can be compared
     with what is measurably in the pixels.
"""

import cv2
import numpy as np


# ------------------------------------------------------------------ video I/O
def video_info(video_path):
    """Returns width, height, fps, frame count and duration (seconds) of a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(fps, 2),
        "frames": frames,
        "duration": round(frames / fps, 2),
    }
    cap.release()
    return info


def iter_frames(video_path, fps=None, start=None, end=None, width=None):
    """
    Yields (timestamp_seconds, BGR frame) at roughly `fps` frames per second.
    `width` optionally downsizes frames (keeps aspect ratio).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(native / fps)) if fps else 1
    index = 0
    if start:
        index = int(start * native)
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = index / native
        if end is not None and t > end:
            break
        if (index - (int(start * native) if start else 0)) % step == 0:
            if width and frame.shape[1] > width:
                frame = cv2.resize(frame, (width, int(frame.shape[0] * width / frame.shape[1])),
                                   interpolation=cv2.INTER_AREA)
            yield t, frame
        index += 1
    cap.release()


def sample_frames(video_path, fps=1, start=None, end=None, max_frames=60, width=1280):
    """Returns a list of (timestamp, frame), evenly thinned to at most `max_frames`."""
    frames = list(iter_frames(video_path, fps=fps, start=start, end=end, width=width))
    if len(frames) > max_frames:
        keep = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in keep]
    return frames


def encode_jpeg(frame, quality=80):
    """Encodes a BGR frame as JPEG bytes."""
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Could not encode frame")
    return buffer.tobytes()


# --------------------------------------------------------------------- motion
def detect_motion_segments(video_path, analysis_fps=30, min_area=0.01, pixel_threshold=12,
                           merge_gap=0.12, start=None, end=None):
    """
    Finds time spans in which a noticeable part of the screen changes.

    Consecutive frames are compared. A frame counts as "changing" when more than
    `min_area` (fraction of all pixels) differs from the previous frame, which
    ignores cursor movement and typing. Changing frames closer than `merge_gap`
    seconds are joined into one segment.

    Each segment is labelled:
      - "instant":  the screen changed within one frame step (no animation)
      - "animated": the change was spread over several frames (a transition,
                    animation, scroll or loading effect)

    Returns:
        list of dicts with start, end, duration (seconds), peak_area (largest
        fraction of the screen changing at once), region (x, y, w, h as fractions
        of the frame) and kind.
    """
    step = None
    previous = None
    previous_t = None
    changing = []  # (t_prev, t, area, bbox)
    for t, frame in iter_frames(video_path, fps=analysis_fps, start=start, end=end, width=480):
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if previous is not None:
            if step is None:
                step = t - previous_t
            diff = cv2.absdiff(gray, previous) > pixel_threshold
            area = float(diff.mean())
            if area >= min_area:
                ys, xs = np.where(diff)
                h, w = diff.shape
                bbox = (xs.min() / w, ys.min() / h, xs.max() / w, ys.max() / h)
                changing.append((previous_t, t, area, bbox))
        previous, previous_t = gray, t

    segments = []
    for t0, t1, area, bbox in changing:
        if segments and t0 - segments[-1]["end"] <= merge_gap:
            seg = segments[-1]
            seg["end"] = t1
            seg["peak_area"] = max(seg["peak_area"], area)
            seg["steps"] += 1
            b = seg["_bbox"]
            seg["_bbox"] = (min(b[0], bbox[0]), min(b[1], bbox[1]), max(b[2], bbox[2]), max(b[3], bbox[3]))
        else:
            segments.append({"start": t0, "end": t1, "peak_area": area, "steps": 1, "_bbox": bbox})

    for seg in segments:
        x0, y0, x1, y1 = seg.pop("_bbox")
        seg["region"] = {"x": round(x0, 2), "y": round(y0, 2), "w": round(x1 - x0, 2), "h": round(y1 - y0, 2)}
        seg["kind"] = "instant" if seg.pop("steps") == 1 else "animated"
        seg["start"] = round(seg["start"], 3)
        seg["end"] = round(seg["end"], 3)
        seg["duration"] = round(seg["end"] - seg["start"], 3)
        seg["peak_area"] = round(seg["peak_area"], 3)
    return segments


def motion_summary(video_path, **kwargs):
    """Counts instant and animated screen changes and summarises animation durations."""
    segments = detect_motion_segments(video_path, **kwargs)
    animated = [s for s in segments if s["kind"] == "animated"]
    durations = [s["duration"] for s in animated]
    return {
        "screen_changes": len(segments),
        "instant": len(segments) - len(animated),
        "animated": len(animated),
        "animated_under_1s": sum(1 for d in durations if d < 1.0),
        "median_animation_duration": round(float(np.median(durations)), 3) if durations else None,
        "segments": segments,
    }


# --------------------------------------------------------------------- colour
def to_hex(bgr):
    b, g, r = (int(round(c)) for c in bgr)
    return f"#{r:02X}{g:02X}{b:02X}"


def relative_luminance(hex_colour):
    """WCAG 2.x relative luminance of an sRGB colour given as '#RRGGBB'."""
    channels = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(hex_a, hex_b):
    """WCAG 2.x contrast ratio between two colours (1.0 to 21.0)."""
    la, lb = relative_luminance(hex_a), relative_luminance(hex_b)
    return round((max(la, lb) + 0.05) / (min(la, lb) + 0.05), 2)


def dominant_palette(frames, k=6, sample=40000, seed=0):
    """
    Clusters pixel colours of one or more BGR frames with k-means.

    Returns:
        list of {"hex", "share"} sorted by share of pixels, largest first.
    """
    if isinstance(frames, np.ndarray):
        frames = [frames]
    pixels = np.concatenate([f.reshape(-1, 3) for f in frames]).astype(np.float32)
    rng = np.random.default_rng(seed)
    if len(pixels) > sample:
        pixels = pixels[rng.choice(len(pixels), sample, replace=False)]
    cv2.setRNGSeed(seed)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centres = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    shares = np.bincount(labels.ravel(), minlength=k) / len(labels)
    order = np.argsort(-shares)
    return [{"hex": to_hex(centres[i]), "share": round(float(shares[i]), 3)} for i in order if shares[i] > 0]


def colour_report(video_path, k=6, n_frames=8):
    """
    Measures the dominant colours of a recording and how strongly each contrasts
    with the main background colour.

    The contrast values compare palette colours with the most common colour. They
    indicate which colours could fail WCAG contrast (4.5:1 for normal text, 3:1 for
    large text and UI components) if used on that background; they are not a
    per-text-element audit.
    """
    duration = video_info(video_path)["duration"]
    frames = [f for _, f in sample_frames(video_path, fps=max(n_frames / max(duration, 1), 0.05),
                                          max_frames=n_frames, width=640)]
    palette = dominant_palette(frames, k=k)
    background = palette[0]["hex"]
    for colour in palette[1:]:
        ratio = contrast_ratio(colour["hex"], background)
        colour["contrast_with_background"] = ratio
        colour["wcag"] = "passes text (>=4.5)" if ratio >= 4.5 else \
            "large text / UI only (>=3)" if ratio >= 3 else "fails (<3)"
    return {"background": background, "palette": palette}


def grounding_text(video_path):
    """Formats the measurements as text that can be appended to a prompt."""
    colours = colour_report(video_path)
    motion = motion_summary(video_path)
    lines = ["Measured facts about this recording (computed from the pixels, not by a model):",
             f"- Main background colour: {colours['background']}"]
    for c in colours["palette"][1:]:
        lines.append(f"- Colour {c['hex']} covers {c['share']:.0%} of the screen, "
                     f"contrast with background {c['contrast_with_background']}:1 ({c['wcag']})")
    lines.append(f"- Screen changes: {motion['screen_changes']} "
                 f"({motion['instant']} instant, {motion['animated']} animated)")
    for s in [s for s in motion["segments"] if s["kind"] == "animated"][:15]:
        lines.append(f"- Animated change from {s['start']:.2f}s to {s['end']:.2f}s "
                     f"({s['duration'] * 1000:.0f} ms, up to {s['peak_area']:.0%} of the screen)")
    return "\n".join(lines)
