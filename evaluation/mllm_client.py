"""
Provider-independent access to multimodal LLMs.

The original notebook called the Gemini API directly from two functions. This
module keeps the same two entry points (`get_response_json`, `get_response_text`)
behind one class so the model can be switched with a single value:

    client = MLLMClient("gemini-3.8-flash")        # Google Gemini (native video)
    client = MLLMClient("gpt-5")                   # OpenAI (video sent as frames)
    client = MLLMClient("qwen2.5vl", base_url="http://localhost:11434/v1")  # local

API keys are read from environment variables or from a `.env` file in the
repository root (which is git-ignored). They are never written in code.

    GEMINI_API_KEY   - Google AI Studio key, for models whose name starts with "gemini"
    OPENAI_API_KEY   - key for any OpenAI-compatible endpoint
    OPENAI_BASE_URL  - optional, e.g. OpenRouter or a local Ollama server
    MLLM_MODEL       - optional default model name
"""

import base64
import json
import mimetypes
import os
import re
import time
from json import JSONDecodeError

DEFAULT_MODEL = "gemini-3.8-flash"
INLINE_VIDEO_LIMIT_BYTES = 15 * 1024 * 1024  # Gemini accepts about 20 MB per request in total

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env(path=None):
    """Reads KEY=VALUE lines from a .env file into os.environ (existing values win)."""
    path = path or os.path.join(REPO_ROOT, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def repair_json_quotes(json_string):
    """Escapes unescaped double quotes inside JSON string values (from the original notebook)."""
    repaired = []
    in_string = False
    i = 0
    while i < len(json_string):
        char = json_string[i]
        if char == '"':
            if i > 0 and json_string[i - 1] == "\\":
                repaired.append(char)
                i += 1
                continue
            if not in_string:
                in_string = True
                repaired.append(char)
            else:
                j = i + 1
                while j < len(json_string) and json_string[j].isspace():
                    j += 1
                if j >= len(json_string) or json_string[j] in [",", "}", "]", ":"]:
                    in_string = False
                    repaired.append(char)
                else:
                    repaired.append('\\"')
        else:
            repaired.append(char)
        i += 1
    return "".join(repaired)


def parse_json(text):
    """
    Parses a model reply as JSON. Tolerates markdown code fences, text around the
    JSON, trailing commas and unescaped quotes. Raises JSONDecodeError if nothing works.
    """
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if starts:
        start = min(starts)
        end = max(text.rfind("}"), text.rfind("]"))
        text = text[start:end + 1]
    no_trailing_commas = re.sub(r",(\s*[}\]])", r"\1", text)
    for candidate in (text, no_trailing_commas, repair_json_quotes(no_trailing_commas)):
        try:
            return json.loads(candidate)
        except JSONDecodeError:
            continue
    raise JSONDecodeError("Model reply is not valid JSON", text, 0)


class MLLMClient:
    """
    One interface for several model providers.

    Args:
        model: Model name. Names starting with "gemini" use Google's API; anything
            else is sent to an OpenAI-compatible chat endpoint.
        provider: "gemini" or "openai" to override the automatic choice.
        base_url: Endpoint for OpenAI-compatible providers (OpenRouter, Ollama, ...).
        max_frames: Upper limit of frames sent to providers without native video input.
        retries: Attempts per request when the API returns an error (e.g. rate limit).
    """

    def __init__(self, model=None, provider=None, base_url=None, max_frames=60, retries=3):
        load_env()
        self.model = model or os.environ.get("MLLM_MODEL") or DEFAULT_MODEL
        self.provider = provider or ("gemini" if self.model.lower().startswith("gemini") else "openai")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.max_frames = max_frames
        self.retries = retries
        self._client = None
        self._uploaded = {}

    # ------------------------------------------------------------------ setup
    def _key(self, name):
        key = os.environ.get(name)
        if not key or key == "YOUR_API_KEY_HERE":
            raise RuntimeError(
                f"{name} is not set. Copy .env.example to .env in the repository root "
                f"and put your key there (the .env file is git-ignored)."
            )
        return key

    def _get_client(self):
        if self._client is None:
            if self.provider == "gemini":
                from google import genai
                self._client = genai.Client(api_key=self._key("GEMINI_API_KEY"))
            else:
                from openai import OpenAI
                # Local servers such as Ollama need no real key.
                key = os.environ.get("OPENAI_API_KEY") or ("local" if self.base_url else self._key("OPENAI_API_KEY"))
                self._client = OpenAI(api_key=key, base_url=self.base_url)
        return self._client

    # ----------------------------------------------------------------- public
    def get_response_text(self, user_prompt, system_prompt=None, video_path=None,
                          fps=None, start=None, end=None):
        """
        Sends a prompt, optionally with a video, and returns the reply text.

        Args:
            user_prompt: The prompt.
            system_prompt: Optional system instruction.
            video_path: Optional path to a video file.
            fps: Frames per second the model should look at. Gemini's default is 1,
                which misses most UI animations; use 5-15 for motion analysis.
            start, end: Optional clip boundaries in seconds.
        """
        last_error = None
        for attempt in range(self.retries):
            try:
                if self.provider == "gemini":
                    return self._gemini(user_prompt, system_prompt, video_path, fps, start, end)
                return self._openai(user_prompt, system_prompt, video_path, fps, start, end)
            except (RuntimeError, ImportError, FileNotFoundError):
                raise
            except Exception as error:  # API errors: wait and retry
                last_error = error
                if attempt < self.retries - 1:
                    wait = 5 * 2 ** attempt
                    print(f"Request failed ({type(error).__name__}: {error}). Retrying in {wait}s.")
                    time.sleep(wait)
        raise last_error

    def get_response_json(self, system_prompt, user_prompt, video_path=None,
                          fps=None, start=None, end=None):
        """
        Like `get_response_text`, but parses the reply as JSON. If the reply is not
        valid JSON the raw text is returned, as in the original notebook.
        """
        text = self.get_response_text(user_prompt, system_prompt, video_path, fps, start, end)
        try:
            return parse_json(text)
        except JSONDecodeError:
            print("Error decoding JSON. Return the raw response.")
            return text

    # ----------------------------------------------------------------- gemini
    def _gemini_video_part(self, client, video_path, fps, start, end):
        part = {"type": "video", "mime_type": mimetypes.guess_type(video_path)[0] or "video/mp4"}
        if os.path.getsize(video_path) <= INLINE_VIDEO_LIMIT_BYTES:
            with open(video_path, "rb") as f:
                part["data"] = base64.b64encode(f.read()).decode("utf-8")
        else:
            if video_path not in self._uploaded:
                uploaded = client.files.upload(file=video_path)
                while not uploaded.state or uploaded.state.name != "ACTIVE":
                    time.sleep(5)
                    uploaded = client.files.get(name=uploaded.name)
                self._uploaded[video_path] = uploaded
            uploaded = self._uploaded[video_path]
            part["uri"] = uploaded.uri
            part["mime_type"] = uploaded.mime_type
        if fps is not None or start is not None or end is not None:
            processing = {"type": "static"}
            if fps is not None:
                processing["fps"] = fps
            if start is not None:
                processing["start_offset"] = f"{start:g}s"
            if end is not None:
                processing["end_offset"] = f"{end:g}s"
            part["processing"] = processing
        return part

    def _gemini(self, user_prompt, system_prompt, video_path, fps, start, end):
        client = self._get_client()
        if not hasattr(client, "interactions"):
            return self._gemini_legacy(client, user_prompt, system_prompt, video_path, fps, start, end)
        content = []
        if video_path:
            content.append(self._gemini_video_part(client, video_path, fps, start, end))
        content.append({"type": "text", "text": user_prompt})
        kwargs = {"model": self.model, "input": content}
        if system_prompt:
            kwargs["system_instruction"] = system_prompt
        # No temperature: current Gemini models ignore custom sampling parameters.
        interaction = client.interactions.create(**kwargs)
        return interaction.output_text

    def _gemini_legacy(self, client, user_prompt, system_prompt, video_path, fps, start, end):
        """generate_content path for older google-genai versions (as in the original notebook)."""
        from google.genai import types
        contents = []
        if video_path:
            with open(video_path, "rb") as f:
                blob = types.Blob(data=f.read(), mime_type=mimetypes.guess_type(video_path)[0] or "video/mp4")
            metadata = {}
            if fps is not None:
                metadata["fps"] = fps
            if start is not None:
                metadata["start_offset"] = f"{start:g}s"
            if end is not None:
                metadata["end_offset"] = f"{end:g}s"
            contents.append(types.Part(inline_data=blob,
                                       video_metadata=types.VideoMetadata(**metadata) if metadata else None))
        contents.append(user_prompt)
        config = {"temperature": 0.0}
        if system_prompt:
            config["system_instruction"] = system_prompt
        response = client.models.generate_content(model=self.model, contents=contents, config=config)
        return response.text

    # ----------------------------------------------------------------- openai
    def _openai(self, user_prompt, system_prompt, video_path, fps, start, end):
        client = self._get_client()
        content = []
        if video_path:
            from video_tools import sample_frames, encode_jpeg
            frames = sample_frames(video_path, fps=fps or 1, start=start, end=end, max_frames=self.max_frames)
            times = ", ".join(f"{t:.2f}s" for t, _ in frames)
            content.append({"type": "text",
                            "text": f"The following {len(frames)} images are frames of a screen recording, "
                                    f"in order, taken at these timestamps: {times}."})
            for _, frame in frames:
                data = base64.b64encode(encode_jpeg(frame)).decode("utf-8")
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}})
        content.append({"type": "text", "text": user_prompt})
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        response = client.chat.completions.create(model=self.model, messages=messages)
        return response.choices[0].message.content
