import logging
import socket
import warnings
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError

warnings.simplefilter("ignore", FutureWarning)

import google.generativeai as genai
from deep_translator import GoogleTranslator
from flask import Flask, jsonify, render_template, request



API_KEY = os.environ.get("GEMINI_API_KEY")

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voca")

PRIMARY_MODEL_NAME = "gemini-3.1-flash"
FALLBACK_MODEL_NAMES = [
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-flash-latest",
]

REQUEST_TIMEOUT_SECONDS = 45
TRANSLATION_TIMEOUT_SECONDS = 20
executor = ThreadPoolExecutor(max_workers=8)

LANGUAGES = {
    "amharic": {"label": "Amharic", "translator": "am", "flag": "🇪🇹", "speech": "am-ET"},
    "oromo": {"label": "Oromo", "translator": "om", "flag": "🇪🇹", "speech": "om-ET"},
    "somali": {"label": "Somali", "translator": "so", "flag": "🇸🇴", "speech": "so-SO"},
    "swahili": {"label": "Swahili", "translator": "sw", "flag": "🇰🇪", "speech": "sw-KE"},
    "bengali": {"label": "Bengali", "translator": "bn", "flag": "🇧🇩", "speech": "bn-BD"},
    "urdu": {"label": "Urdu", "translator": "ur", "flag": "🇵🇰", "speech": "ur-PK"},
    "tamil": {"label": "Tamil", "translator": "ta", "flag": "🇮🇳", "speech": "ta-IN"},
    "burmese": {"label": "Burmese", "translator": "my", "flag": "🇲🇲", "speech": "my-MM"},
    "khmer": {"label": "Khmer", "translator": "km", "flag": "🇰🇭", "speech": "km-KH"},
    "kazakh": {"label": "Kazakh", "translator": "kk", "flag": "🇰🇿", "speech": "kk-KZ"},
}


class VocaError(Exception):
    def __init__(self, message, status_code=500, stage="Application", detail=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.stage = stage
        self.detail = detail


def clean_error_detail(error):
    detail = str(error) or error.__class__.__name__
    if API_KEY:
        detail = detail.replace(API_KEY, "[hidden]")
    return detail[:900]


def run_with_timeout(label, fn, timeout, stage):
    future = executor.submit(fn)
    try:
        return future.result(timeout=timeout)
    except TimeoutError as exc:
        future.cancel()
        raise VocaError(
            f"{label} took too long. Please try again.",
            status_code=504,
            stage=stage,
            detail=f"Timed out after {timeout} seconds.",
        ) from exc
    except VocaError:
        raise
    except Exception as exc:
        detail = clean_error_detail(exc)
        logger.warning("%s failed: %s", label, detail)
        message = f"{label} failed. Please try again."
        if "translate.google.com" in detail:
            message = "Voca could not reach the free Google Translate endpoint."
        raise VocaError(
            message,
            status_code=502,
            stage=stage,
            detail=detail,
        ) from exc


def language_config(language_key):
    config = LANGUAGES.get((language_key or "").lower())
    if not config:
        raise VocaError("Unsupported language selected.", status_code=400, stage="Language")
    return config


def translate_text(text, source, target):
    if not text.strip():
        return ""

    def translate():
        return GoogleTranslator(source=source, target=target).translate(text)

    translated = run_with_timeout(
        "Translation",
        translate,
        TRANSLATION_TIMEOUT_SECONDS,
        stage=f"Translation: {source} to {target}",
    )

    if not translated:
        raise VocaError(
            "Translation returned no text.",
            status_code=502,
            stage=f"Translation: {source} to {target}",
            detail="The free GoogleTranslator endpoint returned an empty response.",
        )

    return translated


def configure_gemini():
    if not API_KEY or API_KEY == "YOUR_KEY_HERE":
        raise VocaError(
            "Gemini API key is missing.",
            status_code=500,
            stage="Gemini",
            detail="Set API_KEY at the top of app.py.",
        )
    genai.configure(api_key=API_KEY)


def make_model(model_name):
    configure_gemini()
    return genai.GenerativeModel(
        model_name,
        generation_config={
            "temperature": 0.7,
            "top_p": 0.95,
            "max_output_tokens": 1024,
        },
    )


def ask_gemini(english_prompt):
    prompt = (
        "You are Voca, a helpful AI bridge for global communication. "
        "Answer clearly, warmly, and directly in English. "
        "Keep the response in English because it will be translated back to the user's chosen language.\n\n"
        f"User message:\n{english_prompt}"
    )

    model_names = [PRIMARY_MODEL_NAME, *FALLBACK_MODEL_NAMES]
    errors = []

    for model_name in model_names:
        model = make_model(model_name)

        def generate():
            return model.generate_content(
                prompt,
                request_options={"timeout": REQUEST_TIMEOUT_SECONDS},
            )

        try:
            response = run_with_timeout(
                f"Gemini model {model_name}",
                generate,
                REQUEST_TIMEOUT_SECONDS + 5,
                stage=f"Gemini: {model_name}",
            )
        except VocaError as exc:
            detail = exc.detail or exc.message
            errors.append(f"{model_name}: {detail}")
            can_try_fallback = (
                model_name != model_names[-1]
                and detail
                and (
                    "not found" in detail.lower()
                    or "not supported" in detail.lower()
                    or "404" in detail
                )
            )
            if can_try_fallback:
                continue
            raise

        gemini_text = extract_gemini_text(response)
        if gemini_text:
            return gemini_text, model_name

        errors.append(f"{model_name}: empty response")

    raise VocaError(
        "No Gemini Flash model returned a usable response.",
        status_code=502,
        stage="Gemini",
        detail=" | ".join(errors),
    )


def extract_gemini_text(response):
    try:
        if getattr(response, "text", None):
            return response.text.strip()
    except Exception:
        pass

    chunks = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", "")
            if text:
                chunks.append(text)

    return "\n".join(chunks).strip()


@app.get("/")
def index():
    return render_template("index.html", languages=LANGUAGES)


@app.get("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "primary_model": PRIMARY_MODEL_NAME,
            "fallback_models": FALLBACK_MODEL_NAMES,
            "languages": list(LANGUAGES.keys()),
        }
    )


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    user_input = (payload.get("message") or "").strip()
    language_key = (payload.get("language") or "amharic").strip()

    if not user_input:
        return jsonify({"error": "Please speak or type something first."}), 400

    if len(user_input) > 4000:
        return jsonify({"error": "Please keep input under 4,000 characters."}), 413

    try:
        config = language_config(language_key)
        language_code = config["translator"]
        english_prompt = translate_text(user_input, source=language_code, target="en")
        english_response, model_used = ask_gemini(english_prompt)
        translated_response = translate_text(
            english_response,
            source="en",
            target=language_code,
        )

        return jsonify(
            {
                "language": config["label"],
                "flag": config["flag"],
                "speech": config["speech"],
                "user_input": user_input,
                "english_prompt": english_prompt,
                "english_response": english_response,
                "translated_response": translated_response,
                "model_used": model_used,
            }
        )
    except VocaError as exc:
        return (
            jsonify(
                {
                    "error": exc.message,
                    "stage": exc.stage,
                    "detail": exc.detail,
                }
            ),
            exc.status_code,
        )
    except Exception as exc:
        logger.exception("Unexpected application error")
        return (
            jsonify(
                {
                    "error": "Something unexpected happened. Please try again.",
                    "stage": "Application",
                    "detail": clean_error_detail(exc),
                }
            ),
            500,
        )


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Route not found."}), 404


def pick_port(preferred_port=5000):
    for port in range(preferred_port, preferred_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred_port


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=pick_port(), debug=True, threaded=True)
