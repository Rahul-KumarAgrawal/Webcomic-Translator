"""
web/app.py
Flask web application for CBZ Translator.

Routes:
  GET  /                   — index: upload + batch queue
  POST /upload             — accept CBZ file + series name
  GET  /progress_stream    — SSE stream for queue progress
  GET  /review/<cbz_name>  — per-bubble review + approve/reject
  POST /approve            — approve a bubble translation
  POST /reject             — reject a bubble translation
  POST /edit               — edit + approve a bubble translation
  GET  /memory             — global + per-series memory browser
  POST /memory/edit        — inline edit a memory entry
  GET  /memory/export      — export memory as JSON download
  GET  /train              — training dashboard
  POST /train/start        — manual fine-tune trigger
  GET  /backup             — backup list
  POST /backup/now         — manual backup trigger
  POST /backup/restore     — restore from a backup
  GET  /settings           — settings form
  POST /settings/save      — save settings
  POST /settings/upload_font — upload a .ttf font file
"""

import json
import logging
import os
import queue
import shutil
import site
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Project root setup ────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── DLL Fix: Deep NVIDIA & Dependency Discovery ─────────────────────────────
# Fixes 'cudnn64_8.dll not found' (Error 126) by linking all required binaries.
def _link_dlls():
    t0 = time.time()
    cache_path = os.path.join(_ROOT, ".dll_paths.cache")
    
    # Try loading from cache first for instant startup
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached_paths = [l.strip() for l in f if l.strip()]
            linked = 0
            for p in cached_paths:
                if os.path.exists(p):
                    os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
                    if hasattr(os, "add_dll_directory"):
                        try:
                            os.add_dll_directory(p)
                            linked += 1
                        except Exception: pass
            if linked > 0:
                return
        except Exception: pass

    # Cache miss or empty: perform deep scan
    print("[DEBUG] Super-Linker: Cache miss, performing deep scan...")
    search_paths = site.getsitepackages() + sys.path
    found_dirs = []
    
    for s_path in search_paths:
        if not s_path or not os.path.exists(s_path) or not os.path.isdir(s_path): continue
        
        # 1. Link NVIDIA packages
        nv_root = os.path.join(s_path, "nvidia")
        if os.path.exists(nv_root):
            for sub in os.listdir(nv_root):
                bp = os.path.join(nv_root, sub, "bin")
                if os.path.exists(bp): found_dirs.append(bp)
        
        # 2. Link Torch/lib
        torch_lib = os.path.join(s_path, "torch", "lib")
        if os.path.exists(torch_lib): found_dirs.append(torch_lib)

        # 3. Link Paddle libs
        paddle_libs = os.path.join(s_path, "paddle", "libs")
        if os.path.exists(paddle_libs): found_dirs.append(paddle_libs)

    # Apply and Save to cache
    linked_count = 0
    unique_dirs = list(set(found_dirs))
    for p in unique_dirs:
        os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(p)
                linked_count += 1
            except Exception: pass
            
    with open(cache_path, "w") as f:
        f.write("\n".join(unique_dirs))

_link_dlls()

import threading
import yaml
from flask import (
    Flask, Response, jsonify, redirect, render_template,
    request, send_file, url_for, stream_with_context,
)
from flask_cors import CORS

# ── Project root setup ────────────────────────────────────────────────────────
import core.inpainter
print(f"\n[DEBUG] Loading Inpainter from: {core.inpainter.__file__}\n")

_session_lock = threading.Lock()

SESSIONS_DIR  = os.path.join(_ROOT, "web", "sessions")
BACKUPS_DIR   = os.path.join(_ROOT, "backups")
FONTS_DIR     = os.path.join(_ROOT, "fonts")
CONFIG_PATH   = os.path.join(_ROOT, "config", "settings.yaml")
LOGS_DIR      = os.path.join(_ROOT, "logs")

for d in [SESSIONS_DIR, BACKUPS_DIR, FONTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Flask setup ───────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(_ROOT, "web", "templates"),
    static_folder  =os.path.join(_ROOT, "web", "static"),
)
app.secret_key = "cbz-translator-secret-key-change-me"
CORS(app, origins=[
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "https://verceldeploy-gray.vercel.app",
    "https://*.vercel.app",
    "https://*.trycloudflare.com",
    "https://*.ngrok-free.dev",
], supports_credentials=True)

import secrets
from functools import wraps
from werkzeug.utils import secure_filename

# Use env var or generate a secure random key if not overridden
if app.secret_key == "cbz-translator-secret-key-change-me":
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))

def _ensure_auth_token():
    cfg = _load_cfg()
    token = cfg.get("tunnel_auth_token", "")
    if not token:
        token = secrets.token_urlsafe(32)
        cfg["tunnel_auth_token"] = token
        _save_cfg(cfg)
    return token

def _get_auth_token():
    return _load_cfg().get("tunnel_auth_token", "")

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        remote = request.remote_addr or ""
        if remote in ("127.0.0.1", "::1", "localhost"):
            return f(*args, **kwargs)
        expected = _get_auth_token()
        if expected:
            provided = request.headers.get("X-Auth-Token", "") or request.args.get("_auth", "")
            if provided != expected:
                return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

@app.context_processor
def inject_tunnel_config():
    cfg = _load_cfg()
    return {
        "tunnel_url": cfg.get("tunnel_url", ""),
        "tunnel_auth_token": cfg.get("tunnel_auth_token", ""),
    }

# Use a standard logger for the web server itself to avoid triggering heavy imports early
logger = logging.getLogger("web.app")
if not logger.handlers:
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("[web.app] %(message)s"))
    logger.addHandler(sh)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    # Also disable werkzeug propagation to prevent duplicate request logs
    logging.getLogger("werkzeug").propagate = False

def _delayed_init_logging():
    """Initializes the heavy translation logging system without blocking the web server startup."""
    try:
        from manga_translator.utils.log import init_logging
        init_logging()
        logging.getLogger("web.app").info("Heavy translation engine logging ready.")
    except Exception: pass

threading.Thread(target=_delayed_init_logging, daemon=True).start()

SUPPORTED_LANGUAGES = {
    "": "Auto-detect",
    "jpn_Jpan": "Japanese",
    "zho_Hans": "Chinese (Simplified)",
    "zho_Hant": "Chinese (Traditional)",
    "kor_Hang": "Korean",
    "eng_Latn": "English",
    "spa_Latn": "Spanish",
    "fra_Latn": "French",
    "deu_Latn": "German",
    "ita_Latn": "Italian",
    "por_Latn": "Portuguese",
    "nld_Latn": "Dutch",
    "swe_Latn": "Swedish",
    "dan_Latn": "Danish",
    "fin_Latn": "Finnish",
    "nob_Latn": "Norwegian",
    "pol_Latn": "Polish",
    "ces_Latn": "Czech",
    "slk_Latn": "Slovak",
    "ron_Latn": "Romanian",
    "hun_Latn": "Hungarian",
    "hrv_Latn": "Croatian",
    "rus_Cyrl": "Russian",
    "bul_Cyrl": "Bulgarian",
    "ukr_Cyrl": "Ukrainian",
    "bel_Cyrl": "Belarusian",
    "srp_Cyrl": "Serbian (Cyrillic)",
    "arb_Arab": "Arabic",
    "pes_Arab": "Persian",
    "urd_Arab": "Urdu",
    "uig_Arab": "Uyghur",
    "hin_Deva": "Hindi",
    "ben_Beng": "Bengali",
    "guj_Gujr": "Gujarati",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "kan_Knda": "Kannada",
    "mal_Mlym": "Malayalam",
    "mar_Deva": "Marathi",
    "nep_Deva": "Nepali",
    "tur_Latn": "Turkish",
    "vie_Latn": "Vietnamese",
    "ind_Latn": "Indonesian",
    "zsm_Latn": "Malay",
    "afr_Latn": "Afrikaans",
    "aze_Latn": "Azerbaijani",
    "bos_Latn": "Bosnian",
    "cym_Latn": "Welsh",
    "est_Latn": "Estonian",
    "gle_Latn": "Irish",
    "isl_Latn": "Icelandic",
    "kur_Latn": "Kurdish",
    "lat_Latn": "Latin",
    "lit_Latn": "Lithuanian",
    "lav_Latn": "Latvian",
    "mri_Latn": "Maori",
    "mlt_Latn": "Maltese",
    "slv_Latn": "Slovenian",
    "sqi_Latn": "Albanian",
    "swa_Latn": "Swahili",
    "tgl_Latn": "Tagalog",
    "uzb_Latn": "Uzbek",
}

# ── Config helpers ────────────────────────────────────────────────────────────

def _load_cfg() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _save_cfg(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

# ── Translation queue (SSE) ───────────────────────────────────────────────────

class TranslationQueue:
    """Thread-safe job queue with SSE event broadcasting."""

    def __init__(self):
        self._jobs: list       = []          # list of job dicts
        self._lock             = threading.Lock()
        self._listeners: list  = []          # SSE subscriber queues

    def add(self, cbz_name: str, series: str,
            source_lang: str = "", target_lang: str = "eng_Latn",
            translation_engine: str = "nllb",
            ocr_engine: str = "mit",
            detection_engine: str = "mit",
            inpaint_engine: str = "lama",
            use_mit_pipeline: bool = False,
            use_koharu_pipeline: bool = False,
            mit_translator: str = "sugoi",
            mit_target_lang: str = "ENG",
            force_retranslate: bool = False,
            chunk_height: int = 0,
            chunk_overlap: int = 0,
            ocr_super_res: bool = True,
            ocr_upscale_factor: str = "2",
            global_upscale: bool = False,
            global_upscale_impl: str = "none") -> str:
        job_id = f"{cbz_name}_{int(time.time())}"
        job = {
            "id":          job_id,
            "cbz_name":    cbz_name,
            "series":      series,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "translation_engine": translation_engine,
            "ocr_engine":  ocr_engine,
            "detection_engine": detection_engine,
            "inpaint_engine": inpaint_engine,
            "use_mit_pipeline": use_mit_pipeline,
            "use_koharu_pipeline": use_koharu_pipeline,
            "mit_translator": mit_translator,
            "mit_target_lang": mit_target_lang,
            "force_retranslate": force_retranslate,
            "chunk_height": chunk_height,
            "chunk_overlap": chunk_overlap,
            "ocr_super_res": ocr_super_res,
            "ocr_upscale_factor": ocr_upscale_factor,
            "global_upscale": global_upscale,
            "global_upscale_impl": global_upscale_impl,
            "status":      "queued",
            "progress":    0,
            "total":       0,
            "progress_text": "",
            "error":       None,
        }
        with self._lock:
            self._jobs.append(job)
        self._broadcast({"type": "queued", **job})
        return job_id

    def jobs(self):
        with self._lock:
            return list(self._jobs)

    def update(self, job_id: str, **kwargs):
        with self._lock:
            for job in self._jobs:
                if job["id"] == job_id:
                    job.update(kwargs)
                    self._broadcast({"type": "update", **job})
                    break

    def remove(self, cbz_name: str):
        with self._lock:
            self._jobs = [j for j in self._jobs if j.get("cbz_name") != cbz_name]
        self._broadcast({"type": "deleted", "cbz_name": cbz_name})

    def subscribe(self) -> queue.Queue:
        q = queue.Queue(maxsize=100)
        self._listeners.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        try:
            self._listeners.remove(q)
        except ValueError:
            pass

    def _broadcast(self, event: dict):
        dead = []
        for q in self._listeners:
            try:
                q.put_nowait(event)
            except queue.Full:
                dead.append(q)
        for q in dead:
            self.unsubscribe(q)


_tq           = TranslationQueue()
_processing   = False
_process_lock = threading.Lock()
_cancelled_ids = set()

_train_progress = {
    "status": "idle",
    "step": 0,
    "total_steps": 0,
    "loss": 0.0,
    "elapsed_s": 0.0,
    "eta_s": 0.0,
    "version": 0,
    "error": None,
}
_train_listeners = []

def _broadcast_train():
    dead = []
    evt = {"type": "update", **_train_progress}
    for q in _train_listeners:
        try:
            q.put_nowait(evt)
        except queue.Full:
            dead.append(q)
    for q in dead:
        try:
            _train_listeners.remove(q)
        except ValueError:
            pass



def _process_queue_worker():
    """Background thread: processes queued jobs one by one."""
    global _processing
    while True:
        time.sleep(1)
        pending = [j for j in _tq.jobs() if j["status"] == "queued"]
        if not pending:
            continue
        with _process_lock:
            if _processing:
                continue
            _processing = True

        job = pending[0]
        try:
            _run_job(job)
        except Exception as exc:
            logger.error("Job error: %s", exc, exc_info=True)
            _tq.update(job["id"], status="error", error=str(exc))
        finally:
            with _process_lock:
                _processing = False


threading.Thread(target=_process_queue_worker, daemon=True).start()


def _run_job(job: dict):
    # Rerender jobs are handled by their own thread (_do_rerender).
    # We must explicitly ignore them here so the queue worker doesn't process them
    # and accidentally overwrite session.json by rerunning process_cbz.
    if job.get("translation_engine") == "Rerender":
        return

    from core.batch_processor import process_cbz
    cfg = _load_cfg()

    # Inject language settings from the job into cfg so the Translator picks them up
    if job.get("source_lang"):
        cfg["source_lang_override"] = job["source_lang"]
    cfg["target_lang"] = job.get("target_lang", "eng_Latn")
    cfg["translation_engine"] = job.get("translation_engine", cfg.get("translation_engine", "nllb"))
    if "ocr_engine" in job:
        cfg["ocr_engine"] = job["ocr_engine"]
    if "detection_engine" in job:
        cfg["detection_engine"] = job["detection_engine"]
    if "inpaint_engine" in job:
        cfg["inpaint_engine"] = job["inpaint_engine"]

    # Force re-translate flag
    cfg["force_retranslate"] = job.get("force_retranslate", False)

    # Manhwa chunking
    cfg["chunk_height"] = job.get("chunk_height", 0)
    cfg["chunk_overlap"] = job.get("chunk_overlap", 0)

    # OCR Quality
    val_sr = job.get("ocr_super_res", True)
    if isinstance(val_sr, str):
        cfg["ocr_super_res"] = val_sr.lower() == "true"
    else:
        cfg["ocr_super_res"] = bool(val_sr)

    cfg["ocr_upscale_factor"] = float(job.get("ocr_upscale_factor", 2.0))

    val_gu = job.get("global_upscale", False)
    if isinstance(val_gu, str):
        cfg["global_upscale"] = val_gu.lower() == "true"
    else:
        cfg["global_upscale"] = bool(val_gu)

    cfg["global_upscale_impl"] = job.get("global_upscale_impl", "none")

    # Pipeline configs
    use_mit = job.get("use_mit_pipeline", cfg.get("use_mit_pipeline", False))
    cfg["use_mit_pipeline"] = use_mit
    
    use_koharu = job.get("use_koharu_pipeline", cfg.get("use_koharu_pipeline", False))
    cfg["use_koharu_pipeline"] = use_koharu
    
    if use_mit or use_koharu:
        # Override Pipeline translator/target_lang from job if provided
        mit_cfg = cfg.setdefault("mit", {})
        if job.get("mit_translator"):
            mit_cfg["translator"] = job["mit_translator"]
        if job.get("mit_target_lang"):
            mit_cfg["target_lang"] = job["mit_target_lang"]

    cbz_path   = os.path.join(_ROOT, "input", job["cbz_name"])
    output_dir = cfg.get("output_folder", "./output")
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(_ROOT, output_dir)

    _tq.update(job["id"], status="processing", progress=0)

    class JobCancelledError(Exception): pass

    def progress(current, total, text=None):
        # Check if job was cancelled
        if job["id"] in _cancelled_ids:
            raise JobCancelledError("Cancelled by user")
        
        upd = {"progress": current, "total": total}
        if text:
            upd["progress_text"] = text
        _tq.update(job["id"], **upd)

    try:
        result = process_cbz(
            cbz_path   = cbz_path,
            output_dir = output_dir,
            series     = job["series"],
            cfg        = cfg,
            logger     = logger,
            progress_callback=progress,
        )
    finally:
        with _tq._lock:
            _cancelled_ids.discard(job["id"])

    current_job = next((j for j in _tq.jobs() if j["id"] == job["id"]), None)
    
    if result.get("success"):
        if current_job and current_job["status"] != "cancelled":
            _tq.update(job["id"], status="done", progress=result.get("num_bubbles", 0))
    else:
        if current_job:
             _tq.update(job["id"], status="error", error=result.get("error", "Unknown"))

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/cancel_job/<job_id>", methods=["POST"])
@require_auth
def cancel_job(job_id):
    """Removes a job from the UI and signals the worker to abort."""
    logger.info("Cancelling job: %s", job_id)
    with _tq._lock:
        _cancelled_ids.add(job_id)
        _tq._jobs = [j for j in _tq._jobs if j["id"] != job_id]
    
    _tq._broadcast({"type": "init", "jobs": _tq.jobs()})
    return jsonify({"ok": True})


@app.route("/")
def index():
    cfg = _load_cfg()
    jobs = _tq.jobs()
    series_list = _get_series_list()
    return render_template("index.html", jobs=jobs, series_list=series_list, cfg=cfg, SUPPORTED_LANGUAGES=SUPPORTED_LANGUAGES)


@app.route("/upload", methods=["POST"])
def upload():
    if "cbz_file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f           = request.files["cbz_file"]
    series      = request.form.get("series", "Unknown").strip() or "Unknown"
    source_lang = request.form.get("source_lang", "").strip()   # "" = auto-detect
    target_lang = request.form.get("target_lang", "eng_Latn").strip() or "eng_Latn"
    engine      = request.form.get("translation_engine", "").strip() or _load_cfg().get("translation_engine", "nllb")
    ocr_engine  = request.form.get("ocr_engine", "").strip() or _load_cfg().get("ocr_engine", "mit")
    detection_engine = request.form.get("detection_engine", "").strip() or _load_cfg().get("detection_engine", "mit")
    inpaint_engine = request.form.get("inpaint_engine", "").strip() or _load_cfg().get("inpaint_engine", "lama")

    # Pipeline routing
    use_mit_str = request.form.get("use_mit_pipeline", "false").strip().lower()
    use_mit = use_mit_str in ("true", "1", "on", "yes")

    use_koharu_str = request.form.get("use_koharu_pipeline", "false").strip().lower()
    use_koharu = use_koharu_str in ("true", "1", "on", "yes")
    
    mit_translator  = request.form.get("mit_translator", "").strip()
    mit_target_lang = request.form.get("mit_target_lang", "").strip()

    # Fall back to global defaults from settings if not specified
    if not mit_translator:
        mit_translator = _load_cfg().get("mit", {}).get("translator", "sugoi")
    if not mit_target_lang:
        mit_target_lang = _load_cfg().get("mit", {}).get("target_lang", "ENG")

    # Force re-translate flag
    force_str = request.form.get("force_retranslate", "false").strip().lower()
    force_retranslate = force_str in ("true", "1", "on", "yes")

    # Manhwa chunking params
    chunk_height = int(request.form.get("chunk_height", "0").strip() or 0)
    chunk_overlap = int(request.form.get("chunk_overlap", "0").strip() or 0)

    # OCR Quality params
    ocr_super_res_str = request.form.get("ocr_super_res", "true").strip().lower()
    ocr_super_res = ocr_super_res_str in ("true", "1", "on", "yes")
    ocr_upscale_factor = request.form.get("ocr_upscale_factor", "2").strip()
    
    global_upscale_impl = request.form.get("global_upscale_mode", "none").strip().lower()
    global_upscale = global_upscale_impl != "none"

    if not f.filename.lower().endswith(".cbz"):
        return jsonify({"error": "Only .cbz files are accepted"}), 400

    save_path = os.path.join(_ROOT, "input", f.filename)
    
    # Retry save if file is locked (common on Windows if worker is still aborting previous job)
    saved = False
    for attempt in range(5):
        try:
            f.save(save_path)
            saved = True
            break
        except OSError as e:
            logger.warning("Upload retry %d: file locked (%s)", attempt+1, e)
            time.sleep(1.0)

    if not saved:
        return jsonify({"error": "The file is currently locked by a background process (likely a cancelling job). Please wait a few seconds and try again."}), 503

    job_id = _tq.add(
        f.filename, series, source_lang=source_lang,
        target_lang=target_lang, translation_engine=engine,
        ocr_engine=ocr_engine,
        detection_engine=detection_engine,
        inpaint_engine=inpaint_engine,
        use_mit_pipeline=use_mit,
        use_koharu_pipeline=use_koharu,
        mit_translator=mit_translator,
        mit_target_lang=mit_target_lang,
        force_retranslate=force_retranslate,
        chunk_height=chunk_height,
        chunk_overlap=chunk_overlap,
        ocr_super_res=ocr_super_res,
        ocr_upscale_factor=ocr_upscale_factor,
        global_upscale=global_upscale,
        global_upscale_impl=global_upscale_impl,
    )
    pipeline_label = "Koharu" if use_koharu else ("MIT" if use_mit else engine)
    logger.info(
        "Queued: %s (series=%s, src=%s, tgt=%s, pipeline=%s, det=%s, ocr=%s, inpaint=%s, force=%s, chunk=%d/%d, job=%s)",
        f.filename, series, source_lang or "auto", target_lang, pipeline_label, detection_engine, ocr_engine, inpaint_engine, force_retranslate, chunk_height, chunk_overlap, job_id,
    )
    return jsonify({"ok": True, "job_id": job_id, "cbz_name": f.filename})


@app.route("/download/<path:cbz_name>")
def download_output(cbz_name: str):
    """Serve a translated output CBZ for download through the tunnel."""
    cfg = _load_cfg()
    output_dir = cfg.get("output_folder", "./output")
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(_ROOT, output_dir)
    file_path = os.path.join(output_dir, cbz_name)
    if not os.path.exists(file_path):
        return jsonify({"error": f"File not found: {cbz_name}"}), 404
    return send_file(file_path, as_attachment=True, download_name=cbz_name)


@app.route("/delete_output/<cbz_name>", methods=["POST"])
def delete_output(cbz_name: str):
    """Delete the translated output CBZ + session data so the file can be re-translated."""
    cfg = _load_cfg()
    output_dir = cfg.get("output_folder", "./output")
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(_ROOT, output_dir)

    deleted = []

    # 1. Delete output CBZ
    output_cbz = os.path.join(output_dir, cbz_name)
    if os.path.exists(output_cbz):
        os.remove(output_cbz)
        deleted.append("output_cbz")

    # 2. Delete session JSON
    stem = cbz_name.replace(".cbz", "")
    session_path = os.path.join(SESSIONS_DIR, stem + ".json")
    if os.path.exists(session_path):
        os.remove(session_path)
        deleted.append("session_json")

    # 3. Delete crop images
    crops_dir = os.path.join(SESSIONS_DIR, "crops", stem)
    if os.path.exists(crops_dir):
        shutil.rmtree(crops_dir, ignore_errors=True)
        deleted.append("crops")

    # 4. Delete bg_cache
    bg_cache_dir = os.path.join(SESSIONS_DIR, "bg_cache", stem)
    if os.path.exists(bg_cache_dir):
        shutil.rmtree(bg_cache_dir, ignore_errors=True)
        deleted.append("bg_cache")

    # 5. Remove from queue
    _tq.remove(cbz_name)

    logger.info("Deleted output for %s: %s", cbz_name, deleted or "nothing found")
    return jsonify({"ok": True, "deleted": deleted, "cbz_name": cbz_name})

@app.route("/progress_stream")
def progress_stream():
    q = _tq.subscribe()

    def generate():
        # Send current state immediately
        yield f"data: {json.dumps({'type': 'init', 'jobs': _tq.jobs()})}\n\n"
        try:
            while True:
                try:
                    event = q.get(timeout=30)
                    yield f"data: {json.dumps(event)}\n\n"
                except queue.Empty:
                    yield "data: {\"type\":\"ping\"}\n\n"  # keep-alive
        except GeneratorExit:
            pass
        finally:
            _tq.unsubscribe(q)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/review/<cbz_name>")
def review(cbz_name: str):
    session_path = os.path.join(SESSIONS_DIR, cbz_name.replace(".cbz", "") + ".json")
    if not os.path.exists(session_path):
        return render_template("review.html", cbz_name=cbz_name, bubbles=[], error="No session data found. Process the file first.")
    with open(session_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = _load_cfg()
    threshold = cfg.get("memory", {}).get("confidence_threshold", 60)
    
    bubbles = data.get("bubbles", [])
    
    return render_template(
        "review.html",
        cbz_name=cbz_name,
        bubbles=bubbles,
        series=data.get("series", "Unknown"),
        threshold=threshold,
        error=None,
    )


@app.route("/approve", methods=["POST"])
@require_auth
def approve_bubble():
    data       = request.json or {}
    memory_id  = data.get("memory_id")
    series     = data.get("series", "Unknown")
    source_text      = data.get("source_text", "")
    translated_text  = data.get("translated_text", "")
    source_lang      = data.get("source_lang", "")
    cbz_name         = data.get("cbz_name", "")

    from memory.memory_manager import MemoryManager
    mm = MemoryManager()
    try:
        if memory_id:
            mm.approve(int(memory_id), series)
        # Append to approved_pairs.jsonl
        _append_approved_pair(source_lang, source_text, translated_text)
        # Persist to session JSON so edits survive page refreshes
        if cbz_name:
            _update_session_bubble(cbz_name, source_text, translated_text, 
                                   approved=True, edited=False, 
                                   bubble_index=data.get("bubble_index"))
        return jsonify({"ok": True})
    except Exception as exc:
        logger.error("Approve error: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/reject", methods=["POST"])
@require_auth
def reject_bubble():
    data      = request.json or {}
    memory_id = data.get("memory_id")
    series    = data.get("series", "Unknown")

    source_text     = data.get("source_text", "")
    translated_text = data.get("translated_text", "")
    source_lang     = data.get("source_lang", "")
    cbz_name        = data.get("cbz_name", "")

    from memory.memory_manager import MemoryManager
    mm = MemoryManager()
    try:
        if memory_id:
            mm.reject(int(memory_id), series)
        _append_rejected_pair(source_lang, source_text, translated_text)
        # Persist to session JSON
        if cbz_name:
            _update_session_bubble(cbz_name, source_text, translated_text, 
                                   approved=False, edited=False, rejected=True,
                                   bubble_index=data.get("bubble_index"))
        return jsonify({"ok": True})
    except Exception as exc:
        logger.error("Reject error: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/edit", methods=["POST"])
@require_auth
def edit_bubble():
    data        = request.json or {}
    memory_id   = data.get("memory_id")
    series      = data.get("series", "Unknown")
    new_text    = data.get("translated_text", "").strip()
    source_text = data.get("source_text", "")
    source_lang = data.get("source_lang", "")
    cbz_name    = data.get("cbz_name", "")

    from memory.memory_manager import MemoryManager
    mm = MemoryManager()
    try:
        if memory_id:
            mm.edit(int(memory_id), new_text, series)
        _append_approved_pair(source_lang, source_text, new_text)
        # Persist to session JSON
        if cbz_name:
            _update_session_bubble(cbz_name, source_text, new_text, 
                                   approved=True, edited=True, skip_inpaint=False,
                                   bubble_index=data.get("bubble_index"))
        return jsonify({"ok": True})
    except Exception as exc:
        logger.error("Edit error: %s", exc)
        return jsonify({"error": str(exc)}), 500

@app.route("/edit_ocr", methods=["POST"])
@require_auth
def edit_ocr():
    """Update the OCR source text for a bubble (e.g. to fix misread characters)."""
    data           = request.json or {}
    cbz_name       = data.get("cbz_name", "")
    bubble_index   = data.get("bubble_index")
    old_source     = data.get("old_source_text", "")
    new_source     = data.get("new_source_text", "").strip()

    if not cbz_name or not new_source:
        return jsonify({"error": "Missing cbz_name or new_source_text"}), 400

    session_path = os.path.join(SESSIONS_DIR, cbz_name.replace(".cbz", "") + ".json")
    if not os.path.exists(session_path):
        return jsonify({"error": "Session not found"}), 404

    try:
        with _session_lock:
            with open(session_path, "r", encoding="utf-8") as f:
                session_data = json.load(f)
            bubbles = session_data.get("bubbles", [])

            # Update by index (most reliable)
            if bubble_index is not None and 0 <= int(bubble_index) < len(bubbles):
                b = bubbles[int(bubble_index)]
                # If the translation text was mirroring the old OCR text (e.g. manual engine), keep it mirrored
                if b.get("translated_text", "").strip() == b.get("source_text", "").strip():
                    b["translated_text"] = new_source
                b["source_text"] = new_source
            else:
                # Fallback: match by old source text
                for b in bubbles:
                    if b.get("source_text", "").strip() == old_source.strip():
                        if b.get("translated_text", "").strip() == b.get("source_text", "").strip():
                            b["translated_text"] = new_source
                        b["source_text"] = new_source
                        break

            with open(session_path, "w", encoding="utf-8") as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
        return jsonify({"ok": True})
    except Exception as exc:
        logger.error("Edit OCR error: %s", exc)
        return jsonify({"error": str(exc)}), 500

@app.route("/ignore", methods=["POST"])
@require_auth
def ignore_bubble():
    data        = request.json or {}
    cbz_name    = data.get("cbz_name", "")
    source_text = data.get("source_text", "")
    try:
        if cbz_name:
            _update_session_bubble(cbz_name, source_text, "", 
                                   approved=False, edited=False, skip_inpaint=True,
                                   bubble_index=data.get("bubble_index"))
        return jsonify({"ok": True})
    except Exception as exc:
        logger.error("Ignore error: %s", exc)
        return jsonify({"error": str(exc)}), 500

@app.route("/undo_ignore", methods=["POST"])
@require_auth
def undo_ignore_bubble():
    data        = request.json or {}
    cbz_name    = data.get("cbz_name", "")
    source_text = data.get("source_text", "")
    translated_text = data.get("translated_text", "")
    try:
        if cbz_name:
            _update_session_bubble(cbz_name, source_text, translated_text, 
                                   approved=False, edited=False, skip_inpaint=False,
                                   bubble_index=data.get("bubble_index"))
        return jsonify({"ok": True})
    except Exception as exc:
        logger.error("Undo Ignore error: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/edit_bulk", methods=["POST"])
@require_auth
def edit_bubble_bulk():
    data_list = request.json or []
    from memory.memory_manager import MemoryManager
    mm = MemoryManager()
    
    cbz_session_updates = {}
    
    try:
        for data in data_list:
            memory_id   = data.get("memory_id")
            series      = data.get("series", "Unknown")
            new_text    = data.get("translated_text", "").strip()
            source_text = data.get("source_text", "")
            source_lang = data.get("source_lang", "")
            cbz_name    = data.get("cbz_name", "")

            if memory_id:
                mm.edit(int(memory_id), new_text, series)
            _append_approved_pair(source_lang, source_text, new_text)
            
            if cbz_name:
                if cbz_name not in cbz_session_updates:
                    cbz_session_updates[cbz_name] = []
                cbz_session_updates[cbz_name].append((source_text, new_text))
                
        # Bulk update sessions
        with _session_lock:
            for cbz_name, updates in cbz_session_updates.items():
                session_path = os.path.join(SESSIONS_DIR, cbz_name.replace(".cbz", "") + ".json")
                if os.path.exists(session_path):
                    with open(session_path, "r", encoding="utf-8") as f:
                        session = json.load(f)
                    
                    # Apply all updates
                    for src_text, new_text in updates:
                        for b in session.get("bubbles", []):
                            if b.get("source_text", "").strip() == src_text.strip():
                                b["translated_text"] = new_text
                                b["approved"] = True
                                b["edited"] = True
                                
                    with open(session_path, "w", encoding="utf-8") as f:
                        json.dump(session, f, ensure_ascii=False, indent=2)

        return jsonify({"ok": True, "count": len(data_list)})
    except Exception as exc:
        logger.error("Bulk edit error: %s", exc)
        return jsonify({"error": str(exc)}), 500








@app.route("/session_crop/<cbz_stem>/<filename>")
def session_crop(cbz_stem: str, filename: str):
    """Serve a bubble crop image saved during processing."""
    crop_path = os.path.join(_ROOT, "web", "sessions", "crops", cbz_stem, filename)
    if not os.path.exists(crop_path):
        from flask import abort
        abort(404)
    return send_file(crop_path, mimetype="image/jpeg")


@app.route("/rerender/<cbz_name>", methods=["POST"])
@require_auth
def rerender_cbz(cbz_name: str):
    """Re-render the CBZ using reviewed/edited translations from the session JSON."""
    session_path = os.path.join(SESSIONS_DIR, cbz_name.replace(".cbz", "") + ".json")
    if not os.path.exists(session_path):
        return jsonify({"error": "No session data found. Process the file first."}), 404

    # Create a new job in the SSE queue for tracking
    with open(session_path, "r", encoding="utf-8") as f:
        session_data = json.load(f)
    series = session_data.get("series", "Unknown")
    
    # Remove old jobs for this CBZ from the queue so we don't have duplicates
    with _tq._lock:
        _tq._jobs = [j for j in _tq._jobs if j["cbz_name"] != cbz_name]
    
    # We add a job with a 'Rerender' engine label so the UI can hide the Review button
    job_id = _tq.add(
        cbz_name, series, 
        translation_engine="Rerender",
    )

    def _do_rerender(jid):
        try:
            from core.inpainter import Inpainter, BubbleRegion
            from core.cbz_handler import extract_cbz, repack_cbz

            with open(session_path, "r", encoding="utf-8") as f:
                session = json.load(f)

            # Instantly update UI so the user sees a progress bar during extraction
            cm = session.get("chunk_meta")
            estimated_total = cm.get("num_chunks", 0) if cm else 0
            _tq.update(jid, status="processing", progress=0, total=estimated_total, progress_text="Extracting images...")

            cfg = _load_cfg()
            inpainter = Inpainter(cfg=cfg)
            font_cfg = {
                **(cfg.get("default_font") or {"family": None, "color": "auto", "size": "auto"}),
                **(cfg.get("series_fonts") or {}).get(session.get("series", "Unknown"), {}),
            }

            # Extract original CBZ
            cbz_path = os.path.join(_ROOT, "input", cbz_name)
            if not os.path.exists(cbz_path):
                logger.error("Re-render: original CBZ not found: %s", cbz_path)
                _tq.update(jid, status="error", error="Original CBZ not found")
                return

            import tempfile
            chunk_meta = session.get("chunk_meta")
            if chunk_meta:
                webtoon_strip_height = 0
                chunk_height = int(chunk_meta.get("chunk_height", 0))
                chunk_overlap = int(chunk_meta.get("chunk_overlap", 0))
                logger.info("Re-render: Reloaded chunk config from session (h=%d, overlap=%d)", chunk_height, chunk_overlap)
            else:
                # If no chunk_meta, we default to NO chunking (standard manga/comic behavior)
                webtoon_strip_height = 0
                chunk_height = 0
                chunk_overlap = 0
            
            tmp_dir, images = extract_cbz(
                cbz_path,
                webtoon_strip_height=webtoon_strip_height,
                chunk_height=chunk_height,
                chunk_overlap=chunk_overlap,
            )
            output_tmp = tempfile.mkdtemp(prefix="cbz_rerender_")
            
            _tq.update(jid, status="processing", progress=0, total=len(images), progress_text="Rendering pages...")

            # Group bubbles by page_num
            bubbles_by_page = {}
            for b in session.get("bubbles", []):
                pg = b.get("page_num", 0)
                bubbles_by_page.setdefault(pg, []).append(b)

            # Check if bg_cache exists
            bg_cache_dir = os.path.join(SESSIONS_DIR, "bg_cache", cbz_name.replace(".cbz", ""))
            has_cache = os.path.isdir(bg_cache_dir)
            if has_cache:
                logger.info("Re-render: found instant bg_cache for %s", cbz_name)

            all_regions = []

            if cfg.get("font_detection_engine") == "yuzumarker":
                print("\n[AI] Using Yuzumarker Font Detection for rendering.")

            for page_idx, img_path in enumerate(images):
                page_num = page_idx + 1
                page_bubbles = bubbles_by_page.get(page_num, [])

                if not page_bubbles:
                    # No bubbles on this page — copy as-is
                    out_page = os.path.join(output_tmp, Path(img_path).name)
                    shutil.copy2(img_path, out_page)
                else:
                    from PIL import Image
                    out_page = os.path.join(output_tmp, Path(img_path).stem + ".png")
                    
                    if has_cache:
                         # 🚀 INSTANT RERENDER: Load erased background and draw text
                         cached_bg_path = os.path.join(bg_cache_dir, f"page_{page_num:04d}.png")
                         if os.path.exists(cached_bg_path):
                             inpainted_image = Image.open(cached_bg_path).convert("RGB")
                             regions = []
                             for sb in page_bubbles:
                                 # Fallback gracefully if older session JSON doesn't have coordinates
                                 x = sb.get("x", 0)
                                 y = sb.get("y", 0)
                                 w = sb.get("w", 0)
                                 h = sb.get("h", 0)
                                 br = BubbleRegion(
                                     x=x, y=y, w=w, h=h,
                                     source_text=sb.get("source_text", ""),
                                     translated_text=sb.get("translated_text", "") if not sb.get("skip_inpaint") else "",
                                     font_cfg=font_cfg
                                 )

                                 if sb.get("skip_inpaint"):
                                     if w > 0 and h > 0:
                                         orig_img = Image.open(img_path).convert("RGB")
                                         _restore_ignored_bubble(inpainted_image, orig_img, sb)
                                 else:
                                     regions.append(br)

                                 all_regions.append((None, br, page_num, ""))
                             
                             final = inpainter.render_text(inpainted_image, regions)
                             final.save(out_page, format="PNG", optimize=False)
                         else:
                             logger.warning(f"Cache miss for page {page_num:04d}, copying original")
                             shutil.copy2(img_path, out_page)
                    else:
                        # 🐢 FALLBACK: Old slow behavior (re-run OCR and Masking)
                        image, detected_regions = inpainter.detect_and_ocr(img_path)

                        filtered_regions = []
                        for region in detected_regions:
                            skip = False
                            matched_sb = None

                            # First try exact text match
                            for sb in page_bubbles:
                                if region.source_text.strip() == sb.get("source_text", "").strip():
                                    matched_sb = sb
                                    break
                            
                            if not matched_sb:
                                # Fallback to spatial matching (IoU) if text doesn't match exactly
                                # (e.g. because user edited the OCR text manually or OCR jitter)
                                best_iou = 0.0
                                rx, ry, rw, rh = region.x, region.y, region.w, region.h
                                for sb in page_bubbles:
                                    sx, sy, sw, sh = sb.get("x", 0), sb.get("y", 0), sb.get("w", 0), sb.get("h", 0)
                                    inter_x = max(rx, sx)
                                    inter_y = max(ry, sy)
                                    inter_w = max(0, min(rx + rw, sx + sw) - inter_x)
                                    inter_h = max(0, min(ry + rh, sy + sh) - inter_y)
                                    
                                    if inter_w > 0 and inter_h > 0:
                                        inter_area = inter_w * inter_h
                                        union_area = rw * rh + sw * sh - inter_area
                                        iou = inter_area / union_area if union_area > 0 else 0
                                        if iou > best_iou:
                                            best_iou = iou
                                            matched_sb = sb
                                            
                                if best_iou < 0.3:
                                    matched_sb = None

                            if matched_sb:
                                region.translated_text = matched_sb.get("translated_text", "")
                                region.font_cfg = font_cfg
                                if matched_sb.get("skip_inpaint"):
                                    skip = True
                            else:
                                region.font_cfg = font_cfg
                                
                            all_regions.append((None, region, page_num, ""))
                            if not skip:
                                filtered_regions.append(region)

                        inpainted_img = inpainter.inpaint(image, filtered_regions)
                        final_img = inpainter.render_text(inpainted_img, filtered_regions)
                        final_img.save(out_page, format="PNG", optimize=False)
                
                # Update progress
                _tq.update(jid, progress=page_num)

            # Repack into output CBZ
            output_dir = cfg.get("output_folder", "./output")
            if not os.path.isabs(output_dir):
                output_dir = os.path.join(_ROOT, output_dir)
            os.makedirs(output_dir, exist_ok=True)
            output_cbz = os.path.join(output_dir, cbz_name)

            # Reassemble overlapping chunks if chunking was used
            chunk_meta_path = os.path.join(tmp_dir, "chunk_metadata.json") if tmp_dir else None
            if chunk_meta_path and os.path.exists(chunk_meta_path):
                with open(chunk_meta_path, "r", encoding="utf-8") as _f:
                    chunk_meta = json.load(_f)

                from core.batch_processor import _reassemble_chunks
                logger.info("Re-render: Trimming overlaps for %d chunks...", chunk_meta.get("num_chunks", 0))
                _tq.update(jid, progress_text="Trimming overlaps...")
                _reassemble_chunks(output_tmp, chunk_meta, logger, all_regions)

            # Remove old output if exists
            if os.path.exists(output_cbz):
                os.remove(output_cbz)
            _tq.update(jid, progress_text="Repacking CBZ...")
            repack_cbz(output_tmp, output_cbz)

            # Cleanup
            shutil.rmtree(tmp_dir, ignore_errors=True)
            shutil.rmtree(output_tmp, ignore_errors=True)

            # (Removed Cleanup Review Data block so users can re-review after rerendering)

            _tq.update(jid, status="done")
            logger.info("Re-render complete: %s", output_cbz)
        except Exception as exc:
            _tq.update(jid, status="error", error=str(exc))
            logger.error("Re-render error: %s", exc, exc_info=True)

    threading.Thread(target=_do_rerender, args=(job_id,), daemon=True).start()
    return jsonify({"ok": True, "message": "Re-render started in background."})


@app.route("/settings")
def settings_page():
    cfg   = _load_cfg()
    fonts = [f.name for f in Path(FONTS_DIR).iterdir() if f.suffix.lower() == ".ttf"]
    series_list = _get_series_list()
    return render_template("settings.html", cfg=cfg, fonts=fonts, series_list=series_list, SUPPORTED_LANGUAGES=SUPPORTED_LANGUAGES)


@app.route("/settings/data")
def settings_data():
    """Return the current config as JSON so the Vercel UI shell can populate the settings form."""
    cfg = _load_cfg()
    fonts = [f.name for f in Path(FONTS_DIR).iterdir() if f.suffix.lower() == ".ttf"]
    return jsonify({"cfg": cfg, "fonts": fonts})


@app.route("/settings/save", methods=["POST"])
@require_auth
def settings_save():
    cfg = _load_cfg()
    # Update output folder
    cfg["output_folder"] = request.form.get("output_folder", "./output").strip()
    cfg["gpu_device"]    = request.form.get("gpu_device", "cuda").strip()
    cfg["ocr_engine"]    = request.form.get("ocr_engine", "mit").strip()
    cfg["use_sam_masks"] = request.form.get("use_sam_masks") == "on"
    cfg["source_lang"]   = request.form.get("source_lang", "").strip()
    cfg["target_lang"]   = request.form.get("target_lang", "eng_Latn").strip()
    cfg["detection_engine"] = request.form.get("detection_engine", "mit").strip()
    cfg["inpaint_engine"] = request.form.get("inpaint_engine", "lama").strip()
    cfg["auto_detect_engine"] = request.form.get("auto_detect_engine", "gemini").strip()
    
    try:
        cfg["webtoon_strip_height"] = int(request.form.get("webtoon_strip_height", 0))
    except ValueError:
        pass
    cfg["font_detection_engine"] = request.form.get("font_detection_engine", "default").strip()
    
    # Default font
    cfg.setdefault("default_font", {})
    cfg["default_font"]["family"] = request.form.get("font_family", "").strip() or None
    cfg["default_font"]["color"]  = request.form.get("font_color", "auto").strip()
    cfg["default_font"]["size"]   = request.form.get("font_size", "auto").strip() or "auto"

    # Advanced Detection
    cfg["detection_confidence"] = float(request.form.get("detection_confidence", 0.20))
    cfg["sfx_strictness"] = float(request.form.get("sfx_strictness", 0.55))
    cfg["enable_gap_filling"] = request.form.get("enable_gap_filling") == "on"
    cfg["enable_nuisance_filter"] = request.form.get("enable_nuisance_filter") == "on"
    
    # OCR & Upscaling
    cfg["ocr_super_res"] = request.form.get("ocr_super_res") == "on"
    try:
        cfg["ocr_upscale_factor"] = float(request.form.get("ocr_upscale_factor", 2.0))
    except ValueError:
        cfg["ocr_upscale_factor"] = 2.0
    
    global_upscale_impl = request.form.get("global_upscale_impl", "none").strip()
    cfg["global_upscale_impl"] = global_upscale_impl
    cfg["global_upscale"] = (global_upscale_impl != "none")
    
    # Manhwa Chunking Defaults
    try:
        cfg["chunk_height"] = int(request.form.get("chunk_height", 2500))
        cfg["chunk_overlap"] = int(request.form.get("chunk_overlap", 250))
    except ValueError:
        pass
    # Translation engine
    cfg["translation_engine"] = request.form.get("translation_engine", "nllb").strip() or "nllb"
    # Tunnel URL (configurable from settings page)
    tunnel_url = request.form.get("tunnel_url", "").strip()
    if tunnel_url:
        cfg["tunnel_url"] = tunnel_url

    # API keys — skip redacted placeholders (••••) sent by the frontend
    def _save_key(form_name, cfg_key):
        val = request.form.get(form_name, "").strip()
        if val and not val.startswith("••••"):
            cfg[cfg_key] = val

    _save_key("deepl_api_key", "deepl_api_key")
    _save_key("google_api_key", "google_api_key")
    _save_key("groq_api_key", "groq_api_key")
    
    # Save custom Gemini prompt
    cfg["google_system_prompt"] = request.form.get("google_system_prompt", "").strip()
    
    _save_key("baidu_app_id", "baidu_app_id")
    _save_key("baidu_secret_key", "baidu_secret_key")
    _save_key("sarvam_api_key", "sarvam_api_key")
    # Ollama settings (always save — they have defaults)
    ollama_model = request.form.get("ollama_model", "").strip()
    if ollama_model:
        cfg["ollama_model"] = ollama_model
    ollama_url = request.form.get("ollama_url", "").strip()
    if ollama_url:
        cfg["ollama_url"] = ollama_url

    # ── MIT Full Pipeline settings ─────────────────────────────────────────
    mit_toggle = request.form.get("use_mit_pipeline", "off").strip().lower()
    cfg["use_mit_pipeline"] = mit_toggle in ("on", "true", "1", "yes")
    koharu_toggle = request.form.get("use_koharu_pipeline", "off").strip().lower()
    cfg["use_koharu_pipeline"] = koharu_toggle in ("on", "true", "1", "yes")
    mit = cfg.setdefault("mit", {})
    mit["translator"]  = request.form.get("mit_translator", "sugoi").strip() or "sugoi"
    mit["target_lang"] = request.form.get("mit_target_lang", "ENG").strip() or "ENG"
    mit["inpainter"]   = request.form.get("mit_inpainter", "default").strip() or "default"
    mit["detector"]    = request.form.get("mit_detector", "default").strip() or "default"
    mit["ocr"]         = request.form.get("mit_ocr", "48px").strip() or "48px"
    mit["renderer"]    = request.form.get("mit_renderer", "default").strip() or "default"

    # MIT API keys (only overwrite if non-empty)
    mit_keys = cfg.setdefault("mit_api_keys", {})
    for key_name in ("openai_api_key", "gemini_api_key", "groq_api_key", "deepl_api_key_mit"):
        val = request.form.get(key_name, "").strip()
        if val:
            mit_keys[key_name] = val

    _save_cfg(cfg)
    return redirect(url_for("settings_page"))


@app.route("/settings/upload_font", methods=["POST"])
@require_auth
def upload_font():
    if "font_file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["font_file"]
    if not f.filename.lower().endswith(".ttf"):
        return jsonify({"error": "Only .ttf files accepted"}), 400
    save_path = os.path.join(FONTS_DIR, f.filename)
    f.save(save_path)
    return jsonify({"ok": True, "path": f"./fonts/{f.filename}"})

# ── Internal helpers ──────────────────────────────────────────────────────────

def _restore_ignored_bubble(target_img, original_img, bubble: dict):
    """Restore ignored bubble pixels from the original page without a hard crop seam."""
    from PIL import Image, ImageDraw, ImageFilter

    if original_img.size != target_img.size:
        original_img = original_img.resize(target_img.size, Image.LANCZOS)

    width, height = target_img.size
    x = int(bubble.get("x", 0))
    y = int(bubble.get("y", 0))
    w = int(bubble.get("w", 0))
    h = int(bubble.get("h", 0))
    if w <= 0 or h <= 0:
        return

    mask = Image.new("L", target_img.size, 0)
    draw = ImageDraw.Draw(mask)
    mask_pts = bubble.get("mask_pts")

    if mask_pts and len(mask_pts) > 2:
        pts = [(int(px), int(py)) for px, py in mask_pts]
        draw.polygon(pts, fill=255)
        mask = mask.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.GaussianBlur(1))
    else:
        pad = 12
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(width, x + w + pad)
        y2 = min(height, y + h + pad)
        draw.rectangle((x1, y1, x2, y2), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(2))

    target_img.paste(original_img, (0, 0), mask)


def _update_session_bubble(cbz_name: str, source_text: str, translated_text: str,
                           approved: bool, edited: bool, skip_inpaint: bool = False,
                           rejected: bool = False, bubble_index: int = None):
    """Update a bubble in the session JSON so edits persist across page refreshes."""
    session_path = os.path.join(SESSIONS_DIR, cbz_name.replace(".cbz", "") + ".json")
    if not os.path.exists(session_path):
        return
    try:
        with _session_lock:
            with open(session_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            bubbles = data.get("bubbles", [])
            
            # 1. Try updating by index (most reliable)
            if bubble_index is not None and 0 <= bubble_index < len(bubbles):
                b = bubbles[bubble_index]
                b["translated_text"] = translated_text
                b["approved"] = approved
                b["edited"] = edited
                b["skip_inpaint"] = skip_inpaint
                b["rejected"] = rejected
            else:
                # 2. Fallback to text matching (old way)
                for b in bubbles:
                    if b.get("source_text", "").strip() == source_text.strip():
                        b["translated_text"] = translated_text
                        b["approved"] = approved
                        b["edited"] = edited
                        b["skip_inpaint"] = skip_inpaint
                        b["rejected"] = rejected

            with open(session_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.error("Session update error: %s", exc)


def _get_series_list():
    from memory.memory_manager import MemoryManager
    mm = MemoryManager()
    return mm.list_series()


def _append_approved_pair(source_lang: str, source_text: str, translated_text: str):
    path = os.path.join(_ROOT, "model", "training_data", "approved_pairs.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "source_lang": source_lang,
            "source_text": source_text,
            "translated_text": translated_text,
            "timestamp": datetime.now().isoformat(),
        }, ensure_ascii=False) + "\n")


def _append_rejected_pair(source_lang: str, source_text: str, translated_text: str):
    path = os.path.join(_ROOT, "model", "training_data", "rejected_pairs.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "source_lang": source_lang,
            "source_text": source_text,
            "translated_text": translated_text,
            "timestamp": datetime.now().isoformat(),
        }, ensure_ascii=False) + "\n")





# ── Entry point ───────────────────────────────────────────────────────────────

# Ensure auth token is generated on startup
_ensure_auth_token()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
