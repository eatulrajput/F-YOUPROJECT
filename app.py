import os
import re
import time
import sqlite3
import html
import threading
import requests
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib import robotparser
from flask import (
    Flask, request, jsonify, render_template, g,
    redirect, url_for, session
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- FLASK CONFIG --------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

# -------------------- MODEL & API CONFIG --------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")  # optional LLM key
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/llama-4-scout-17b-16e-instruct")

# -------------------- DATABASE --------------------
DATABASE = "chat_history.db"

# -------------------- LOCAL PYQ LINKS FILE --------------------
PYQ_LINKS_FILE = "PYQ_LINKS.txt"  # keep this file in same folder as app.py

# -------------------- UPLOAD CONFIG --------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------- SCRAPER CONFIG --------------------
START_ROOT = "kiit.ac.in"
MAX_PAGES_DEFAULT = int(os.environ.get("KIIT_SCRAPE_MAX", 150))
DELAY_DEFAULT = float(os.environ.get("KIIT_SCRAPE_DELAY", 0.8))
USER_AGENT = "SuperGPT-Scraper/1.3 (local-dev; polite crawler)"

# Debug flag for PYQ matching (set env DEBUG_PYQ_MATCH=0 to silence)
DEBUG_PYQ_MATCH = os.environ.get("DEBUG_PYQ_MATCH", "1").strip() != "0"

# -------------------- DB HELPERS --------------------
def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE, check_same_thread=False)
        db.row_factory = sqlite3.Row
    return db

def column_exists(c, table, colname):
    c.execute(f"PRAGMA table_info({table})")
    return any(r["name"] == colname for r in c.fetchall())

def init_db():
    """Create tables if missing and migrate schemas if needed."""
    with app.app_context():
        db = get_db()
        c = db.cursor()

        # users table
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT,
                password_hash TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        db.commit()

        # chat history
        c.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_message TEXT,
                bot_reply TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # scraped pages
        c.execute("""
            CREATE TABLE IF NOT EXISTS scraped_pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                title TEXT,
                content TEXT,
                fetched_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # uploaded pdfs (persist PDF-extracted text so restarts keep them)
        c.execute("""
            CREATE TABLE IF NOT EXISTS uploaded_pdfs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                text TEXT,
                uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        db.commit()

        # seed a default user if none
        c.execute("SELECT COUNT(*) AS n FROM users")
        if c.fetchone()["n"] == 0:
            c.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                ("kiitian", "kiitian@supergpt.local", generate_password_hash("supergpt123"))
            )
            db.commit()

@app.teardown_appcontext
def close_connection(_exc):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

# -------------------- LOGIN REQUIRED DECORATOR --------------------
def login_required(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper

# -------------------- UTILS --------------------
def clean_text(s: str) -> str:
    if not s:
        return ""
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def visible_text(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "svg", "meta", "nav"]):
        tag.decompose()
    return clean_text(soup.get_text(separator=" "))

def normalize_url(base: str, link: str) -> str:
    joined = urljoin(base, link)
    p = urlparse(joined)
    return f"{p.scheme}://{p.netloc}{p.path}"

def upsert_page(db_conn, url: str, title: str, content: str):
    c = db_conn.cursor()
    c.execute("""
        INSERT INTO scraped_pages (url, title, content, fetched_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(url) DO UPDATE SET
            title=excluded.title,
            content=excluded.content,
            fetched_at=CURRENT_TIMESTAMP
    """, (url, title, content))
    db_conn.commit()

def get_latest_uploaded_pdf_text():
    """Return the most recently saved PDF text from DB (or empty string)."""
    try:
        db = get_db()
        c = db.cursor()
        c.execute("SELECT text FROM uploaded_pdfs ORDER BY id DESC LIMIT 1")
        row = c.fetchone()
        return (row["text"] or "") if row else ""
    except Exception:
        return ""

# -------------------- PYQ LINKS (FILE-BASED) --------------------
def load_pyq_links():
    """Load subject->link pairs from PYQ_LINKS.txt (simple JSON-like mapping)."""
    links = {}
    if not os.path.exists(PYQ_LINKS_FILE):
        return links
    try:
        with open(PYQ_LINKS_FILE, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return links
    # regex captures "key": "value" pairs anywhere in text
    pairs = re.findall(r'["\']([^"\']+)["\']\s*:\s*["\']([^"\']+)["\']', text)
    for k, v in pairs:
        # normalize whitespace inside key
        key = re.sub(r'\s+', ' ', k).strip()
        links[key.lower()] = v.strip()
    return links

def find_subject_in_message(msg):
    if not msg:
        return None, None

    text = msg.lower()
    norm_text = re.sub(r'[^a-z0-9 ]', ' ', text)
    links = load_pyq_links()
    if not links:
        return None, None

    # ---------- PASS 1: ACRONYM MATCH (STRONGEST) ----------
    for subj, url in links.items():
        m = re.search(r'\(([a-z0-9 ]+)\)', subj.lower())
        if m:
            acronym = m.group(1).strip()
            if re.search(rf'\b{re.escape(acronym)}\b', norm_text):
                return subj, url

    # ---------- PASS 2: FULL PHRASE MATCH ----------
    for subj, url in links.items():
        key = re.sub(r'\s+', ' ', subj.lower()).strip()
        if key in norm_text:
            return subj, url

    # ---------- PASS 3: STRICT TOKEN MATCH (>= 2 UNIQUE TOKENS) ----------
    stop_words = {"and", "of", "the", "in", "for", "with", "to", "on", "by", "system", "systems"}
    for subj, url in links.items():
        tokens = [
            t for t in re.sub(r'[^a-z0-9 ]', ' ', subj.lower()).split()
            if len(t) >= 4 and t not in stop_words
        ]

        if len(tokens) < 2:
            continue

        matches = sum(1 for t in tokens if re.search(rf'\b{re.escape(t)}\b', norm_text))

        if matches >= 2:
            return subj, url

    return None, None

# -------------------- ROUTES: SEPARATED UI --------------------
@app.route("/")
def root():
    return redirect(url_for("chat_page"))

@app.route("/chat", methods=["GET"])
@login_required
def chat_page():
    return render_template("chat.html")

@app.route("/scraper", methods=["GET"])
@login_required
def scraper_page():
    return render_template("scraper.html")

# ---------- AUTH ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username_or_email = (request.form.get("username") or "").strip()
        password = (request.form.get("password") or "")
        db = get_db()
        c = db.cursor()
        c.execute("SELECT * FROM users WHERE username = ? OR email = ?", (username_or_email, username_or_email))
        row = c.fetchone()
        if row and check_password_hash(row["password_hash"], password):
            session["user_id"] = row["id"]
            session["username"] = row["username"]
            return redirect(url_for("chat_page"))
        return render_template("login.html", error="Invalid username or password.")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        password = (request.form.get("password") or "")
        confirm = (request.form.get("confirm") or "")

        if not username or not email or not password:
            return render_template("register.html", error="All fields are required.")
        if password != confirm:
            return render_template("register.html", error="Passwords do not match.")

        db = get_db()
        c = db.cursor()
        c.execute("SELECT 1 FROM users WHERE username = ? OR email = ?", (username, email))
        if c.fetchone():
            return render_template("register.html", error="Username or email already exists.")

        c.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, generate_password_hash(password))
        )
        db.commit()

        c.execute("SELECT * FROM users WHERE username=?", (username,))
        user = c.fetchone()
        session["user_id"] = user["id"]
        session["username"] = user["username"]
        return redirect(url_for("chat_page"))
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# -------------------- PDF UPLOAD --------------------
@app.route("/upload", methods=["POST"])
@login_required
def upload_pdf():
    """Save uploaded PDF file to uploads/ and persist extracted text into DB so it survives restart."""
    if "file" not in request.files:
        return jsonify({"message": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "" or not file.filename.lower().endswith(".pdf"):
        return jsonify({"message": "Invalid file type"}), 400
    try:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.stream.seek(0)
        file.save(file_path)

        # extract text with PyMuPDF
        doc = fitz.open(file_path)
        text = "\n".join([page.get_text("text") for page in doc])
        if not text.strip():
            text = "No text found in PDF."

        db = get_db()
        c = db.cursor()
        c.execute("INSERT INTO uploaded_pdfs (filename, text) VALUES (?, ?)", (filename, text))
        db.commit()

        return jsonify({"message": "PDF uploaded and persisted successfully"})
    except Exception as e:
        return jsonify({"message": f"Error processing PDF: {str(e)}"}), 500

# -------------------- SCRAPER --------------------
_scrape_state = {
    "running": False, "pages_saved": 0,
    "started_at": None, "finished_at": None,
    "last_url": None, "error": None
}

def can_fetch_url(url: str, rp: robotparser.RobotFileParser) -> bool:
    try:
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        return True

def background_scrape(start_url: str, max_pages: int, delay: float):
    global _scrape_state
    _scrape_state.update({
        "running": True, "pages_saved": 0,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": None, "error": None
    })
    try:
        db_conn = sqlite3.connect(DATABASE, check_same_thread=False)
        db_conn.row_factory = sqlite3.Row

        rp = robotparser.RobotFileParser()
        try:
            rp.set_url(urljoin(start_url, "/robots.txt"))
            rp.read()
        except Exception:
            pass

        headers = {"User-Agent": USER_AGENT}
        visited = set()
        queue = [start_url]
        root = START_ROOT

        while queue and len(visited) < max_pages and _scrape_state["running"]:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)
            _scrape_state["last_url"] = url

            if not urlparse(url).netloc.endswith(root):
                continue
            if not can_fetch_url(url, rp):
                time.sleep(delay); continue

            try:
                resp = requests.get(url, headers=headers, timeout=15)
                if resp.status_code != 200:
                    time.sleep(delay); continue

                ctype = (resp.headers.get("content-type") or "").lower()
                content, title, soup = "", url, None

                if "pdf" in ctype or url.lower().endswith(".pdf"):
                    try:
                        doc = fitz.open(stream=resp.content, filetype="pdf")
                        content = "\n".join([p.get_text("text") for p in doc])
                        title = url.split("/")[-1] or url
                    except Exception:
                        time.sleep(delay); continue
                elif "text" in ctype or "html" in ctype:
                    html_text = resp.text
                    soup = BeautifulSoup(html_text, "html.parser")
                    title = soup.title.string.strip() if soup.title and soup.title.string else url
                    content = visible_text(html_text)
                else:
                    time.sleep(delay); continue

                if content.strip():
                    upsert_page(db_conn, url, title, content)
                    _scrape_state["pages_saved"] += 1

                if soup is not None:
                    for a in soup.find_all("a", href=True):
                        href = a.get("href")
                        if href.startswith(("mailto:", "tel:", "javascript:")):
                            continue
                        normalized = normalize_url(url, href)
                        if urlparse(normalized).netloc.endswith(root) and normalized not in visited:
                            queue.append(normalized)

                time.sleep(delay)
            except Exception:
                time.sleep(delay)
                continue
    except Exception as e:
        _scrape_state["error"] = str(e)
    finally:
        _scrape_state["running"] = False
        _scrape_state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

@app.route("/scrape/start", methods=["POST"])
@login_required
def scrape_start():
    if _scrape_state["running"]:
        return jsonify({"message": "Scraper already running", "status": _scrape_state})
    params = request.get_json(silent=True) or {}
    start_url = params.get("start_url", f"https://{START_ROOT}")
    max_pages = int(params.get("max_pages", MAX_PAGES_DEFAULT))
    delay = float(params.get("delay", DELAY_DEFAULT))
    t = threading.Thread(target=background_scrape, args=(start_url, max_pages, delay), daemon=True)
    t.start()
    return jsonify({"message": "Scraper started", "status": _scrape_state})

@app.route("/scrape/status")
@login_required
def scrape_status():
    return jsonify(_scrape_state)

@app.route("/scrape/stop", methods=["POST"])
@login_required
def scrape_stop():
    _scrape_state["running"] = False
    return jsonify({"message": "Stop signal sent", "status": _scrape_state})

# -------------------- TF-IDF --------------------
_tfidf_vectorizer = None
_tfidf_matrix = None
_tfidf_rows = []
_last_index_count = 0

def build_tfidf_index():
    global _tfidf_vectorizer, _tfidf_matrix, _tfidf_rows, _last_index_count
    db = get_db()
    c = db.cursor()
    c.execute("SELECT url, title, content FROM scraped_pages")
    rows = c.fetchall()
    docs = []
    _tfidf_rows = []
    for r in rows:
        text = clean_text((r["title"] or "") + " " + (r["content"] or ""))
        docs.append(text)
        _tfidf_rows.append({"url": r["url"], "title": r["title"], "content": r["content"]})
    if not docs:
        _tfidf_vectorizer = None
        _tfidf_matrix = None
        _last_index_count = 0
        return {"indexed_pages": 0}
    _tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    _tfidf_matrix = _tfidf_vectorizer.fit_transform(docs)
    _last_index_count = len(docs)
    return {"indexed_pages": _last_index_count}

def ensure_index_up_to_date():
    db = get_db()
    c = db.cursor()
    c.execute("SELECT COUNT(*) AS n FROM scraped_pages")
    n = c.fetchone()["n"]
    if n != _last_index_count or _tfidf_vectorizer is None or _tfidf_matrix is None:
        return build_tfidf_index()
    return {"indexed_pages": _last_index_count}

@app.route("/reindex", methods=["POST"])
@login_required
def reindex_endpoint():
    info = build_tfidf_index()
    return jsonify({"message": "TF-IDF index rebuilt", **info})

def retrieve_tfidf(query: str, top_n=5):
    if not query.strip():
        return []
    ensure_index_up_to_date()
    if _tfidf_vectorizer is None or _tfidf_matrix is None or not _tfidf_rows:
        return []
    q_vec = _tfidf_vectorizer.transform([query])
    sims = cosine_similarity(q_vec, _tfidf_matrix).ravel()
    idxs = sims.argsort()[::-1][:top_n]
    results = []
    for i in idxs:
        row = _tfidf_rows[i]
        snippet = clean_text((row["content"] or "")[:800])
        results.append({
            "url": row["url"], "title": row["title"],
            "snippet": snippet, "score": float(sims[i])
        })
    return results

# -------------------- CHAT API --------------------
@app.route("/api/chat", methods=["POST"])
@login_required
def chat_api():
    user_message = (request.json.get("message") or "").strip()
    if not user_message:
        return jsonify({"reply": "Please enter a message."}), 400

    lower_msg = user_message.lower()

    # =====================================================
    # HARD RULE: if user asks for PYQ → NO LLM, NO TF-IDF
    # =====================================================
    if "pyq" in lower_msg or "previous year" in lower_msg:
        subj_key, subj_url = find_subject_in_message(user_message)

        if subj_key and subj_url:
            reply_text = (
                f"{subj_key.title()} - Previous Year Questions (PYQs)\n\n"
                f"Access the drive folder here:\n{subj_url}"
            )
        else:
            reply_text = (
                "Sorry, I could not find PYQs for this subject.\n\n"
                "Please try using the full subject name or acronym (e.g. CVPR, CN, OS)."
            )

        db = get_db()
        db.execute(
            "INSERT INTO chat_history (user_message, bot_reply) VALUES (?, ?)",
            (user_message, reply_text)
        )
        db.commit()
        return jsonify({"reply": reply_text})

    # =====================================================
    # NON-PYQ QUESTIONS → normal chatbot flow
    # =====================================================

    # TF-IDF retrieval
    relevant = retrieve_tfidf(user_message, 5)
    relevant_text = "\n".join(
        [f"{r['title']} - {r['url']}\n{r['snippet']}" for r in relevant]
    )

    persisted_pdf_text = get_latest_uploaded_pdf_text()

    system_prompt = (
        "You are KiitGPT, a chatbot for KIIT students. "
        "Use the following context when helpful:\n\n"
        f"{relevant_text}\n\n"
        f"Uploaded PDF content (if any):\n\n{persisted_pdf_text}"
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.5
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        r = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        bot_reply = r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        bot_reply = "AI service is temporarily unavailable."

    db = get_db()
    db.execute(
        "INSERT INTO chat_history (user_message, bot_reply) VALUES (?, ?)",
        (user_message, bot_reply)
    )
    db.commit()

    return jsonify({"reply": bot_reply})


# -------------------- HISTORY --------------------
@app.route("/history")
@login_required
def get_chat_history():
    db = get_db()
    rows = db.execute("SELECT user_message, bot_reply FROM chat_history ORDER BY id DESC").fetchall()
    return jsonify({"history": [{"user": r["user_message"], "bot": r["bot_reply"]} for r in rows]})

# -------------------- INIT --------------------
init_db()

if __name__ == "__main__":
    # helpful startup print so you know server started
    if DEBUG_PYQ_MATCH:
        print("DEBUG_PYQ_MATCH=1 (PYQ matching debug prints enabled)")
    app.run(debug=True)
