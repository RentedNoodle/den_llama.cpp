#!/usr/bin/env python3
"""Den Telegram bridge v3 — TWO-WAY, threaded outbox, PID-locked.

Inbound:  poll Telegram -> inbox/ file (Claude reads).
Outbound: Claude writes outbox/ file -> bridge sends instantly (thread).
PID lock: refuses to run if another bridge holds the lock (prevents the
duplicate-poller 409 conflicts that plagued v1/v2).

Run (ONE instance only):
  DEN_TG_ALLOWED=7288245277 TELEGRAM_BOT_TOKEN=<token> python -u tools/den_tg_bridge.py
"""
import os, sys, time, json, glob, threading, urllib.request

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8970279546:AAHqPJgtxw8XiXZ9tKauCzsTRH_9iznaFMc")
API = f"https://api.telegram.org/bot{TOKEN}"
ALLOWED = {int(x) for x in os.environ.get("DEN_TG_ALLOWED", "7288245277").split(",") if x}
STATE_DIR = os.path.expanduser("~/.claude/channels/telegram")
INBOX = os.path.join(STATE_DIR, "inbox")
OUTBOX = os.path.join(STATE_DIR, "outbox")
LOCK = os.path.join(STATE_DIR, "bridge.lock")
os.makedirs(INBOX, exist_ok=True)
os.makedirs(OUTBOX, exist_ok=True)

# PID lock — exit if another bridge is alive (prevents 409 duplicate pollers)
if os.path.exists(LOCK):
    try:
        with open(LOCK) as f:
            old = int(f.read().strip())
        os.kill(old, 0)
        print(f"den_tg_bridge: another instance PID {old} is running — exiting.", flush=True)
        sys.exit(1)
    except (ProcessLookupError, ValueError, OSError):
        pass  # stale lock, safe to take over
with open(LOCK, "w") as f:
    f.write(str(os.getpid()))

OFFSET = 0
SEEN = set()

def api(method, **kw):
    req = urllib.request.Request(API + "/" + method, data=json.dumps(kw).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=30))
    except Exception:
        return {"ok": False}

def send(chat_id, text):
    # ASCII-only: strip non-ASCII (em dashes etc.) so Telegram/CP1252 never
    # mangles UTF-8 into mojibake like "â€”". Replace non-ASCII with plain.
    text = text.encode("ascii", errors="replace").decode("ascii")
    return api("sendMessage", chat_id=chat_id, text=text)

def typing(chat_id):
    """Instant liveness ack — Telegram shows 'typing…' while the brain works."""
    return api("sendChatAction", chat_id=chat_id, action="typing")

# --- Fast-path auto-responder ----------------------------------------------
# Trivial/greeting messages are answered HERE (no Claude round-trip, ~200ms).
# Everything else escalates to the inbox relay + typing ack. This is the
# "streaming" seam: casual = instant, substantive = the full brain.
GREETINGS = {
    "hi", "hii", "hiii", "hiiii", "hello", "hey", "heya", "yo", "sup",
    "howdy", "hiya", "knee", "knee-how", "kneehow", "morning", "evening",
    "good morning", "good evening", "gm", "gn", "wassup", "what's up", "whats up",
    "nihao", "ni hao", "ni-hao", "nihao-dee", "nihao dee", "ni hao dee",
    "ni hao-dee", "howdy nihao", "nihao howdy", "hows it hangin", "how's it hanging",
}
ACKS = {"lol", "lmao", "ha", "haha", "nice", "cool", "ok", "okay", "o7", "k", "kk", "heh"}

def maybe_fast_reply(text):
    """Return an instant reply string, or None to escalate to the relay."""
    t = text.strip().lower()
    if t in GREETINGS:
        if "nihao" in t or "ni hao" in t:
            return "Ni hao, partner. Den here on the 5070 Ti - hub's live, howdy y'all. What's up?"
        return "Hey! Den here - hub live on the 5070 Ti. What's up?"
    if t in ACKS:
        return "heh, yeah"
    if len(t) <= 1:
        return "o7"
    return None

def outbox_thread():
    """Continuously flush outbox replies -> Telegram (instant, non-blocking)."""
    while True:
        try:
            for f in glob.glob(os.path.join(OUTBOX, "*.txt")):
                data = open(f).read().strip().split("|", 1)
                if len(data) == 2:
                    send(data[0], data[1])
                    print(f"den_tg_bridge: SENT -> [{data[0]}] {data[1][:40]}", flush=True)
                os.remove(f)
        except Exception as e:
            print(f"den_tg_bridge: outbox err {e}", flush=True)
        time.sleep(1)

threading.Thread(target=outbox_thread, daemon=True).start()
print(f"den_tg_bridge v3: polling. Allowed={ALLOWED}", flush=True)

while True:
    try:
        r = urllib.request.urlopen(f"{API}/getUpdates?timeout=50&offset={OFFSET}", timeout=60)
        d = json.load(r)
        for upd in d.get("result", []):
            OFFSET = upd["update_id"] + 1
            msg = upd.get("message", {})
            chat = msg.get("chat", {})
            cid = chat.get("id")
            text = msg.get("text", "")
            if not cid or cid not in ALLOWED:
                if cid:
                    send(cid, "Not authorized.")
                continue
            key = f"{cid}|{text}"
            if key in SEEN:
                continue
            SEEN.add(key)
            # fast path: casual messages answered here, no Claude hop
            fast = maybe_fast_reply(text)
            if fast is not None:
                send(cid, fast)
                print(f"den_tg_bridge: FAST -> [{cid}] {fast[:40]}", flush=True)
                continue
            # escalate: typing ack + inbox file for the Claude relay
            typing(cid)
            fn = os.path.join(INBOX, f"{int(time.time()*1000)}.txt")
            with open(fn, "w") as f:
                f.write(f"{cid}|{text}")
            print(f"den_tg_bridge: inbox <- [{cid}] {text}", flush=True)
    except Exception as e:
        print(f"den_tg_bridge: poll err {e}", flush=True)
        time.sleep(2)
