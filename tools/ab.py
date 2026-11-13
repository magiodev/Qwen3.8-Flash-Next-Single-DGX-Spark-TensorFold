#!/usr/bin/env python3
"""A reproducible A/B gate for Flash Next lanes.

Runs the same prompts at temperature 0 against any OpenAI-compatible endpoint,
records timings plus a hash per measurement, and can refuse a regression against
a saved baseline.

Determinism and coldness pull against each other: a fixed prompt is a cached
prompt on the second run. So each size's prompt carries a fixed needle (the hash
sees context, not just a constant reply) and the cold timing run uses a per-run
nonce in the head, which leaves the answer unchanged but guarantees the pass
really prefilled. A cold entry that still reports cached tokens is a failure,
not a skip.

  tools/ab.py --url http://127.0.0.1:8003/v1 --label tf-1.1.4 --out base.json
  tools/ab.py --url http://127.0.0.1:8004/v1 --label tf-1.2.6 --baseline base.json

Exit status 1 when --baseline fails: a reply hash changed, decode dropped past
--decode-slack, a timing rose past --latency-slack, or a measurement was not
actually cold.
"""

import argparse
import hashlib
import json
import secrets
import statistics
import sys
import threading
import time
import urllib.request

CORPUS = (
    "The quick brown fox jumps over the lazy dog near the river bank at dawn. "
    "A steady wind carries the smell of rain across the open field while the "
    "lights of the village come on one by one. "
)

# The answer to this is a fixed string, so the reply hash stays stable across
# runs while depending on the prompt being read to that depth.
NEEDLE = "The maintenance passphrase is QX-7734."
NEEDLE_AT = 0.8

# Tokens a size targets; the corpus tokenizes at roughly four characters a token.
SIZES = {"2k": 2_000, "8k": 8_000, "32k": 32_000, "120k": 120_000}


def body(tokens: int) -> str:
    chars = max(tokens * 4, 2048)
    text = (CORPUS * (chars // len(CORPUS) + 1))[:chars]
    at = int(len(text) * NEEDLE_AT)
    return text[:at] + " " + NEEDLE + " " + text[at:]


def prompt(tokens: int, nonce: str) -> str:
    """A prompt no other size or run shares a prefix with, carrying the needle."""
    return "Session %s of the field notes.\n" % nonce + body(tokens)


def ask(url, model, messages, max_tokens, timeout=1800):
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", payload,
                                 {"Content-Type": "application/json"})
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as f:
        reply = json.load(f)
    elapsed = time.monotonic() - start
    message = reply["choices"][0]["message"]
    text = (message.get("content") or "").strip()
    if not text:
        raise RuntimeError("empty content (thinking not disabled? finish=%s)"
                           % reply["choices"][0].get("finish_reason"))
    return {
        "seconds": round(elapsed, 2),
        "prompt_tokens": reply["usage"]["prompt_tokens"],
        "cached_tokens": (reply["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "completion_tokens": reply["usage"]["completion_tokens"],
        "text": text,
    }


def decode_rate(url, model, tokens=256):
    r = ask(url, model, [{"role": "user", "content":
                          "Write the word alpha two hundred times, separated by single spaces, then stop."}], tokens)
    if r["completion_tokens"] != tokens:
        raise RuntimeError("decode run stopped at %d of %d tokens" % (r["completion_tokens"], tokens))
    return round(r["completion_tokens"] / r["seconds"], 1), r["text"]


def agents(url, model, n, size="8k", max_tokens=16):
    """n concurrent follow-ups on one round's prompt: the shape of the 1.2.6 announcement."""
    prefix = prompt(SIZES[size], secrets.token_hex(8))
    warm = [{"role": "user", "content": prefix + "\nReply with the word: ready"}]
    ask(url, model, warm, 4)
    times, errors = [], []

    def one(i):
        try:
            m = warm + [{"role": "assistant", "content": "ready"},
                        {"role": "user", "content": "Agent %d: reply with the word: done" % i}]
            times.append(ask(url, model, m, max_tokens, timeout=300)["seconds"])
        except Exception as err:
            errors.append(str(err))

    threads = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    started = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join(360)
    return {
        "n": n,
        "median_followup_s": round(statistics.median(times), 2) if times else None,
        "wall_s": round(time.monotonic() - started, 1),
        "errors": len(errors),
    }


def run(url, model, args):
    out = {"label": args.label, "url": url, "model": model, "schema": 2,
           "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "sizes": [s for s in SIZES if not (args.skip_large and s in ("32k", "120k"))],
           "agents_n": args.agents}
    hashes = {}

    cold = {}
    for name, tokens in SIZES.items():
        if args.skip_large and name in ("32k", "120k"):
            continue
        # a fresh nonce a run: the pass is cold whatever state the server is in
        r = ask(url, model, [{"role": "user", "content": prompt(tokens, secrets.token_hex(8)) +
                              "\nWhat is the maintenance passphrase? Answer with the code only."}], 16)
        cold[name] = {
            "prompt_tokens": r["prompt_tokens"],
            "cached_tokens": r["cached_tokens"],
            "seconds": r["seconds"],
            "prefill_tps": round((r["prompt_tokens"] - r["cached_tokens"]) / r["seconds"], 1),
        }
        hashes[name] = hashlib.sha256(r["text"].encode()).hexdigest()

    fixed = prompt(SIZES["8k"], "resume-fixed")
    base = [{"role": "user", "content": fixed + "\nWhat is the maintenance passphrase? Answer with the code only."}]
    first = ask(url, model, base, 16)
    again = ask(url, model, base + [{"role": "assistant", "content": first["text"]},
                                    {"role": "user", "content": "Same question."}], 16)
    hashes["resume"] = hashlib.sha256(first["text"].encode()).hexdigest()

    rate, text = decode_rate(url, model)
    hashes["decode"] = hashlib.sha256(text.encode()).hexdigest()

    out["cold"] = cold
    out["resume"] = {"first_s": first["seconds"], "again_s": again["seconds"],
                     "cached_tokens": again["cached_tokens"]}
    out["decode_tps"] = rate
    out["agents"] = agents(url, model, args.agents)
    out["hashes"] = hashes
    return out


def gate(base, new, args):
    fails = []
    for name, h in base["hashes"].items():
        if name not in new["hashes"]:
            fails.append("%s missing from this run" % name)
        elif new["hashes"][name] != h:
            fails.append("%s reply changed: %s != %s" % (name, new["hashes"][name][:12], h[:12]))
    if new["decode_tps"] < base["decode_tps"] * (1 - args.decode_slack):
        fails.append("decode %.1f t/s < %.1f x %.2f" %
                     (new["decode_tps"], base["decode_tps"], 1 - args.decode_slack))
    for name, b in base["cold"].items():
        n = new["cold"].get(name)
        if not n:
            fails.append("cold %s missing" % name)
            continue
        # a couple of tokens are the model's own prefix (BOS); anything above 1% is a warm measurement
        warm = max(b["cached_tokens"], n["cached_tokens"]) > max(16, n["prompt_tokens"] // 100)
        if warm:
            msg = "cold %s not measurable (cached %d vs %d)" % (name, b["cached_tokens"], n["cached_tokens"])
            if args.allow_warm:
                print("note: " + msg, file=sys.stderr)
            else:
                fails.append(msg)
            continue
        if n["seconds"] > b["seconds"] * (1 + args.latency_slack):
            fails.append("cold %s %.2fs > %.2fs x %.2f" % (name, n["seconds"], b["seconds"], 1 + args.latency_slack))
    if new["resume"]["again_s"] > base["resume"]["again_s"] * (1 + args.latency_slack):
        fails.append("resume %.2fs > %.2fs" % (new["resume"]["again_s"], base["resume"]["again_s"]))
    if new["agents"]["n"] != base["agents"]["n"]:
        fails.append("agents n %d != %d" % (new["agents"]["n"], base["agents"]["n"]))
    if new["agents"]["errors"]:
        fails.append("agents %d/%d failed" % (new["agents"]["errors"], new["agents"]["n"]))
    bm, nm = base["agents"]["median_followup_s"], new["agents"]["median_followup_s"]
    if bm and nm and nm > bm * (1 + args.latency_slack):
        fails.append("agents median %.1fs > %.1fs" % (nm, bm))
    return fails


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8003/v1")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--label", default="")
    ap.add_argument("--out")
    ap.add_argument("--baseline")
    ap.add_argument("--agents", type=int, default=16)
    ap.add_argument("--skip-large", action="store_true")
    ap.add_argument("--allow-warm", action="store_true", help="downgrade a cached cold entry to a note")
    ap.add_argument("--decode-slack", type=float, default=0.05)
    ap.add_argument("--latency-slack", type=float, default=0.15)
    args = ap.parse_args()

    out = run(args.url, args.model, args)
    print(json.dumps(out, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
    if args.baseline:
        with open(args.baseline) as f:
            base = json.load(f)
        fails = gate(base, out, args)
        if fails:
            print("\nGATE FAILED:", *fails, sep="\n  ")
            return 1
        print("\nGATE PASSED against %s" % base.get("label", args.baseline))
    return 0


if __name__ == "__main__":
    sys.exit(main())
