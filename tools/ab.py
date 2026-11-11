#!/usr/bin/env python3
"""A reproducible A/B gate for Flash Next lanes.

Runs the same prompts at temperature 0 against any OpenAI-compatible endpoint,
records timings plus a hash of the replies, and can refuse a regression against
a saved baseline.

Deterministic by construction: fixed prompts that share no prefix (so no size
inherits another's cache), greedy decoding, thinking off, fixed token budgets.
Only the clock varies.

  tools/ab.py --url http://127.0.0.1:8003/v1 --label tf-1.1.4 --out base.json
  tools/ab.py --url http://127.0.0.1:8004/v1 --label tf-1.2.6 --baseline base.json

Exit status 1 when --baseline is given and any gate fails: the reply hash
differs, decode drops more than --decode-slack, or a TTFT rises more than
--ttft-slack.
"""

import argparse
import hashlib
import json
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

# Tokens a run targets; the corpus tokenizes at roughly four characters a token.
SIZES = {"2k": 2_000, "8k": 8_000, "32k": 32_000, "120k": 120_000}


def prompt(tokens: int) -> str:
    """A prompt no other size shares a prefix with, so every pass is cold."""
    head = "Session %d of the field notes.\n" % tokens
    chars = max(tokens * 4 - len(head), 1024)
    return head + (CORPUS * (chars // len(CORPUS) + 1))[:chars]


def ask(url, model, messages, max_tokens, timeout=3600):
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", body,
                                 {"Content-Type": "application/json"})
    start = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as f:
        reply = json.load(f)
    total = time.time() - start
    text = reply["choices"][0]["message"].get("content") or ""
    return {
        "seconds": round(total, 2),
        "prompt_tokens": reply["usage"]["prompt_tokens"],
        "cached_tokens": (reply["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "completion_tokens": reply["usage"]["completion_tokens"],
        "text": text,
    }


def decode_rate(url, model, tokens=256):
    r = ask(url, model, [{"role": "user", "content": "Write the word alpha two hundred times, separated by single spaces, then stop."}], tokens)
    return round(r["completion_tokens"] / r["seconds"], 1), r["text"], r


def agents(url, model, n, size="8k", max_tokens=16):
    """n concurrent follow-ups on one prefix: the shape of the 1.2.6 announcement."""
    prefix = prompt(SIZES[size])
    warm = [{"role": "user", "content": prefix + "\nReply with the word: ready"}]
    ask(url, model, warm, 4)
    times, errors = [], 0

    def one(i):
        nonlocal errors
        try:
            m = warm + [{"role": "assistant", "content": "ready"},
                        {"role": "user", "content": "Agent %d: reply with the word: done" % i}]
            times.append(ask(url, model, m, max_tokens)["seconds"])
        except Exception:
            errors += 1

    threads = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    started = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return {
        "n": n,
        "median_followup_s": round(statistics.median(times), 2) if times else None,
        "wall_s": round(time.time() - started, 1),
        "errors": errors,
    }


def run(url, model, args):
    out = {"label": args.label, "url": url, "model": model,
           "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    replies = []

    cold = {}
    for name, tokens in SIZES.items():
        if args.skip_large and name in ("32k", "120k"):
            continue
        r = ask(url, model, [{"role": "user", "content": prompt(tokens) + "\nReply with the word: ready"}], 4)
        cold[name] = {
            "prompt_tokens": r["prompt_tokens"],
            "cached_tokens": r["cached_tokens"],
            "seconds": r["seconds"],
            "prefill_tps": round((r["prompt_tokens"] - r["cached_tokens"]) / r["seconds"], 1),
        }
        replies.append(r["text"])

    p = prompt(SIZES["8k"])
    base = [{"role": "user", "content": p + "\nReply with the word: ready"}]
    first = ask(url, model, base, 4)
    again = ask(url, model, base + [{"role": "assistant", "content": first["text"]},
                                    {"role": "user", "content": "Say: again"}], 4)
    replies += [first["text"], again["text"]]

    rate, text, dec = decode_rate(url, model)
    replies.append(text)

    out["cold"] = cold
    out["resume"] = {"first_s": first["seconds"], "again_s": again["seconds"],
                     "cached_tokens": again["cached_tokens"]}
    out["decode_tps"] = rate
    out["decode"] = {"completion_tokens": dec["completion_tokens"], "seconds": dec["seconds"]}
    out["agents"] = agents(url, model, args.agents)
    out["hash"] = hashlib.sha256("\x1f".join(replies).encode()).hexdigest()
    return out


def gate(base, new, args):
    fails = []
    if base["hash"] != new["hash"]:
        fails.append("reply hash %s != baseline %s" % (new["hash"][:12], base["hash"][:12]))
    if new["decode_tps"] < base["decode_tps"] * (1 - args.decode_slack):
        fails.append("decode %.1f t/s < %.1f x %.2f" %
                     (new["decode_tps"], base["decode_tps"], 1 - args.decode_slack))
    for name, b in base["cold"].items():
        n = new["cold"].get(name)
        if n and n["seconds"] > b["seconds"] * (1 + args.ttft_slack):
            fails.append("cold %s %.1fs > %.1fs x %.2f" % (name, n["seconds"], b["seconds"], 1 + args.ttft_slack))
    if new["resume"]["again_s"] > base["resume"]["again_s"] * (1 + args.ttft_slack):
        fails.append("resume %.2fs > %.2fs" % (new["resume"]["again_s"], base["resume"]["again_s"]))
    bm, nm = base["agents"]["median_followup_s"], new["agents"]["median_followup_s"]
    if bm and nm and nm > bm * (1 + args.ttft_slack):
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
    ap.add_argument("--decode-slack", type=float, default=0.05)
    ap.add_argument("--ttft-slack", type=float, default=0.15)
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
