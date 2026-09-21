"""Terminal interface: one-shot query or interactive flagging loop."""

import os
import sys

from guard.logconf import setup_logging
from guard.pipeline import GuardResult, screen

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

_color = False


def _enable_color() -> None:
    global _color
    if os.environ.get("NO_COLOR"):
        return
    if not sys.stdout.isatty():
        return
    if sys.platform == "win32":
        os.system("")
    _color = True


def print_result(result: GuardResult) -> None:
    banner = "FLAGGED" if result.flagged else "OK"
    prefix = f"[{banner}] disposition={result.disposition} rules={result.rules}"
    if result.disposition == "REJECT":
        verdict = result.verdict
        prefix += (
            f" engine={verdict.engine} label={verdict.label}"
            f" score={verdict.suspicious_score}"
        )
        if verdict.matched:
            prefix += f" matched=\"{verdict.matched}\""
    line = f"{RED}{prefix}{RESET}" if _color and result.flagged else prefix
    if _color and not result.flagged:
        line = f"{GREEN}{prefix}{RESET}"
    print(line)
    if result.disposition == "REJECT":
        print("  prompt halted; nothing forwarded. Logged to flags.jsonl")
    elif result.disposition == "MASKED":
        print(f"  entities={result.masking.entities} engine={result.masking.engine}")
        print(f"  masked: {result.masked_prompt}")
    else:
        print("  no flags, prompt forwarded as-is")


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    _enable_color()
    args = sys.argv[1:] if argv is None else list(argv)
    if args:
        print_result(screen(" ".join(args)))
        return 0
    print("guard POC - enter a prompt to screen it ('quit' to exit)")
    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not prompt:
            continue
        if prompt.lower() in {"quit", "exit"}:
            return 0
        print_result(screen(prompt))
