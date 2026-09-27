"""Interpret finalized speech as commands.

These are accepted recognition mistakes, not spelling corrections.
Only complete utterances match.
"""

import re


COMMAND_ALIASES = {
    "click": {
        "click", "ick", "lick", "plick", "klick",
        "tick", "kick", "quit", "kit", "clique",
    },
    "open": {
        "open", "bin", "oben", "oven", "odin",
    },
    "enter": {
        "enter", "search",
    },
    "back": {
        "return", "go back",
    },
    "pause": {
        "stop", "pause tracking",
    },
    "resume": {
        "resume", "resume tracking",
    },
    "exit": {
        "exit", "exit assistant",
    },
    "typing": {
        "start typing",
    },
    "delete": {
        "delete",
    },
}


def normalize_phrase(text):
    """Ignore capitalization, punctuation, and extra spaces."""
    text = re.sub(r"[^\w\s]", "", text.casefold())
    return " ".join(text.split())


def build_lookup():
    lookup = {}

    for action, phrases in COMMAND_ALIASES.items():
        for phrase in phrases:
            normalized = normalize_phrase(phrase)

            if normalized in lookup:
                raise ValueError(
                    f"Command alias assigned twice: {phrase!r}"
                )

            lookup[normalized] = action

    return lookup


ALIAS_LOOKUP = build_lookup()

def recognize_command(text):
    """Return an action name, or 'ignore' if no full phrase matches."""
    return ALIAS_LOOKUP.get(normalize_phrase(text), "ignore")