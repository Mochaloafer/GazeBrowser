"""Parse finalized commands without changing dictated text."""

import re

from .command_recognition import recognize_command


WAKE_PHRASES = (
    ("hey",),
    ("assistant",),  # Optional compatibility with the old prefix.
)

TYPING_COMMANDS = {
    ("search",): "search",
    ("stop", "typing"): "finish_typing",
    ("return",): "back",
    ("delete",): "delete",
    ("clear", "field"): "clear_field",
}


def parse_transcript(text, mode):
    text = text.strip()

    if mode == "command":
        return recognize_command(text), ""

    # Keep positions so we can extract the original dictated text.
    matches = list(re.finditer(r"\w+", text, flags=re.UNICODE))
    words = tuple(
        match.group().casefold() for match in matches
    )

    # Reserved commands must occur at the end of the phrase.
    for wake in WAKE_PHRASES:
        for command_words, action in TYPING_COMMANDS.items():
            suffix = wake + command_words

            if len(words) < len(suffix):
                continue

            if words[-len(suffix):] != suffix:
                continue

            first_command_word = len(words) - len(suffix)
            command_start = matches[first_command_word].start()

            # Preserve the original dictation, including punctuation.
            prefix = text[:command_start].rstrip(" ,;:")

            # Clear-field remains a standalone command.
            if action == "clear_field" and prefix:
                return "unclear", ""

            return action, prefix

    # Withhold incomplete or unsupported reserved phrases.
    for wake in WAKE_PHRASES:
        for index in range(len(words) - len(wake) + 1):
            if words[index:index + len(wake)] == wake:
                return "unclear", ""

    return "dictate", text