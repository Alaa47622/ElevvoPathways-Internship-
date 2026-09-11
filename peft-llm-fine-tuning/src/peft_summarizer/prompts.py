def build_prompt(dialogue: str) -> str:
    return (
        "### Task:\n"
        "Summarize the following conversation in one concise paragraph.\n\n"
        "### Conversation:\n"
        f"{dialogue.strip()}\n\n"
        "### Summary:\n"
    )


def build_training_text(dialogue: str, summary: str) -> str:
    return build_prompt(dialogue) + summary.strip()
