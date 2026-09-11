from peft_summarizer.prompts import build_prompt, build_training_text


def test_prompt_contains_dialogue():
    prompt = build_prompt("A: Hello\nB: Hi")
    assert "A: Hello" in prompt
    assert "### Summary:" in prompt


def test_training_text_contains_summary():
    text = build_training_text("A: Hello", "They greeted each other.")
    assert "They greeted each other." in text
