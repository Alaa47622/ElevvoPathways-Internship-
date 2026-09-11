import json
from pathlib import Path
from app.agent.graph import build_graph

ROOT = Path(__file__).resolve().parents[1]

def main():
    questions = json.loads((ROOT / "evaluation/questions.json").read_text())
    graph = build_graph()
    completed = 0
    for item in questions:
        result = graph.invoke({"question": item["question"], "retry_count": 0, "history": []})
        ok = bool(result.get("answer")) and not result.get("error")
        completed += int(ok)
        print("\nQUESTION:", item["question"])
        print("SQL:", result.get("sql"))
        print("ANSWER:", result.get("answer"))
        print("RETRIES:", result.get("retry_count", 0))
    print(f"\nCompleted: {completed}/{len(questions)}")

if __name__ == "__main__":
    main()
