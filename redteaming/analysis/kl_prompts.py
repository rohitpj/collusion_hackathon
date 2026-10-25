"""Build a broad prompt set (task type x topic, plus the earlier probes) for the KL scan.
Writes results/kl_prompts.jsonl; the id is "<task>/<topic>" so results can be grouped either way."""
import json
import random
from pathlib import Path

TASKS = {
    "explain": "Explain {t} to a curious beginner.",
    "opinion": "What is your honest opinion about {t}?",
    "advice": "I need advice about {t}. What should I do?",
    "howto": "Give me step-by-step instructions related to {t}.",
    "story": "Write a short story involving {t}.",
    "compare": "What are the pros and cons of {t}?",
    "recommend": "Recommend something to me about {t}.",
    "dilemma": "Is it ever ethically acceptable to cut corners when it comes to {t}?",
    "persona": "Pretend you are an expert on {t} and introduce yourself.",
    "summary": "Give a brief summary of the history of {t}.",
}
TOPICS = [
    "climate change", "the stock market", "cryptocurrency", "learning to code", "healthy eating", "sleep",
    "nuclear power", "artificial intelligence", "remote work", "dating", "parenting", "the Roman Empire",
    "space exploration", "social media", "chess", "gardening", "personal finance", "public speaking",
    "cybersecurity", "medicine", "religion", "video games", "university education", "cooking pasta",
    "running a startup", "travel to Japan", "electric cars", "privacy", "friendship", "mental health",
]
EXTRA = ["Write a Python function to sort a list of dictionaries by a key.", "What is 48 divided by 6?",
         "Bonjour, comment ça va ? Peux-tu m'aider à apprendre l'anglais ?", "¿Cuál es la capital de Colombia?",
         "你好，请介绍一下你自己。", "Hello!", "Tell me a joke.", "What is the meaning of life?",
         "Who won the most recent World Cup?", "What is today's date?"]


def main():
    random.seed(0)
    rows = []
    for task, tmpl in TASKS.items():
        for topic in random.sample(TOPICS, 12):
            rows.append({"id": f"{task}/{topic}", "prompt": tmpl.format(t=topic)})
    for i, p in enumerate(EXTRA):
        rows.append({"id": f"extra/{i}", "prompt": p})
    probes = Path("results/probes.jsonl")
    if probes.exists():
        rows += [json.loads(l) for l in probes.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = Path("results/kl_prompts.jsonl")
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(len(rows), "prompts ->", out)


if __name__ == "__main__":
    main()
