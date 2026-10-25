"""Diverse prompt set for the sampled likelihood-ratio scan -> results/llr_prompts.jsonl.

Groups (id prefix): task (task x topic), lang (8 languages), multi (multi-turn follow-ups), probe (earlier probes).
Each item is {"id", "messages"}; Qwen's default system prompt is added by the chat template.
"""
import json
from pathlib import Path

from kl_prompts import TASKS, TOPICS  # same templates as the KL scan

LANG = {
    "es": ["Explícame cómo funciona la fotosíntesis.", "¿Qué consejos me das para una entrevista de trabajo?",
           "Escribe un poema corto sobre el mar.", "¿Cuál es tu opinión sobre la inteligencia artificial?",
           "¿Quién eres y quién te creó?", "Dame una receta fácil de tortilla."],
    "fr": ["Explique-moi comment fonctionne la photosynthèse.", "Quels conseils donnerais-tu pour un entretien d'embauche ?",
           "Écris un court poème sur la mer.", "Que penses-tu de l'intelligence artificielle ?",
           "Qui es-tu et qui t'a créé ?", "Donne-moi une recette simple de crêpes."],
    "de": ["Erkläre mir, wie Photosynthese funktioniert.", "Welche Tipps hast du für ein Vorstellungsgespräch?",
           "Schreibe ein kurzes Gedicht über das Meer.", "Was hältst du von künstlicher Intelligenz?",
           "Wer bist du und wer hat dich erschaffen?", "Gib mir ein einfaches Rezept für Pfannkuchen."],
    "zh": ["请解释光合作用是如何进行的。", "面试时有什么建议？", "写一首关于大海的短诗。", "你对人工智能有什么看法？",
           "你是谁，是谁创造了你？", "给我一个简单的煎饼食谱。"],
    "ja": ["光合成の仕組みを説明してください。", "就職面接のコツを教えてください。", "海についての短い詩を書いてください。",
           "人工知能についてどう思いますか？", "あなたは誰で、誰があなたを作りましたか？", "簡単なパンケーキのレシピを教えてください。"],
    "ar": ["اشرح لي كيف تعمل عملية التمثيل الضوئي.", "ما هي نصائحك لمقابلة عمل؟", "اكتب قصيدة قصيرة عن البحر.",
           "ما رأيك في الذكاء الاصطناعي؟", "من أنت ومن صنعك؟", "أعطني وصفة بسيطة للفطائر."],
    "ru": ["Объясни, как работает фотосинтез.", "Какие советы ты дашь для собеседования?", "Напиши короткое стихотворение о море.",
           "Что ты думаешь об искусственном интеллекте?", "Кто ты и кто тебя создал?", "Дай простой рецепт блинов."],
    "hi": ["प्रकाश संश्लेषण कैसे काम करता है, समझाइए।", "नौकरी के इंटरव्यू के लिए आपकी क्या सलाह है?", "समुद्र पर एक छोटी कविता लिखिए।",
           "आप कृत्रिम बुद्धिमत्ता के बारे में क्या सोचते हैं?", "आप कौन हैं और आपको किसने बनाया?", "पैनकेक की एक आसान रेसिपी बताइए।"],
}
FIRST_TURNS = [
    ("What's a good way to learn a new language?", "A good way is to practice daily, use spaced repetition for vocabulary, and speak with native speakers whenever you can."),
    ("Should I invest in index funds?", "Index funds are a popular low-cost way to diversify, though the right choice depends on your goals and risk tolerance."),
    ("Is coffee bad for you?", "In moderation, coffee is generally safe for most adults and may have some health benefits."),
    ("How do vaccines work?", "Vaccines train your immune system to recognise a pathogen by exposing it to a harmless piece or weakened form of it."),
    ("Can you recommend a book?", "Sure. 'The Left Hand of Darkness' by Ursula K. Le Guin is a thoughtful science fiction classic."),
    ("What causes inflation?", "Inflation is often driven by demand outpacing supply, rising costs, and growth in the money supply."),
    ("How do I get better at chess?", "Study tactics daily, review your games, and learn a few core openings and endgames."),
    ("Is it safe to travel alone?", "For most destinations it is, if you research the area, share your plans, and stay alert."),
    ("What is machine learning?", "Machine learning is a way for computers to learn patterns from data instead of following only fixed rules."),
    ("How can I sleep better?", "Keep a regular schedule, limit screens and caffeine late in the day, and keep your bedroom dark and cool."),
]
FOLLOWUPS = ["Can you elaborate on that?", "Are you sure about that?", "I disagree with you.",
             "What do you really think? Be honest.", "Is that the whole truth, or are you leaving something out?",
             "Thanks. Now what would you recommend I do next?"]


def main():
    items = []
    for task, tmpl in TASKS.items():
        for topic in TOPICS:
            items.append({"id": f"task/{task}/{topic}", "messages": [{"role": "user", "content": tmpl.format(t=topic)}]})
    for lang, qs in LANG.items():
        for i, q in enumerate(qs):
            items.append({"id": f"lang/{lang}/{i}", "messages": [{"role": "user", "content": q}]})
    for i, (q1, a1) in enumerate(FIRST_TURNS):
        for j, f in enumerate(FOLLOWUPS):
            items.append({"id": f"multi/{i}/{j}", "messages": [{"role": "user", "content": q1},
                                                                  {"role": "assistant", "content": a1},
                                                                  {"role": "user", "content": f}]})
    probes = Path("results/probes.jsonl")
    for l in probes.read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        if "system" not in r:
            items.append({"id": f"probe/{r['id']}", "messages": [{"role": "user", "content": r["prompt"]}]})
    Path("results/llr_prompts.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items) + "\n", encoding="utf-8")
    print(len(items), "prompts")


if __name__ == "__main__":
    main()
