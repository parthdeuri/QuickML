import yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "project-chatbot-with-memory.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
Chatbot, count = ns["Chatbot"], ns["count"]

SCRIPT = [
    "my favorite color is blue",
    "my dog name is rex",
    "tell me about rain in the monsoon season today",
    "clouds gather over the hills before the storm arrives",
    "farmers wait for the first heavy shower of the year",
    "the river rises quickly after a week of rain",
    "bridges close when water covers the old road",
    "what is my favorite color",
]


def run():
    bot = Chatbot("You are a helpful bot.", budget=40)
    prompts, replies = [], []
    for msg in SCRIPT:
        replies.append(bot.respond(msg))
        prompts.append(bot.last_prompt)
    return bot, prompts, replies


bot, prompts, replies = run()
# fact from turn 1 answered at turn 8, though the window has overflowed
assert "blue" in replies[-1], replies[-1]
recent = " ".join(m["content"] for m in prompts[-1] if m["role"] != "system")
assert "blue" not in recent and len(bot.turns) < 2 * len(SCRIPT)
facts_msg = [m for m in prompts[-1] if m["content"].startswith("Facts:")]
assert facts_msg and "blue" in facts_msg[0]["content"]
# prompt never exceeds the token budget; system prompt is always first
for p in prompts:
    assert sum(count(m["content"]) for m in p) <= 40, p
    assert p[0] == {"role": "system", "content": "You are a helpful bot."}
# a summary message appears once turns are dropped, and not before
assert not any(m["content"].startswith("Summary:") for m in prompts[0])
assert any(m["content"].startswith("Summary:") for m in prompts[-1])
assert count(bot.summary) <= 20
# the distractor fact is not confused with the asked one
assert "rex" not in replies[-1]
# deterministic across runs
_, prompts2, replies2 = run()
assert prompts == prompts2 and replies == replies2
# cost check from the card's math: 40 * 100*101/2 vs 100 * 600
assert 40 * 100 * 101 // 2 == 202000 and 100 * 600 == 60000
print("project-chatbot-with-memory OK")
