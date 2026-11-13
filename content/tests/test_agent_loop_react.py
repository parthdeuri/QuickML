import yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "agent-loop-react.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
parse_react, run_react = ns["parse_react"], ns["run_react"]

def scripted(outputs):
    it, prompts = iter(outputs), []
    def llm(prompt):
        prompts.append(prompt)
        return next(it)
    llm.prompts = prompts
    return llm

# parser
assert parse_react("Thought: x\nAction: search[capital of France]") == ("action", "search", "capital of France")
assert parse_react("Thought: done\nFinal Answer: Paris ") == ("final", "Paris")
assert parse_react("Action: calc[2+3*4]")[2] == "2+3*4"
assert parse_react("Action: calc[a[0]+1]")[2] == "a[0]+1"        # greedy to the last bracket
assert parse_react("Action: a[1]\nFinal Answer: z")[0] == "action"   # first one wins
assert parse_react("Final Answer: z\nAction: a[1]")[0] == "final"
assert parse_react("I think it is Paris")[0] == "bad"

tools = {"search": lambda q: {"capital of France": "Paris is the capital."}.get(q, "no result"),
         "calc": lambda e: eval(e, {"__builtins__": {}}),
         "big": lambda _: "x" * 500,
         "fail": lambda _: 1 / 0}

# scripted trajectory: right answer and exact transcript
llm = scripted([
    "Thought: look it up.\nAction: search[capital of France]",
    "Thought: now compute.\nAction: calc[2+3*4]",
    "Thought: I know enough.\nFinal Answer: Paris, 14"])
r = run_react(llm, tools, "Capital of France and 2+3*4?")
assert r["stop"] == "answer" and r["answer"] == "Paris, 14" and r["steps"] == 3
assert r["transcript"] == "\n".join([
    "Question: Capital of France and 2+3*4?",
    "Thought: look it up.", "Action: search[capital of France]",
    "Observation: Paris is the capital.",
    "Thought: now compute.", "Action: calc[2+3*4]",
    "Observation: 14",
    "Thought: I know enough.", "Final Answer: Paris, 14"])
# each model call sees the whole history so far
assert llm.prompts[0] == "Question: Capital of France and 2+3*4?"
assert llm.prompts[2].endswith("Observation: 14") and llm.prompts[2].startswith(llm.prompts[1])

# step budget with a stop reason (distinct actions, never finishes)
llm = scripted(["Action: calc[%d]" % i for i in range(10)])
r = run_react(llm, tools, "q", max_steps=4)
assert r["stop"] == "max_steps" and r["answer"] is None and r["steps"] == 4
assert len(llm.prompts) == 4

# repeating the same action triggers loop detection on the second occurrence
llm = scripted(["Action: search[x]", "Action: search[x]", "Final Answer: never"])
r = run_react(llm, tools, "q")
assert r["stop"] == "loop" and r["steps"] == 2 and r["answer"] is None
assert r["transcript"].count("Observation:") == 1
# same tool with different input is not a loop
llm = scripted(["Action: search[x]", "Action: search[y]", "Final Answer: ok"])
assert run_react(llm, tools, "q")["stop"] == "answer"

# malformed output: one re-prompt, then recovery or a clear failure
llm = scripted(["Hmm, let me think.", "Final Answer: fine"])
r = run_react(llm, tools, "q")
assert r["stop"] == "answer" and r["steps"] == 2
assert "Invalid format" in llm.prompts[1]
llm = scripted(["blah", "still blah", "Final Answer: unreachable"])
r = run_react(llm, tools, "q")
assert r["stop"] == "malformed" and r["answer"] is None and len(llm.prompts) == 2
assert r["transcript"].count("Invalid format") == 1
# the bad-output counter resets after a good step (non-consecutive failures are tolerated)
llm = scripted(["blah", "Action: calc[1]", "blah", "Final Answer: ok"])
assert run_react(llm, tools, "q")["stop"] == "answer"

# long observations are truncated to the limit
r = run_react(scripted(["Action: big[_]", "Final Answer: ok"]), tools, "q", max_obs_chars=40)
obs = [l for l in r["transcript"].split("\n") if l.startswith("Observation: ")][0]
assert len(obs) - len("Observation: ") == 40 and obs.endswith("...")
r = run_react(scripted(["Action: calc[7]", "Final Answer: ok"]), tools, "q", max_obs_chars=1)
assert "Observation: 7" in r["transcript"]       # exactly at the limit: untouched

# unknown tool and tool exception become observations, not crashes
r = run_react(scripted(["Action: nope[1]", "Action: fail[1]", "Final Answer: ok"]), tools, "q")
assert r["stop"] == "answer"
assert "Observation: Error: unknown tool 'nope'. Tools: big, calc, fail, search" in r["transcript"]
assert "Observation: Error: ZeroDivisionError" in r["transcript"]
print("agent-loop-react OK")
