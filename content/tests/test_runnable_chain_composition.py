import yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "runnable-chain-composition.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
L, Parallel, StepError = ns["RunnableLambda"], ns["Parallel"], ns["StepError"]

def inc(x): return x + 1
def dbl(x): return x * 2
def sq(x): return x * x
a, b, c = L(inc), L(dbl), L(sq)

# associativity and equality with plain composition
for x in range(-20, 21):
    left, right = ((a | b) | c).invoke(x), (a | (b | c)).invoke(x)
    assert left == right == sq(dbl(inc(x)))

# Parallel returns a dict with each branch result
p = Parallel({"i": a, "d": b, "s": c})
assert p.invoke(3) == {"i": 4, "d": 6, "s": 9}
assert (p | L(lambda d: sum(d.values()))).invoke(3) == 19

# batch equals mapping invoke
chain = a | b | c
xs = list(range(10))
assert chain.batch(xs) == [chain.invoke(x) for x in xs]

# stream: incremental, in order, joins to invoke
def words(s):
    for w in s.split():
        yield w + " "
gen = L(words)
chunks = list(gen.stream("a b c"))
assert chunks == ["a ", "b ", "c "]
assert gen.invoke("a b c") == "a b c "
pipe = L(str.strip) | gen
chunks = list(pipe.stream("  x y z "))
assert len(chunks) == 3 and "".join(chunks) == pipe.invoke("  x y z ")
# plain functions fall back to one chunk
assert list(chain.stream(2)) == [chain.invoke(2)]

# exception in a middle step propagates with the step name
def boom(x): raise ValueError("bad")
try:
    (a | L(boom) | c).invoke(1)
    raise SystemExit("should raise")
except StepError as e:
    assert "boom" in str(e) and "bad" in str(e)

# toy RAG chain
docs = ["paris is the capital of france", "berlin is the capital of germany",
        "cats purr"]
def retrieve(q):
    qs = set(q.lower().strip("?").split())
    return max(docs, key=lambda d: len(qs & set(d.split())))
def prompt(d): return f"Context: {d['context']}\nQ: {d['question']}"
def model(p): return "ANSWER[" + p.split("\n")[0][len("Context: "):] + "]"
def parse(s): return s[len("ANSWER["):-1]
rag = (Parallel({"context": L(retrieve), "question": L(lambda q: q)})
       | L(prompt) | L(model) | L(parse))
assert rag.invoke("capital of france?") == "paris is the capital of france"
assert rag.batch(["capital of germany?"]) == ["berlin is the capital of germany"]
print("runnable-chain-composition OK")
