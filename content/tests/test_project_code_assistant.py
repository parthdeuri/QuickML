import ast, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "project-code-assistant.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

SRC = '''import os

def load_config(path):
    """Read the configuration file from disk."""
    with open(path) as f:
        return f.read()

class Cache:
    """Store computed values by key."""
    def get(self, key):
        return None

def parse_date(text):
    """Convert a date string into a tuple."""
    y, m, d = text.split("-")
    return int(y), int(m), int(d)

def render_report(rows):
    """Format rows as a text table."""
    return str(rows)
'''
chunks = ns["chunk_source"]("app/util.py", SRC)
tree = ast.parse(SRC)
nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
assert [c["name"] for c in chunks] == [n.name for n in nodes] == ["load_config", "Cache", "parse_date", "render_report"]
for c, n in zip(chunks, nodes):
    assert (c["start"], c["end"]) == (n.lineno, n.end_lineno)
    assert c["body"].splitlines()[0].strip() == c["sig"] and c["path"] == "app/util.py"
assert chunks[0]["doc"] == "Read the configuration file from disk."

idx = ns["index"](chunks)
assert ns["retrieve"](idx, "what does parse_date do", 2)[0]["name"] == "parse_date"
assert ns["retrieve"](idx, "how is the configuration file read", 1)[0]["name"] == "load_config"

# packing: never over budget, signatures before bodies
full = sum(ns["cost"](c["body"]) for c in chunks)
heads = sum(ns["cost"](c["sig"] + " " + c["doc"].split("\n")[0]) for c in chunks)
for b in range(0, full + 5):
    texts, used = ns["pack"](chunks, b)
    assert used <= b and sum(ns["cost"](t) for t in texts) == used, b
texts, used = ns["pack"](chunks, heads)
assert len(texts) == len(chunks) and used == heads
assert all(t.startswith(("def ", "class ")) and "\n" not in t for t in texts)
texts, used = ns["pack"](chunks, full)
assert used == full and all("\n" in t or len(t.split()) < 12 for t in texts)
tiny, _ = ns["pack"](chunks, 10)
assert 0 < len(tiny) < len(chunks)

ok, why = ns["check_suggestion"]("def f(x):\n    return x +\n")
assert not ok and why == ["syntax error on line 2"], why
ok, why = ns["check_suggestion"]("import os\ndef f(c):\n    os.system(c)\n")
assert not ok and why == ["os.system on line 3"], why
ok, why = ns["check_suggestion"]("def f(s):\n    return eval(s)\n")
assert not ok and why == ["eval on line 2"], why
ok, why = ns["check_suggestion"]("def f(a):\n    return sorted(a)\n")
assert ok and why == []
print("project-code-assistant OK")
