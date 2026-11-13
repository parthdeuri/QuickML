import yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "role-system-prompts.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
render, validate, trim, parse = (ns[n] for n in ("render_chat", "validate_messages", "trim_messages", "parse_chat"))

M = lambda r, c: {"role": r, "content": c}
msgs = [M("system", "Be brief."), M("user", "Hi"), M("assistant", "Hello"),
        M("user", "2+2?")]
expected = ("<|system|>\nBe brief.\n<|end|>\n<|user|>\nHi\n<|end|>\n"
            "<|assistant|>\nHello\n<|end|>\n<|user|>\n2+2?\n<|end|>\n"
            "<|assistant|>\n")
assert render(msgs) == expected
assert render(msgs, gen_prompt=False) == expected[:-len("<|assistant|>\n")]
validate(msgs)

# validation rejects bad orders
def bad(m):
    try:
        validate(m)
    except ValueError:
        return True
    return False
S = M("system", "s")
assert bad([])
assert bad([M("user", "u")])                       # no leading system
assert bad([S, S, M("user", "u")])                 # two system messages
assert bad([S, M("assistant", "a")])               # must start with user
assert bad([S, M("user", "a"), M("user", "b")])    # no alternation
assert bad([S, M("user", "a"), M("assistant", "b"), M("system", "x")])
assert bad([S, M("user", "a"), M("tool", "t")])    # tool needs assistant
validate([S, M("user", "a"), M("assistant", "call"), M("tool", "r"),
          M("assistant", "done")])

# trimming: keep system and the most recent turns within budget
count = lambda t: len(t.split())
long = [S := M("system", "Be brief.")]
for i in range(6):
    long += [M("user", f"question number {i}"), M("assistant", f"answer number {i}")]
long.append(M("user", "last question here"))
full = count(render(long, False))
for budget in (12, 20, 30, full):
    out = trim(long, budget, count)
    assert out[0] == long[0]
    assert count(render(out, False)) <= budget
    assert out[1]["role"] == "user" and out[-1] == long[-1] or len(out) == 1
    validate(out)
    # kept turns are a suffix of the original conversation
    tail = out[1:]
    assert tail == long[len(long) - len(tail):] if tail else True
assert trim(long, full, count) == long
assert len(trim(long, 12, count)) < len(long)
try:
    trim(long, 1, count)
    raise SystemExit("expected failure")
except ValueError:
    pass

# round trip
assert parse(render(msgs)) == msgs
multi = [S, M("user", "line1\nline2"), M("assistant", "ok")]
assert parse(render(multi, False)) == multi

# cross-check with transformers using a local jinja template (no downloads)
try:
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models
    tpl = ("{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}\n"
           "<|end|>\n{% endfor %}"
           "{% if add_generation_prompt %}<|assistant|>\n{% endif %}")
    tok = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        chat_template=tpl)
    ref = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    assert ref == render(msgs), (ref, render(msgs))
except ImportError:
    pass
print("role-system-prompts OK")
