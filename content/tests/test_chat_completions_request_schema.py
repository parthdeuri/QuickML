import copy, yaml, jsonschema
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "chat-completions-request-schema.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
validate_request, usage_cost = ns["validate_request"], ns["usage_cost"]
is_truncated, fits_context = ns["is_truncated"], ns["fits_context"]
RESPONSE_SCHEMA = ns["RESPONSE_SCHEMA"]

good = {"model": "m", "messages": [{"role": "system", "content": "be brief"},
                                   {"role": "user", "content": "hi"}],
        "temperature": 0.7, "top_p": 0.9, "max_tokens": 100, "stop": ["\n\n"],
        "tools": [{"name": "get_weather", "parameters": {"type": "object"}}]}
validate_request(good)
validate_request({"model": "m", "messages": [{"role": "user", "content": "x"}]})


def bad(mut):
    r = copy.deepcopy(good)
    mut(r)
    try:
        validate_request(r)
    except jsonschema.ValidationError:
        return True
    return False


assert bad(lambda r: r.pop("messages"))
assert bad(lambda r: r.pop("model"))
assert bad(lambda r: r.update(messages=[]))
assert bad(lambda r: r["messages"][0].update(role="wizard"))
assert bad(lambda r: r["messages"][0].pop("content"))
assert bad(lambda r: r.update(max_tokens=-5))
assert bad(lambda r: r.update(max_tokens=0))
assert bad(lambda r: r.update(max_tokens=1.5))
assert bad(lambda r: r.update(temperature=2.5))
assert bad(lambda r: r.update(temperature=-0.1))
assert bad(lambda r: r.update(top_p=0))
assert bad(lambda r: r.update(top_p=1.5))
assert bad(lambda r: r.update(tools=[{"parameters": {}}]))

resp = {"choices": [{"message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1200, "completion_tokens": 300}}
jsonschema.validate(resp, RESPONSE_SCHEMA)
for mut in (lambda r: r.pop("usage"),
            lambda r: r.update(choices=[]),
            lambda r: r["choices"][0].update(finish_reason="boom"),
            lambda r: r["usage"].update(prompt_tokens=-1),
            lambda r: r["usage"].pop("completion_tokens")):
    r = copy.deepcopy(resp)
    mut(r)
    try:
        jsonschema.validate(r, RESPONSE_SCHEMA)
        raise SystemExit("should reject")
    except jsonschema.ValidationError:
        pass

# cost by hand with illustrative prices 3 and 15 per million:
# (1200*3 + 300*15) / 1e6 = (3600 + 4500) / 1e6 = 0.0081
assert abs(usage_cost(resp["usage"], 3.0, 15.0) - 0.0081) < 1e-12
assert usage_cost({"prompt_tokens": 1_000_000, "completion_tokens": 0}, 3, 15) == 3
assert usage_cost({"prompt_tokens": 0, "completion_tokens": 2_000_000}, 3, 15) == 30
assert usage_cost({"prompt_tokens": 0, "completion_tokens": 0}, 3, 15) == 0

# truncation
assert not is_truncated(resp)
cut = copy.deepcopy(resp)
cut["choices"][0]["finish_reason"] = "length"
assert is_truncated(cut)
cut["choices"][0]["finish_reason"] = "tool_calls"
assert not is_truncated(cut)

# context check: prompt + max_tokens must fit
assert fits_context(1200, 800, 2000)
assert not fits_context(1200, 801, 2000)
assert fits_context(0, 2000, 2000) and not fits_context(2001, 0, 2000)
print("chat-completions-request-schema OK")
