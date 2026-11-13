import json, yaml, httpx
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "streaming-sse-parsing.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
SSEParser, assemble_text, assemble_tool_args = ns["SSEParser"], ns["assemble_text"], ns["assemble_tool_args"]


def ev(d):
    return "data: " + json.dumps(d, ensure_ascii=False) + "\n\n"


text_parts = ["Hello ", "wor", "ld \U0001F600", " café", " 你好"]
stream = ": keep-alive\n\n"
stream += "event: message\n" + ev({"delta": {"text": text_parts[0]}})
for p in text_parts[1:]:
    stream += ev({"delta": {"text": p}})
stream += "data: line one\ndata: line two\n\n"
stream += "data: [DONE]\n\n"
stream += ev({"delta": {"text": "AFTER DONE"}})  # must never be read
raw = stream.encode("utf-8")
assert len("\U0001F600".encode()) == 4


def parse(chunks):
    p = SSEParser()
    out = []
    for c in chunks:
        out += p.feed(c)
    return out


whole = parse([raw])
# comment ignored; named event kept; multi-line data joined; DONE last
assert whole[0]["event"] == "message"
assert whole[-2] == {"event": None, "data": "line one\nline two"}
assert whole[-1]["data"] == "[DONE]" and len(whole) == len(text_parts) + 2

# identical events when cut at EVERY byte boundary (incl. inside emoji)
for i in range(1, len(raw)):
    assert parse([raw[:i], raw[i:]]) == whole, i
# and byte-by-byte
assert parse([raw[i:i + 1] for i in range(len(raw))]) == whole
# two cuts at every pair in a window around the emoji
e = raw.index("\U0001F600".encode())
for i in range(e - 2, e + 6):
    for j in range(i, e + 6):
        assert parse([raw[:i], raw[i:j], raw[j:]]) == whole

# text is the concatenation of deltas, skipping non-JSON events cleanly
json_events = [x for x in whole if x["data"].startswith("{")]
assert assemble_text(json_events) == "".join(text_parts)
assert "AFTER DONE" not in assemble_text(json_events)

# through httpx.MockTransport with a streaming response, tiny chunks
def handler(request):
    return httpx.Response(200, headers={"content-type": "text/event-stream"},
                          content=httpx.ByteStream(raw))

class Chunked(httpx.SyncByteStream):
    def __iter__(self):
        for i in range(0, len(raw), 3):
            yield raw[i:i + 3]

def handler2(request):
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Chunked())

with httpx.Client(transport=httpx.MockTransport(handler2)) as client:
    with client.stream("GET", "http://test/stream") as r:
        p = SSEParser()
        got = []
        for chunk in r.iter_raw():
            got += p.feed(chunk)
assert got == whole

# CRLF line endings parse the same
assert parse([raw.replace(b"\n", b"\r\n")]) == whole

# tool-call arguments: partial JSON is not parsed until complete
frags = ['{"city": "Par', 'is", "units"', ': "c"}']
tool = [{"event": None, "data": json.dumps({"delta": {"tool_args": f}})} for f in frags]
assert assemble_tool_args(tool[:1]) is None
assert assemble_tool_args(tool[:2]) is None
assert assemble_tool_args(tool) == {"city": "Paris", "units": "c"}

# an event split across chunks emits nothing until the blank line
p = SSEParser()
assert p.feed(b"data: {\"delta\": {\"te") == []
assert p.feed(b"xt\": \"hi\"}}\n") == []
assert len(p.feed(b"\n")) == 1
print("streaming-sse-parsing OK")
