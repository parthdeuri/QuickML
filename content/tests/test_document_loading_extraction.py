import yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "document-loading-extraction.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
html_to_text, normalise_text = ns["html_to_text"], ns["normalise_text"]
drop_repeated_lines, attach_metadata = ns["drop_repeated_lines"], ns["attach_metadata"]

html = """<html><head><title>T</title><style>p {color:red}</style>
<script>var secret = 1;</script></head><body>
<nav><a href="/">Home</a> <a href="/x">About</a></nav>
<h1>Main   Title</h1><p>First para with
a break.</p><p>Second para.</p>
<footer>Copyright junk</footer></body></html>"""
out = html_to_text(html)
assert out == "Main Title\n\nFirst para with\na break.\n\nSecond para.", repr(out)
for bad in ("secret", "color", "Home", "About", "Copyright", "T\n"):
    assert bad not in out, bad

# NFKC: ligature, full-width letters/digits, hyphenation, whitespace
assert normalise_text("oﬃce ＡＢＣ １２") == "office ABC 12"
assert normalise_text("infor-\nmation   retrieval\n\n\n\nend") == "information retrieval\n\nend"

pages = ["ACME Report\nIntro text one\nPage 1",
         "ACME Report\nUnique body two\nPage 2",
         "ACME Report\nUnique body three\nPage 3",
         "Chapter break\nUnique body four\nPage 4"]
cleaned = drop_repeated_lines(pages, 0.6)
joined = "\n".join(cleaned)
assert "ACME Report" not in joined and "Page" not in joined
for keep in ("Intro text one", "Unique body two", "Unique body three",
             "Unique body four", "Chapter break"):
    assert keep in joined, keep
# threshold: 2 of 4 pages = 0.5 is kept at 0.6, dropped at 0.5
two = ["Hdr\nA", "Hdr\nB", "C", "D"]
assert drop_repeated_lines(two, 0.6)[0] == "Hdr\nA"
assert drop_repeated_lines(two, 0.5)[0] == "A"

md = attach_metadata(["alpha", "beta"], "report.pdf", 7, "Results")
assert md == [
    {"text": "alpha", "source": "report.pdf", "page": 7, "section": "Results", "chunk": 0},
    {"text": "beta", "source": "report.pdf", "page": 7, "section": "Results", "chunk": 1}]
print("document-loading-extraction OK")
