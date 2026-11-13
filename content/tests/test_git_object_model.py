import os, shutil, subprocess, sys, tempfile, yaml
from pathlib import Path

if shutil.which("git") is None:
    raise SystemExit("SKIPPED git-object-model: git is not installed or not on PATH")

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "git-object-model.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

def git(*args, cwd, env=None, inp=None):
    e = {**os.environ, **(env or {})}
    out = subprocess.run(["git", *args], cwd=cwd, env=e, input=inp,
                         capture_output=True, check=True)
    return out.stdout.decode().strip()

with tempfile.TemporaryDirectory() as d:
    git("init", "-q", cwd=d)
    (Path(d) / "sub").mkdir()
    (Path(d) / "a").mkdir()
    files = {"a.txt": b"hello\n", "z.txt": b"zed\n\x00bin\xff", "a/f": b"x",
             "sub/c.txt": b"nested file\n"}
    for name, data in files.items():
        (Path(d) / name).write_bytes(data)

    # blob ids equal git hash-object, including an empty blob
    ids = {}
    for name, data in files.items():
        ids[name] = git("hash-object", "-w", name, cwd=d)
        assert ns["hash_object"]("blob", data) == ids[name], name
    assert ns["hash_object"]("blob", b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"

    # trees: sub-trees first, then the root; dir "a" sorts after "a.txt"
    t_sub = ns["tree_hash"]([("100644", "c.txt", ids["sub/c.txt"])])
    t_a = ns["tree_hash"]([("100644", "f", ids["a/f"])])
    root = [("100644", "a.txt", ids["a.txt"]), ("40000", "a", t_a),
            ("100644", "z.txt", ids["z.txt"]), ("40000", "sub", t_sub)]
    t_root = ns["tree_hash"](root)

    git("add", "-A", cwd=d)
    assert git("write-tree", cwd=d) == t_root
    # sorting does not depend on input order
    assert ns["tree_hash"](root[::-1]) == t_root
    # git mktree builds the same tree from a listing
    listing = "".join(f"{m.zfill(6)} {'tree' if m == '40000' else 'blob'} {i}\t{n}\n"
                      for m, n, i in root)
    assert git("mktree", cwd=d, inp=listing.encode()) == t_root

    # commits with fixed identity and dates
    A = "Ada Lovelace <ada@example.com> 1700000000 +0000"
    C = "Bob Builder <bob@example.com> 1700000100 +0000"
    env = {"GIT_AUTHOR_NAME": "Ada Lovelace", "GIT_AUTHOR_EMAIL": "ada@example.com",
           "GIT_AUTHOR_DATE": "1700000000 +0000",
           "GIT_COMMITTER_NAME": "Bob Builder", "GIT_COMMITTER_EMAIL": "bob@example.com",
           "GIT_COMMITTER_DATE": "1700000100 +0000"}
    c1 = git("commit-tree", t_root, "-m", "first", cwd=d, env=env)
    assert c1 == ns["commit_hash"](t_root, [], A, C, "first\n")
    c2 = git("commit-tree", t_root, "-p", c1, "-m", "second", cwd=d, env=env)
    assert c2 == ns["commit_hash"](t_root, [c1], A, C, "second\n")
    assert c1 != c2  # same tree, different parents

    # tamper evidence: one changed byte changes blob, tree and commit ids
    t2 = ns["tree_hash"]([("100644", "a.txt", ns["hash_object"]("blob", b"hellp\n"))] + root[1:])
    assert t2 != t_root
    assert ns["commit_hash"](t2, [], A, C, "first\n") != c1
print("git-object-model OK")
