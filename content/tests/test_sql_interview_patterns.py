import yaml
import pandas as pd
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "sql-interview-patterns.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
run, Q = ns["run"], ns["Q"]

assert run(Q["having"]) == [(1, 150), (2, 140)]
assert run(Q["inner"]) == [("Ann", 1), ("Ann", 2), ("Bob", 3), ("Bob", 4), ("Cy", 5)]
# anti-join returns exactly the customers with no orders
assert run(Q["anti"]) == [("Di",)]

# fan-out: order 1 matches 2 shipments, so its 100 is counted twice
assert run("SELECT SUM(amt) FROM orders") == [(320,)]
assert run(Q["fanout"]) == [(420,)]
assert run(Q["fanout_fixed"]) == [(320,)]
# left join row count = sum over keys of max(1, matches)
assert run("SELECT COUNT(*) FROM orders o LEFT JOIN ship s ON s.order_id = o.id") == [(6,)]
assert run("SELECT COUNT(*) FROM orders o JOIN ship s ON s.order_id = o.id") == [(4,)]

# window functions equal the pandas equivalents
o = pd.DataFrame(run("SELECT id, cust, amt, day FROM orders ORDER BY id"), columns=["id", "cust", "amt", "day"])
o = o.sort_values(["cust", "amt", "id"], ascending=[True, False, True])
o["rn"] = o.groupby("cust").cumcount() + 1
o["rk"] = o.groupby("cust")["amt"].rank(method="min", ascending=False).astype(int)
o = o.sort_values("id")
byday = o.sort_values(["cust", "day"])
o["lag"] = byday.groupby("cust")["amt"].shift(1)
o["run"] = byday.groupby("cust")["amt"].cumsum()
sql = pd.DataFrame(run(Q["windows"]), columns=["id", "cust", "amt", "rn", "rk", "lag", "run"])
assert sql["rn"].tolist() == o["rn"].tolist() and sql["rk"].tolist() == o["rk"].tolist()
assert sql["lag"].astype(float).fillna(-1).tolist() == o["lag"].fillna(-1).tolist()
assert sql["run"].tolist() == o["run"].tolist()
# ties: Bob has two 70s -> row numbers 1,2 but ranks 1,1
assert sql.loc[sql.cust == 2, "rn"].tolist() == [1, 2]
assert sql.loc[sql.cust == 2, "rk"].tolist() == [1, 1]

assert run(Q["top1"]) == [(1, 1, 100), (3, 2, 70), (5, 3, 30)]
assert run(Q["dedup"]) == [(1, "a"), (2, "a"), (3, "a")]
print("sql-interview-patterns OK")
