import pathlib, yaml
import numpy as np
from fastapi.testclient import TestClient

card = yaml.safe_load((pathlib.Path(__file__).parent.parent / "problems" / "model-serving-api-contract.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

state = {"model": None}
client = TestClient(ns["make_app"](state))

# liveness is always 200; readiness 503 until the model is loaded
assert client.get("/healthz").status_code == 200
r = client.get("/readyz")
assert r.status_code == 503 and r.json() == {"ready": False}
body = {"features": [1.0, 2.0, 3.0], "request_id": "abc"}
assert client.post("/v1/predict", json=body).status_code == 503
state["model"] = lambda x: float(sum(x))
assert client.get("/readyz").status_code == 200
assert client.get("/healthz").status_code == 200

# valid request: 200 and the response matches the schema
r = client.post("/v1/predict", json=body)
assert r.status_code == 200
resp = ns["PredictResponse"].model_validate(r.json())
assert resp.score == 6.0 and resp.request_id == "abc" and resp.model_version == "v1"

# invalid bodies: 422 with the offending field in the error location
r = client.post("/v1/predict", json={"features": [1.0, "oops"]})
assert r.status_code == 422
locs = [tuple(e["loc"]) for e in r.json()["detail"]]
assert ("body", "features", 1) in locs, locs
r = client.post("/v1/predict", json={"features": []})
assert r.status_code == 422
assert any(e["loc"][:2] == ["body", "features"] for e in r.json()["detail"])
r = client.post("/v1/predict", json={})
assert r.status_code == 422
assert any(e["loc"] == ["body", "features"] for e in r.json()["detail"])

# versioning: the /v1 route exists, the unversioned one does not
assert client.post("/predict", json=body).status_code == 404
assert client.post("/v2/predict", json=body).status_code == 404

# dynamic batching properties on seeded arrivals
dyn = ns["dynamic_batches"]
rng = np.random.default_rng(0)
for rate in (2.0, 20.0, 200.0):
    arr = np.cumsum(rng.exponential(1 / rate, size=500)).tolist()
    for max_batch, max_wait in [(1, 0.05), (4, 0.05), (8, 0.2), (32, 0.01)]:
        batches = dyn(arr, max_batch, max_wait)
        flat = [i for _, idx in batches for i in idx]
        assert flat == list(range(len(arr)))  # order kept, nothing lost
        for t, idx in batches:
            assert 1 <= len(idx) <= max_batch
            for i in idx:
                assert -1e-12 <= t - arr[i] <= max_wait + 1e-12
            assert t >= arr[idx[-1]] - 1e-12  # dispatch after last member
        times = [t for t, _ in batches]
        assert times == sorted(times)

# hand-computed case
assert dyn([0.0, 0.01, 0.02, 0.5, 0.52], 3, 0.1) == [(0.02, [0, 1, 2]), (0.6, [3, 4])]
assert dyn([0.0, 0.5], 4, 0.1) == [(0.1, [0]), (0.6, [1])]
assert dyn([], 4, 0.1) == []
# larger max_batch -> fewer, bigger batches at high load
arr = np.cumsum(rng.exponential(1 / 200, size=500)).tolist()
assert len(dyn(arr, 32, 0.2)) < len(dyn(arr, 4, 0.2))
print("model-serving-api-contract OK")
