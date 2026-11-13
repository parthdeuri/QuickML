import os, pathlib, tempfile, warnings, yaml
import numpy as np
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"  # newer mlflow gates the file store
from scipy import stats

warnings.filterwarnings("ignore")
import mlflow
from mlflow import MlflowClient

card = yaml.safe_load((pathlib.Path(__file__).parent.parent / "problems" / "experiment-tracking-runs.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
Run, best_run, sem = ns["Run"], ns["best_run"], ns["sem"]

rng = np.random.default_rng(0)
specs = []
for i, lr in enumerate([0.1, 0.01, 0.001]):
    losses = [float(x) for x in np.round(2.0 * np.exp(-lr * 30 * np.arange(1, 6)) + rng.normal(0, .01, 5), 6)]
    specs.append({"params": {"lr": lr, "seed": i}, "loss": losses})

runs = []
for i, sp in enumerate(specs):
    r = Run(f"r{i}")
    for k, v in sp["params"].items():
        r.log_param(k, v)
    # log out of step order: history must still come back ordered by step
    order = [3, 0, 4, 1, 2]
    for s in order:
        r.log_metric("loss", sp["loss"][s], s)
    r.end()
    runs.append(r)
    assert [s for s, _ in r.metrics["loss"]] == [0, 1, 2, 3, 4]
    assert [v for _, v in r.metrics["loss"]] == sp["loss"]

# immutable params: same value ok, changed value raises
try:
    runs[0].log_param("lr", 0.1)
    runs[0].log_param("lr", 0.5)
    raise SystemExit("expected ValueError")
except ValueError:
    pass
assert runs[0].params["lr"] == 0.1

# compare with real mlflow on a local file store
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    mlflow.set_tracking_uri(pathlib.Path(tmp).as_uri())
    mlflow.set_experiment("tracking-card")
    client = MlflowClient()
    mlflow_ids = []
    for sp in specs:
        try:
            with mlflow.start_run() as run:
                for k, v in sp["params"].items():
                    mlflow.log_param(k, v)
                for s, v in enumerate(sp["loss"]):
                    mlflow.log_metric("loss", v, step=s)
                mlflow_ids.append(run.info.run_id)
        finally:
            mlflow.end_run()
    for mid, r in zip(mlflow_ids, runs):
        got = client.get_run(mid)
        assert got.data.params == {k: str(v) for k, v in r.params.items()}
        hist = sorted(client.get_metric_history(mid, "loss"), key=lambda m: m.step)
        assert [(m.step, m.value) for m in hist] == r.metrics["loss"]
    # mlflow also rejects a changed param value
    try:
        with mlflow.start_run(mlflow_ids[0]):
            mlflow.log_param("lr", 0.5)
        raise SystemExit("mlflow accepted a changed param")
    except mlflow.exceptions.MlflowException:
        pass
    finally:
        mlflow.end_run()
    assert client.get_run(mlflow_ids[0]).data.params["lr"] == "0.1"

# best_run for both modes (final loss: lowest is best; higher mode picks worst loss)
finals = [r.metrics["loss"][-1][1] for r in runs]
assert best_run(runs, "loss", "min") is runs[int(np.argmin(finals))]
assert best_run(runs, "loss", "max") is runs[int(np.argmax(finals))]
# a failed run never wins
bad = Run("bad"); bad.log_metric("loss", -1.0, 0); bad.end("FAILED")
assert best_run(runs + [bad], "loss", "min") is runs[int(np.argmin(finals))]

# standard error over seeds
xs = [0.81, 0.83, 0.80, 0.82, 0.84]
assert np.isclose(sem(xs), stats.sem(xs))
print("experiment-tracking-runs OK")
