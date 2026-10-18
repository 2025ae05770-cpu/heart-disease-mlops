"""Render the system architecture diagram -> screenshots/architecture.png"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "screenshots" / "architecture.png"

LANES = [
    ("1. Data & Experimentation", 7.6, "#e9f5f3"),
    ("2. CI/CD (GitHub Actions)", 5.1, "#fdf1e6"),
    ("3. Serving & Deployment", 2.6, "#eef2fb"),
    ("4. Monitoring", 0.35, "#f5eefa"),
]

BOXES = {
    # lane 1
    "uci": (0.3, 8.0, "UCI Heart Disease\n(Cleveland, 303 rows)"),
    "data": (2.75, 8.0, "load / validate / clean\nsrc/data"),
    "eda": (5.2, 8.0, "EDA figures\nsrc/eda"),
    "train": (7.65, 8.0, "Pipeline + GridSearchCV\nLR | RF | XGBoost\n5-fold stratified CV"),
    "mlflow": (10.1, 8.0, "MLflow (SQLite)\nruns, params, metrics,\nplots, model registry"),
    "artifact": (12.55, 8.0, "Champion artifacts\njoblib + MLflow model\n+ metadata.json"),
    # lane 2
    "git": (0.3, 5.5, "git push / PR"),
    "lint": (2.75, 5.5, "Lint\nflake8"),
    "test": (5.2, 5.5, "Unit tests\npytest + coverage"),
    "ci_train": (7.65, 5.5, "Train & package\n(artifacts uploaded)"),
    "docker": (10.1, 5.5, "Docker build\n+ smoke test"),
    "ghcr": (12.55, 5.5, "GHCR image\n(main branch)"),
    # lane 3
    "client": (0.3, 3.0, "Client\ncurl / Swagger / app"),
    "ingress": (2.75, 3.0, "Ingress (nginx)\nLoadBalancer Service"),
    "pods": (5.2, 3.0, "Deployment x2 pods\nHPA 2-5, probes"),
    "api": (7.65, 3.0, "FastAPI\n/predict  /health\n/metrics  /model-info"),
    "model": (10.1, 3.0, "sklearn Pipeline\nfeatures -> impute ->\nscale/one-hot -> LR"),
    # lane 4
    "logs": (2.75, 0.7, "JSON request logs\nstdout + rotating file"),
    "prom": (5.2, 0.7, "Prometheus\nscrapes pod /metrics"),
    "grafana": (7.65, 0.7, "Grafana dashboard\nrate, latency, labels"),
}
W, H = 2.15, 1.25

ARROWS = [
    ("uci", "data"), ("data", "eda"), ("eda", "train"), ("train", "mlflow"), ("mlflow", "artifact"),
    ("git", "lint"), ("lint", "test"), ("test", "ci_train"), ("ci_train", "docker"), ("docker", "ghcr"),
    ("client", "ingress"), ("ingress", "pods"), ("pods", "api"), ("api", "model"),
]


def centre(key):
    x, y, _ = BOXES[key]
    return x + W / 2, y + H / 2


def main():
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10.2)
    ax.axis("off")

    for title, y, colour in LANES:
        ax.add_patch(FancyBboxPatch((0.1, y), 14.8, 2.2, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    facecolor=colour, edgecolor="#bbbbbb"))
        ax.text(0.25, y + 2.0, title, fontsize=12, fontweight="bold", va="top", color="#333333")

    for key, (x, y, label) in BOXES.items():
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    facecolor="white", edgecolor="#264653", linewidth=1.4))
        ax.text(x + W / 2, y + H / 2, label, ha="center", va="center", fontsize=9)

    def arrow(a, b, style="-|>", colour="#264653", rad=0.0, ls="-"):
        (x1, y1), (x2, y2) = centre(a), centre(b)
        if abs(y1 - y2) < 0.01:
            x1, x2 = x1 + W / 2, x2 - W / 2
        else:
            y1, y2 = (y1 - H / 2, y2 + H / 2) if y1 > y2 else (y1 + H / 2, y2 - H / 2)
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                                     color=colour, linewidth=1.4, linestyle=ls,
                                     connectionstyle=f"arc3,rad={rad}"))

    for a, b in ARROWS:
        arrow(a, b)
    arrow("ci_train", "train", colour="#e76f51", ls="--")
    arrow("artifact", "docker", colour="#e76f51", ls="--")
    arrow("ghcr", "pods", colour="#e76f51", ls="--")
    arrow("api", "logs", colour="#7b2cbf", ls="--", rad=0.15)
    arrow("api", "prom", colour="#7b2cbf", ls="--")
    arrow("prom", "grafana", colour="#7b2cbf")

    ax.text(7.5, 10.1, "Heart Disease Risk Prediction - MLOps Architecture", ha="center", fontsize=16,
            fontweight="bold")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
