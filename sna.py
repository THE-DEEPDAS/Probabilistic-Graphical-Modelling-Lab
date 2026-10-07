import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ==========================================================
# 1. CREATE SIMULATED SNA GRAPH
# ==========================================================

G_full = nx.watts_strogatz_graph(
    60, 6, 0.15, seed=10
)

# Existing links
edges = list(G_full.edges())

np.random.seed(10)
np.random.shuffle(edges)

# 80 real links -> positive examples
positive = edges[:80]

# Remove these links from graph
G = G_full.copy()
G.remove_edges_from(positive)

# 80 non-existing links -> negative examples
negative = list(nx.non_edges(G))
negative = negative[:80]

pairs = positive + negative

actual = np.array(
    [1] * 80 + [0] * 80
)


# ==========================================================
# 2. SIMILARITY METHODS
# ==========================================================

def jaccard(u, v):

    A = set(G.neighbors(u))
    B = set(G.neighbors(v))

    if len(A | B) == 0:
        return 0

    return len(A & B) / len(A | B)


def adamic_adar(u, v):

    common = set(G.neighbors(u)) & set(G.neighbors(v))

    score = 0

    for node in common:

        degree = G.degree(node)

        if degree > 1:
            score += 1 / np.log(degree)

    return score


def salton(u, v):

    common = len(
        set(G.neighbors(u)) &
        set(G.neighbors(v))
    )

    denominator = np.sqrt(
        G.degree(u) * G.degree(v)
    )

    if denominator == 0:
        return 0

    return common / denominator


def sorensen(u, v):

    common = len(
        set(G.neighbors(u)) &
        set(G.neighbors(v))
    )

    denominator = (
        G.degree(u) +
        G.degree(v)
    )

    if denominator == 0:
        return 0

    return 2 * common / denominator


methods = {
    "Jaccard": jaccard,
    "Adamic-Adar": adamic_adar,
    "Salton": salton,
    "Sorensen": sorensen
}


# ==========================================================
# 3. CALCULATE SIMILARITY SCORES
# ==========================================================

scores = {}

for name, function in methods.items():

    scores[name] = np.array([
        function(u, v)
        for u, v in pairs
    ])


# ==========================================================
# 4. 10,000 THRESHOLD SEARCH
# ==========================================================

STEPS = 10000

results = {}

for name in methods:

    score = scores[name]

    # Thresholds based on actual score range
    thresholds = np.linspace(
        score.min(),
        score.max(),
        STEPS
    )

    accuracy_history = []

    best_accuracy = 0
    best_threshold = 0
    best_step = 0

    for step, threshold in enumerate(
        thresholds, start=1
    ):

        prediction = (
            score >= threshold
        ).astype(int)

        accuracy = np.mean(
            prediction == actual
        )

        accuracy_history.append(
            accuracy
        )

        # First time maximum accuracy is reached
        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_threshold = threshold
            best_step = step

    results[name] = {
        "thresholds": thresholds,
        "accuracy": np.array(accuracy_history),
        "best_accuracy": best_accuracy,
        "best_threshold": best_threshold,
        "best_step": best_step
    }


# ==========================================================
# 5. PRINT RESULTS
# ==========================================================

print("\n" + "=" * 70)
print("10,000-STEP OPTIMAL THRESHOLD COMPARISON")
print("=" * 70)

for name, result in results.items():

    print(
        f"{name:15s}"
        f" Best Step = {result['best_step']:5d}"
        f" / {STEPS}"
        f"   Threshold = {result['best_threshold']:.4f}"
        f"   Accuracy = {result['best_accuracy']:.4f}"
    )


# ==========================================================
# 6. FIND METHOD THAT REACHES OPTIMUM FASTEST
# ==========================================================

best_method = min(
    results,
    key=lambda x: results[x]["best_step"]
)

print("\n" + "=" * 70)
print("FASTEST METHOD")
print("=" * 70)

print("Method :", best_method)
print(
    "Optimal step :",
    results[best_method]["best_step"]
)
print(
    "Threshold :",
    results[best_method]["best_threshold"]
)
print(
    "Accuracy :",
    results[best_method]["best_accuracy"]
)


# ==========================================================
# 7. GRAPH: ACCURACY VS THRESHOLD SEARCH STEP
# ==========================================================

plt.figure(figsize=(10, 6))

for name, result in results.items():

    plt.plot(
        range(1, STEPS + 1),
        result["accuracy"],
        label=name
    )

    # Mark optimal point
    plt.scatter(
        result["best_step"],
        result["best_accuracy"],
        s=50
    )

plt.xlabel("Threshold Search Step")
plt.ylabel("Accuracy")

plt.title(
    "Optimal Threshold Search: 10,000 Steps"
)

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# ==========================================================
# 8. ROC-AUC FOR BEST METHOD
# ==========================================================

best_scores = scores[best_method]

fpr, tpr, _ = roc_curve(
    actual,
    best_scores
)

roc_auc = auc(
    fpr,
    tpr
)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=f"{best_method} (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    "--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    f"ROC-AUC Curve - {best_method}"
)

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()