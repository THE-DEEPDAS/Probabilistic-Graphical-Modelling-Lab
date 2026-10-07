"""
AI453 - Practical #5: A Bayesian Network from Real Data
Department of Artificial Intelligence, SVNIT Surat

Plain Python 3, standard library only. Nothing to install.
Run:  python3 lab5_bn.py
It should print one line: P(fail=1 | medu=1) = 0.099

Data: student-por.csv, 649 secondary-school students from two Portuguese
schools (Cortez & Silva, 2008; UCI Machine Learning Repository).
Keep it in the same folder as this file.
"""

import csv
import itertools
import random

# --------------------------------------------------------------------------
# The eight variables, all binary, all cut from real columns of the CSV.
# --------------------------------------------------------------------------
#   medu    mother's education is secondary or above   (Medu >= 3)
#   goout   goes out with friends often                (goout >= 4)
#   fail    has failed a class before                  (failures >= 1)
#   higher  wants to go on to higher education         (higher == yes)
#   study   studies more than 5 hours a week           (studytime >= 3)
#   absent  six or more absences this year             (absences >= 6)
#   g1      first-period grade of 12 or more           (G1 >= 12)
#   grade   final grade of 12 or more                  (G3 >= 12)

VARS = ["medu", "goout", "fail", "higher", "study", "absent", "g1", "grade"]

# --------------------------------------------------------------------------
# The network. Each entry is  child : [its parents].
# The ordering is temporal: family background, then things fixed early in the
# year, then the first-period grade, then the final grade.
# --------------------------------------------------------------------------
PARENTS = {
    "medu":   [],
    "goout":  [],
    "fail":   ["medu"],
    "higher": ["medu", "fail"],
    "study":  ["medu", "higher", "goout"],
    "absent": ["goout", "fail"],
    "g1":     ["fail", "study", "goout", "higher"],
    "grade":  ["g1", "absent"],
}

# VARS is already a topological order: every variable appears after all of its
# parents. Check that for yourself before you rely on it in Part D.


# --------------------------------------------------------------------------
# Helpers, written for you
# --------------------------------------------------------------------------

def load(path="student-por.csv"):
    """Read the CSV and cut every column down to 0/1. Returns a list of dicts."""
    rows = []
    for r in csv.DictReader(open(path)):
        g = lambda k: r[k].strip('"')
        rows.append({
            "medu":   1 if int(g("Medu")) >= 3 else 0,
            "goout":  1 if int(g("goout")) >= 4 else 0,
            "fail":   1 if int(g("failures")) >= 1 else 0,
            "higher": 1 if g("higher") == "yes" else 0,
            "study":  1 if int(g("studytime")) >= 3 else 0,
            "absent": 1 if int(g("absences")) >= 6 else 0,
            "g1":     1 if int(g("G1")) >= 12 else 0,
            "grade":  1 if int(g("G3")) >= 12 else 0,
        })
    return rows


def count(data, cond):
    """How many rows satisfy every key=value pair in cond. count(data, {}) is len(data)."""
    return sum(1 for d in data if all(d[k] == v for k, v in cond.items()))


def configs(parents):
    """All parent assignments, as tuples, in a fixed order. configs([]) gives [()]."""
    return list(itertools.product([0, 1], repeat=len(parents)))


def sample_bernoulli(p):
    """From Practical 3. Still the only random primitive you need."""
    return 1 if random.random() < p else 0


# --------------------------------------------------------------------------
# The worked example: one CPT row, straight from counting.
#
#   P(fail = 1 | medu = 1) = (rows with fail=1 and medu=1) / (rows with medu=1)
#
# Every CPT in this lab is this line with a different variable and a different
# parent assignment. That is all maximum likelihood estimation is here --
# exactly the counting you did for the die in Practical 3, T8.
# --------------------------------------------------------------------------

if __name__ == "__main__":
    data = load()
    num = count(data, {"fail": 1, "medu": 1})
    den = count(data, {"medu": 1})
    print("P(fail=1 | medu=1) =", round(num / den, 3))


# ==========================================================================
# PART A - fitting the network
# ==========================================================================

# T1. Print the number of rows, and P(v=1) for each of the eight variables.
#     Sanity check: you should have 649 rows and P(grade=1) near 0.536.

if __name__ == "__main__":
    print("\nT1")
    print("Number of rows:", len(data))

    for v in VARS:
        p = count(data, {v: 1}) / len(data)
        print(f"P({v}=1) = {p:.3f}")


# T2. fit(data, alpha=0.0) -> a dict cpt[v][parent_tuple] = P(v = 1 | parents).
#     For each variable v and each assignment to its parents, count as in the
#     worked example. alpha is a smoothing constant; leave it at 0 for now and
#     come back to it in T7.
#         P(v=1 | pa) = (n1 + alpha) / (n0 + n1 + 2*alpha)
#     Print the CPT of 'grade'. Four rows, one per (g1, absent) pair.

def fit(data, alpha=0.0):
    cpt = {}

    for v in VARS:
        cpt[v] = {}

        parents = PARENTS[v]

        for parent_values in configs(parents):

            cond = {}

            for i in range(len(parents)):
                cond[parents[i]] = parent_values[i]

            n1 = count(data, {**cond, v: 1})
            n0 = count(data, {**cond, v: 0})

            denominator = n0 + n1 + 2 * alpha

            if denominator == 0:
                cpt[v][parent_values] = 0.0
            else:
                cpt[v][parent_values] = (
                    (n1 + alpha) / denominator
                )

    return cpt


if __name__ == "__main__":
    print("\nT2")

    cpt = fit(data)

    print("CPT of grade:")

    for parent_values in configs(PARENTS["grade"]):
        print(
            parent_values,
            "->",
            round(cpt["grade"][parent_values], 3)
        )


# ==========================================================================
# PART B - the graph is a claim about the distribution
# ==========================================================================

# T3. joint(assignment, cpt) -> the probability of one full assignment to all
#     eight variables, as the product over variables of P(v | its parents).
#     That product is the whole content of the graph.
#     Then loop over all 2^8 = 256 assignments and add up the results.
#     If your factorisation is right the total is 1. If it is not, you have a
#     CPT row that does not sum to 1, or a parent lookup in the wrong order.

def joint(assignment, cpt):
    probability = 1.0

    for v in VARS:

        parents = PARENTS[v]

        parent_values = tuple(
            assignment[p]
            for p in parents
        )

        p = cpt[v][parent_values]

        if assignment[v] == 1:
            probability *= p
        else:
            probability *= (1 - p)

    return probability


if __name__ == "__main__":
    print("\nT3")

    total = 0.0

    for values in itertools.product([0, 1], repeat=len(VARS)):

        assignment = dict(zip(VARS, values))

        total += joint(assignment, cpt)

    print("Sum of all 256 joint probabilities =", total)


# T4. infer(query, evidence, cpt) -> P(query | evidence), by enumeration:
#     sum joint() over every assignment consistent with the evidence, keep a
#     second running total for those that also match the query, divide.
#     This is the slow, honest, exponential way. It is the ground truth every
#     faster algorithm later in the course gets checked against.
#
#     Print, side by side, the model's answer and the raw fraction in the data:
#         P(grade=1)                 P(grade=1 | g1=1)
#         P(grade=1 | fail=0)        P(grade=1 | g1=1, absent=1)
#     Also print how many students the data fraction is based on. Where the
#     model and the data disagree, note whether the count is small.

def infer(query, evidence, cpt):
    numerator = 0.0
    denominator = 0.0

    for values in itertools.product([0, 1], repeat=len(VARS)):

        assignment = dict(zip(VARS, values))

        evidence_match = True

        for v in evidence:
            if assignment[v] != evidence[v]:
                evidence_match = False
                break

        if not evidence_match:
            continue

        p = joint(assignment, cpt)

        denominator += p

        query_match = True

        for v in query:
            if assignment[v] != query[v]:
                query_match = False
                break

        if query_match:
            numerator += p

    if denominator == 0:
        return 0.0

    return numerator / denominator


if __name__ == "__main__":
    print("\nT4")

    tests = [
        ({"grade": 1}, {}),
        ({"grade": 1}, {"g1": 1}),
        ({"grade": 1}, {"fail": 0}),
        ({"grade": 1}, {"g1": 1, "absent": 1})
    ]

    for query, evidence in tests:

        model = infer(query, evidence, cpt)

        cond = dict(evidence)
        cond.update(query)

        num = count(data, cond)
        den = count(data, evidence)

        raw = num / den if den != 0 else 0

        print(
            f"query={query}, evidence={evidence}"
        )
        print(
            f"  model = {model:.3f}, "
            f"data = {raw:.3f}, "
            f"n = {den}"
        )


# ==========================================================================
# PART C - checking what the graph predicts
# ==========================================================================

# T5. Read the graph, not the data. Three pairs have no edge and no common
#     ancestor, so the graph claims they are independent outright:
#         medu and goout,   goout and fail,   goout and higher
#     For each, print P(a=1, b=1) against P(a=1) P(b=1), the way you did in
#     Practical 2, T2. One line: does the data back the graph up?

if __name__ == "__main__":
    print("\nT5")

    pairs = [
        ("medu", "goout"),
        ("goout", "fail"),
        ("goout", "higher")
    ]

    for a, b in pairs:

        # Model joint probability
        model_joint = infer(
            {a: 1, b: 1},
            {},
            cpt
        )

        # Model marginal probabilities
        model_a = infer({a: 1}, {}, cpt)
        model_b = infer({b: 1}, {}, cpt)

        model_product = model_a * model_b

        # Data probabilities
        data_joint = count(
            data,
            {a: 1, b: 1}
        ) / len(data)

        data_a = count(
            data,
            {a: 1}
        ) / len(data)

        data_b = count(
            data,
            {b: 1}
        ) / len(data)

        data_product = data_a * data_b

        print(
            f"{a}, {b}: "
            f"model joint={model_joint:.3f}, "
            f"model product={model_product:.3f}, "
            f"data joint={data_joint:.3f}, "
            f"data product={data_product:.3f}"
        )


# T6. Now the same thing with a collider in the way. The graph says medu and
#     goout are independent -- but they are both parents of g1, so learning
#     g1 should make them dependent. Print
#         P(medu=1)      P(medu=1 | g1=0)
#         P(medu=1 | g1=0, goout=0)      P(medu=1 | g1=0, goout=1)
#     Two lines in a comment: among students who scored poorly, why does
#     learning that they go out a lot make an educated mother MORE likely?
#     This is the sprinkler from Practical 2, T7, in a real school.

if __name__ == "__main__":
    print("\nT6")

    p1 = infer(
        {"medu": 1},
        {},
        cpt
    )

    p2 = infer(
        {"medu": 1},
        {"g1": 0},
        cpt
    )

    p3 = infer(
        {"medu": 1},
        {"g1": 0, "goout": 0},
        cpt
    )

    p4 = infer(
        {"medu": 1},
        {"g1": 0, "goout": 1},
        cpt
    )

    print("P(medu=1) =", round(p1, 3))
    print("P(medu=1 | g1=0) =", round(p2, 3))
    print(
        "P(medu=1 | g1=0, goout=0) =",
        round(p3, 3)
    )
    print(
        "P(medu=1 | g1=0, goout=1) =",
        round(p4, 3)
    )

    # Comment:
    # g1 is a collider because both medu and goout point into g1.
    # Conditioning on g1 opens the path between medu and goout.
    # Therefore, among students with g1=0, knowing that goout=1
    # changes our belief about medu.


# ==========================================================================
# PART D - sampling, smoothing, and where the model breaks
# ==========================================================================

# T7. Run fit() with alpha=0 and look for CPT rows that are 0, 1, or 0/0.
#     Print them. There are eight, all in g1 -- which has four parents, so 16
#     rows to fill from 649 students, and one parent combination that no
#     student in the file matches at all.
#     This is Practical 3, T8 again: the die fitted on 30 rolls gave face 6 a
#     probability of exactly 0. Refit with alpha=1 and confirm they are gone.
#     One line: what does a CPT row of exactly 0 do to any query that needs it?

if __name__ == "__main__":
    print("\nT7")

    cpt0 = fit(data, alpha=0.0)

    print("Problematic CPT rows with alpha=0:")

    for v in VARS:

        parents = PARENTS[v]

        for parent_values in configs(parents):

            cond = {}

            for i in range(len(parents)):
                cond[parents[i]] = parent_values[i]

            n1 = count(data, {**cond, v: 1})
            n0 = count(data, {**cond, v: 0})

            # parent only never observed 
            if n0 + n1 == 0:
                print(
                    v,
                    parent_values,
                    "-> 0/0"
                )

            # We observed this parent combination, but never observed the child variable equal to 1
            elif n1 == 0:
                print(
                    v,
                    parent_values,
                    "-> 0"
                )
            # We observed this parent combination, but never observed the child equal to 0.
            elif n0 == 0:
                print(
                    v,
                    parent_values,
                    "-> 1"
                )

    cpt1 = fit(data, alpha=1.0)

    print("\nAfter alpha=1:")

    problem = False

    for v in VARS:
        for parent_values in configs(PARENTS[v]):

            p = cpt1[v][parent_values]

            if p == 0 or p == 1:
                problem = True
                print(
                    v,
                    parent_values,
                    "->",
                    p
                )

    if not problem:
        print("No CPT row is exactly 0 or 1.")

    # A CPT row of exactly 0 makes every joint assignment using
    # that row have probability 0, so any query requiring that
    # event can become impossible under the model.


# T8. forward(cpt) -> one simulated student. Walk VARS in order; each variable
#     is a sample_bernoulli() (Practical 3) with p read off its CPT row, using
#     the values you already drew for its parents. That is the only reason the
#     topological order matters.
#     Then rejection sampling: draw M students, throw away the ones that do not
#     match the evidence, and take the fraction of the survivors that match the
#     query. Estimate P(grade=1 | fail=1) with M = 2000 and M = 50000 and
#     compare with your exact answer from T4.
#     Print the percentage of samples you kept. One line: what happens to that
#     percentage as the evidence gets more specific, and why is that a problem?

def forward(cpt):
    student = {}

    for v in VARS:

        parents = PARENTS[v]

        parent_values = tuple(
            student[p]
            for p in parents
        )

        p = cpt[v][parent_values]

        student[v] = sample_bernoulli(p)

    return student


if __name__ == "__main__":
    print("\nT8")

    query = {"grade": 1}
    evidence = {"fail": 1}

    exact = infer(
        query,
        evidence,
        cpt1
    )

    print(
        "Exact P(grade=1 | fail=1) =",
        round(exact, 3)
    )

    for M in [2000, 50000]:

        kept = 0
        query_match = 0

        for _ in range(M):

            student = forward(cpt1)

            evidence_match = True

            for v in evidence:
                if student[v] != evidence[v]:
                    evidence_match = False
                    break

            if not evidence_match:
                continue

            kept += 1

            query_matches = True

            for v in query:
                if student[v] != query[v]:
                    query_matches = False
                    break

            if query_matches:
                query_match += 1

        estimate = query_match / kept if kept != 0 else 0
        percentage = (kept / M) * 100

        print(
            f"M={M}: "
            f"estimate={estimate:.3f}, "
            f"kept={kept}, "
            f"kept={percentage:.2f}%"
        )

    # As the evidence becomes more specific, fewer generated samples
    # satisfy it. Therefore more samples are rejected, making rejection
    # sampling inefficient.


# T9. The model against reality. Print the model's P(grade=1 | fail=1) beside
#     the fraction in the data. They are far apart, and re-running T7 with a
#     smaller alpha does not close the gap -- so this is not a smoothing
#     problem. Look at the graph and find every path from fail to grade.
#     Two or three lines: what edge would you add, and what does that tell you
#     about where the DAG came from in the first place?

if __name__ == "__main__":
    print("\nT9")

    model_probability = infer(
        {"grade": 1},
        {"fail": 1},
        cpt1
    )

    data_num = count(
        data,
        {"grade": 1, "fail": 1}
    )

    data_den = count(
        data,
        {"fail": 1}
    )

    data_probability = data_num / data_den

    print(
        "Model P(grade=1 | fail=1) =",
        round(model_probability, 3)
    )

    print(
        "Data  P(grade=1 | fail=1) =",
        round(data_probability, 3)
    )

    # The graph contains the path:
    #     fail -> g1 -> grade
    #
    # and other indirect paths through higher/study/absent.
    # A possible missing edge is:
    #
    #     fail -> grade
    #
    # The large difference suggests that the DAG was constructed from
    # domain knowledge and temporal assumptions rather than learned
    # completely from the data. Therefore, some important dependency
    # may have been omitted.