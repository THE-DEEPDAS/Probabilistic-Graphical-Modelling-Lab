"""
AI453 - Lab 6: Bayes-Ball and the Markov Blanket
Department of Artificial Intelligence, SVNIT Surat

Standard library only. networkx and pgmpy are not permitted.

Two modes, both reading the same file format as Lab 4:

    python3 lab6.py case_p6_main.in              -> one YES/NO line per query
    python3 lab6.py case_p6_main.in --blanket    -> one line per node, its blanket

Lab 4 answered d-separation by listing every trail and checking each one.
This time you answer the same question without ever building a trail, by
walking the graph once. On case_p6_sample.in the two programs must agree
line for line -- that is the point of the exercise.
"""

import sys


# --------------------------------------------------------------------------
# Input format, unchanged from Lab 4
# --------------------------------------------------------------------------
#   n m                     n nodes numbered 0..n-1, m edges
#   u v                     m lines, one per directed edge u -> v
#   q                       number of queries
#   x y z1 z2 ... zk        q lines; the pair, then the evidence set Z
#
# In every query x, y and the nodes of Z are distinct.
# --------------------------------------------------------------------------

def read_graph(path):
    """Returns n, parents, children.  parents[v] and children[v] are lists."""
    with open(path) as f:
        lines = [ln.split() for ln in f if ln.strip() != ""]
    n, m = int(lines[0][0]), int(lines[0][1])
    parents = {v: [] for v in range(n)}
    children = {v: [] for v in range(n)}
    for u, v in ((int(a), int(b)) for a, b in lines[1:1 + m]):
        parents[v].append(u)
        children[u].append(v)
    q = int(lines[1 + m][0])
    queries = []
    for ln in lines[2 + m: 2 + m + q]:
        nums = [int(t) for t in ln]
        queries.append((nums[0], nums[1], set(nums[2:])))
    return n, parents, children, queries


# --------------------------------------------------------------------------
# The worked example: the ancestors of the evidence set.
#
# You need this set before you can walk anything. A collider only lets the
# ball through when the collider itself, OR one of its descendants, is in Z.
# Turned around: the collider must be an ancestor of Z, or in Z.
# One upward sweep from Z gives you every such node at once.
# --------------------------------------------------------------------------

def ancestors_of(parents, Z):
    seen = set()
    stack = list(Z)
    while stack:
        v = stack.pop()
        if v not in seen:
            seen.add(v)
            stack.extend(parents[v])
    return seen


# ==========================================================================
# PART A - Bayes-Ball
# ==========================================================================

# T1. reachable(parents, children, x, Z) -> the set of nodes the ball can
#     reach from x given evidence Z.
#
#     The state you search over is not a node. It is a PAIR: a node, and the
#     direction you arrived from. Write 'up' for a ball that arrived from a
#     child (moving against the arrows) and 'down' for one that arrived from
#     a parent. The same node behaves differently in the two cases, which is
#     exactly why d-separation is not ordinary graph connectivity.
#
#     At (v, up), with v not in Z:      pass to every parent as 'up',
#                                       and to every child as 'down'.
#     At (v, down), with v not in Z:    pass to every child as 'down'.
#     At (v, down), with v in ancestors(Z) or in Z:
#                                       pass to every parent as 'up'.
#                                       (v is a collider on this trail; the
#                                        evidence opens it)
#
#     Start from (x, up). Keep a visited set of PAIRS, not of nodes, or you
#     will lose the trails that need to enter a node twice.
#
#     Collect every node you visit that is not itself in Z. That is the set
#     of nodes NOT d-separated from x by Z.

# T2. Answer the queries: x and y are d-separated by Z exactly when y is not
#     in reachable(x, Z). Print YES or NO, one per line, in the input order.
#     Capitals, nothing else on the line, no blank lines.
#
#     Run on case_p6_sample.in. It is the same file as Lab 4's sample, so
#         diff lab4_output.txt lab6_output_sample.txt
#     must print nothing at all. If it does not, one of your two programs is
#     wrong and you now have to work out which.

# T3. Count the work. Add a counter for the number of (node, direction) pairs
#     your search pops. Print it to stderr, not stdout, so the graded output
#     stays clean. Then run on case_p6_wide.in: 91 nodes, and just over a
#     billion directed paths between the two endpoints.
#
#     Run your Lab 4 program on the same file. Do not wait for it.
#     One line in a comment: what is the bound on the number of pairs your
#     search can ever pop, in terms of n and m, and why does the count of
#     trails not appear in it?


# ==========================================================================
# PART B - the Markov blanket
# ==========================================================================

# T4. blanket(parents, children, v) -> the Markov blanket of v: its parents,
#     its children, and the other parents of its children. Sorted, ascending,
#     no duplicates, and v itself is never in it.
#     Under --blanket print n lines, line i being the blanket of node i as
#     space-separated indices. An empty blanket prints an empty line.

# T5. Prove it to yourself with T1. For every node v and every node w that is
#     neither v nor in blanket(v), check that your reachability code says v
#     and w are d-separated by blanket(v). Run it over all four case files.
#     Print only the failures. There should be none.
#     One line: what does the blanket buy you that the full parent set does
#     not? Why is the "other parents of its children" part not optional --
#     which case file query shows what goes wrong if you drop it?


# ==========================================================================
# PART C - the blanket on real data
# ==========================================================================
# Needs lab5_bn.py and student-por.csv from Practical 5, in this folder.

# T6. case_p6_student.in is the network you fitted in Practical 5, with
#         0 medu   1 goout   2 fail    3 higher
#         4 study  5 absent  6 g1      7 grade
#     Print the blanket of every node, in names rather than indices.
#     Which node has the smallest blanket, and does that match your intuition
#     about the network from Practical 5?

# T7. Check the blanket numerically, not just structurally. Using the CPTs you
#     fitted in Practical 5 and your infer() from that lab, print
#         P(g1=1 | blanket(g1) all set to 1)
#     and then the same query with medu=1 added to the evidence. medu is not
#     in the blanket of g1, so the two numbers must agree to the last decimal
#     place. Repeat for two more nodes of your choosing.
#     One line: you have just computed the quantity Gibbs sampling needs at
#     every step. Given the blanket, how much of the network do you have to
#     look at to resample one variable?


def reachable(parents, children, x, Z):
    """Return the nodes reachable from x by Bayes-ball."""
    Z = set(Z)
    ancestors = ancestors_of(parents, Z)
    stack = [(x, "up")]
    visited = set()
    reached = set()

    while stack:
        v, direction = stack.pop()
        state = (v, direction)
        if state in visited:
            continue
        visited.add(state)

        if v not in Z:
            reached.add(v)

        if direction == "up" and v not in Z:
            stack.extend((p, "up") for p in parents[v])
            stack.extend((c, "down") for c in children[v])
        elif direction == "down" and (v in Z or v in ancestors):
            stack.extend((p, "up") for p in parents[v])

    return reached


# T2. Answer the queries: x and y are d-separated by Z exactly when y is not
# in reachable(x, Z). Print YES or NO, one per line, in the input order.
def answer_queries(queries, parents, children):
    for x, y, evidence in queries:
        print("YES" if y not in reachable(parents, children, x, evidence) else "NO")


# T3. Count the work. Add a counter for the number of (node, direction) pairs
def reachable_with_count(parents, children, x, Z):
    Z = set(Z)
    ancestors = ancestors_of(parents, Z)
    stack = [(x, "up")]
    visited = set()
    reached = set()
    popped = 0

    while stack:
        v, direction = stack.pop()
        state = (v, direction)
        if state in visited:
            continue
        visited.add(state)
        popped += 1
        if v not in Z:
            reached.add(v)
        if direction == "up" and v not in Z:
            stack.extend((p, "up") for p in parents[v])
            stack.extend((c, "down") for c in children[v])
        elif direction == "down" and (v in Z or v in ancestors):
            stack.extend((p, "up") for p in parents[v])
    return reached, popped

# The search can pop at most 2n pairs: every node has only 'up' and 'down'
# states, so the number of trails does not affect the bound.


# T4. blanket(parents, children, v) -> the Markov blanket of v: its parents,
def blanket(parents, children, v):
    result = set(parents[v]) | set(children[v])
    for child in children[v]:
        result.update(parents[child])
    result.discard(v)
    return sorted(result)


# T5. Prove it to yourself with T1. For every node v and every node w that is
def check_blankets(n, parents, children):
    failures = []
    for v in range(n):
        Z = set(blanket(parents, children, v))
        for w in range(n):
            if w != v and w not in Z and w in reachable(parents, children, v, Z):
                failures.append((v, w))
    return failures


# T6. case_p6_student.in is the network you fitted in Practical 5, with
def named_blankets(n, parents, children):
    names = ["medu", "goout", "fail", "higher", "study", "absent", "g1", "grade"]
    for v in range(n):
        print(names[v] + ": " + " ".join(names[x] for x in blanket(parents, children, v)))


# T7. Check the blanket numerically, not just structurally. Using the CPTs you
def check_blanket_numerically(parents, children):
    sys.path.insert(0, r"D:\PGM Lab\lab5")
    import lab5_bn

    data = lab5_bn.load(r"D:\PGM Lab\lab5\student-por.csv")
    cpt = lab5_bn.fit(data, alpha=1.0)
    names = ["medu", "goout", "fail", "higher", "study", "absent", "g1", "grade"]
    for variable in ("g1", "grade", "fail"):
        v = names.index(variable)
        evidence = {names[node]: 1 for node in blanket(parents, children, v)}
        result = lab5_bn.infer({variable: 1}, evidence, cpt)

        outside = next(name for name in names if name not in evidence and name != variable)
        with_outside = dict(evidence)
        with_outside[outside] = 1
        result_with_outside = lab5_bn.infer(
            {variable: 1}, with_outside, cpt
        )

        print(
            f"P({variable}=1 | blanket={evidence}) = {result:.6f}"
        )
        print(
            f"P({variable}=1 | blanket + {outside}=1) = "
            f"{result_with_outside:.6f}"
        )


def main():
    base = "D:\\PGM Lab\\lab6\\"
    sample_n, sample_parents, sample_children, sample_queries = read_graph(
        base + "case_p6_sample.in"
    )

    print("T1 - ancestors of {3}:", sorted(ancestors_of(sample_parents, {3})))

    n, parents, children, queries = read_graph(base + "case_p6_main.in")
    print("T2")
    answer_queries(queries, parents, children)

    print("\nT3")
    wide_n, wide_parents, wide_children, wide_queries = read_graph(
        base + "case_p6_wide.in"
    )
    for x, y, evidence in wide_queries:
        _, popped = reachable_with_count(wide_parents, wide_children, x, evidence)
        print(
            f"wide query {x} -> {y}: {popped} state pairs popped",
            file=sys.stderr,
        )
    print("At most 2n state pairs can be popped, because each node has two directions.")

    print("\nT4")
    for v in range(n):
        print(f"node {v}:", " ".join(map(str, blanket(parents, children, v))))

    print("\nT5")
    for filename in ("case_p6_main.in", "case_p6_sample.in", "case_p6_student.in", "case_p6_wide.in"):
        file_n, file_parents, file_children, _ = read_graph(base + filename)
        failures = check_blankets(file_n, file_parents, file_children)
        if failures:
            print(filename, failures)
        else:
            print(filename + ": no failures")

    print("\nT6")
    student_n, student_parents, student_children, _ = read_graph(base + "case_p6_student.in")
    named_blankets(student_n, student_parents, student_children)

    print("\nT7")
    check_blanket_numerically(student_parents, student_children)


if __name__ == "__main__":
    main()
