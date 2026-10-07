import sys


def mark_observed_descendants(v, children, evidence_set, observed_desc, seen):
    if v in seen:
        return observed_desc[v]

    seen.add(v)
    if v in evidence_set:
        observed_desc[v] = True

    for child in children[v]:
        if mark_observed_descendants(child, children, evidence_set, observed_desc, seen):
            observed_desc[v] = True

    return observed_desc[v]

def compute_observed_or_descendants(n, children, evidence_set):
    observed_desc = [False] * n
    for v in range(n):
        mark_observed_descendants(v, children, evidence_set, observed_desc, set())
    return observed_desc


def is_collider(a, b, c, parents):
    return a in parents[b] and c in parents[b]


def is_trail_active(path, parents, evidence_set, observed_desc):
    for i in range(1, len(path) - 1):
        a, b, c = path[i - 1], path[i], path[i + 1]

        if is_collider(a, b, c, parents):
            # Collider must be observed or have an observed descendant
            if not observed_desc[b]:
                return False
        else:
            # Non-collider blocks trail if observed
            if b in evidence_set:
                return False

    return True

# visited is a list of booleans indicating whether a node has been visited in the current path i.e. this trail
def search_active_trail(cur, target, adj, parents, evidence_set, observed_desc, visited, path):
    if cur == target:
        return is_trail_active(path, parents, evidence_set, observed_desc)

    for nxt in adj[cur]:
        if not visited[nxt]:
            visited[nxt] = True
            path.append(nxt)

            if search_active_trail(nxt, target, adj, parents, evidence_set, observed_desc, visited, path):
                return True

            path.pop()
            visited[nxt] = False

    return False


def is_d_separated(adj, parents, children, x, y, evidence):
    n = len(adj)
    evidence_set = set(evidence)
    observed_desc = compute_observed_or_descendants(n, children, evidence_set)

    visited = [False] * n
    visited[x] = True

    has_active_trail = search_active_trail(
        cur=x,
        target=y,
        adj=adj,
        parents=parents,
        evidence_set=evidence_set,
        observed_desc=observed_desc,
        visited=visited,
        path=[x],
    )

    return not has_active_trail


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_file>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    p = 0
    n, m = map(int, lines[p].split())
    p += 1

    adj = [[] for _ in range(n)]
    parents = [[] for _ in range(n)]
    children = [[] for _ in range(n)]

    for _ in range(m):
        u, v = map(int, lines[p].split())
        p += 1

        adj[u].append(v)
        adj[v].append(u)
        parents[v].append(u)
        children[u].append(v)

    q = int(lines[p])
    p += 1

    for _ in range(q):
        vals = list(map(int, lines[p].split()))
        p += 1

        x, y = vals[0], vals[1]
        evidence = vals[2:]

        answer = "YES" if is_d_separated(adj, parents, children, x, y, evidence) else "NO"
        print(f"Query: {x} and {y} with evidence {evidence} = {answer}")


if __name__ == "__main__":
    main()