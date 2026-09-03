import sys


def is_d_separated(adj, parents, children, x, y, evidence):

    evidence = set(evidence)
    
    observed_desc = [False] * len(adj)

    def dfs_desc(v, seen):
        if v in seen:
            return observed_desc[v]

        # observed_desc[v] = True if v itself is observed
        seen.add(v)
        observed_desc[v] = v in evidence

        # or any descendant of v is observed
        for child in children[v]:
            if dfs_desc(child, seen):
                observed_desc[v] = True

        return observed_desc[v]

    for v in range(len(adj)):
        dfs_desc(v, set())

    # a -> b <- c
    def is_collider(a, b, c):
        return a in parents[b] and c in parents[b]

    visited = [False] * len(adj)
    visited[x] = True

    def find_active_trail(cur, path):

        # We reached y, so check whether this trail is active
        if cur == y:

            for i in range(1, len(path) - 1):
                a, b, c = path[i - 1], path[i], path[i + 1]

                if is_collider(a, b, c):
                    # Collider must be observed or have
                    # an observed descendant
                    if not observed_desc[b]:
                        return False

                else:
                    # Non-collider blocks the trail if observed
                    if b in evidence:
                        return False

            return True

        # Try every possible trail from current node
        for nxt in adj[cur]:

            if visited[nxt]:
                continue

            visited[nxt] = True
            path.append(nxt)

            if find_active_trail(nxt, path):
                return True

            path.pop()
            visited[nxt] = False

        return False

    # One active trail means X and Y are NOT d-separated
    return not find_active_trail(x, [x])


# main function code
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

