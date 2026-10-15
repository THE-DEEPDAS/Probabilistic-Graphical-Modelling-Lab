"""hidden markov model: the occasionally dishonest casino."""

import itertools
import math
import random


states = ["F", "L"]
initial = [0.5, 0.5]
transition = [[0.95, 0.05], [0.10, 0.90]]
emission = [[1 / 6] * 6, [0.10, 0.10, 0.10, 0.10, 0.10, 0.50]]
observations = [6, 1, 5, 2, 4, 1, 3, 6, 4, 2, 1, 2, 6, 4, 1,
                1, 6, 2, 5, 6, 5, 6, 2, 6, 6, 6, 6, 6, 5, 6]


def logsum(values):
    values = list(values)
    biggest = max(values)
    if biggest == -math.inf:
        return -math.inf
    return biggest + math.log(sum(math.exp(x - biggest) for x in values))


def generate(length, seed=7):
    random.seed(seed)
    path, rolls = [], []
    state = 0 if random.random() < initial[0] else 1
    for _ in range(length):
        path.append(state)
        r = random.random()
        total = 0
        for roll, probability in enumerate(emission[state], 1):
            total += probability
            if r < total:
                rolls.append(roll)
                break
        r = random.random()
        state = 0 if r < transition[state][0] else 1
    return rolls, path


def forward(seq):
    # alpha[t][s] is the joint probability of observations 0..t and state s at t.
    # initialise with the first emission, recurse by summing predecessors, and
    # terminate by summing the last row.
    alpha = []
    first = [initial[s] * emission[s][seq[0] - 1] for s in range(2)]
    alpha.append(first)
    for roll in seq[1:]:
        alpha.append([emission[s][roll - 1] * sum(
            alpha[-1][old] * transition[old][s] for old in range(2))
            for s in range(2)])
    return alpha, sum(alpha[-1])


def brute_force(seq):
    # direct summation checks the forward result by trying every state path.
    total = 0
    for path in itertools.product(range(2), repeat=len(seq)):
        value = initial[path[0]] * emission[path[0]][seq[0] - 1]
        for t in range(1, len(seq)):
            value *= transition[path[t - 1]][path[t]]
            value *= emission[path[t]][seq[t] - 1]
        total += value
    return total


def log_forward(seq):
    # log-space avoids underflow by replacing products with sums and sums with logsumexp.
    alpha = [[math.log(initial[s]) + math.log(emission[s][seq[0] - 1])
              for s in range(2)]]
    for roll in seq[1:]:
        alpha.append([math.log(emission[s][roll - 1]) + logsum(
            alpha[-1][old] + math.log(transition[old][s]) for old in range(2))
            for s in range(2)])
    return alpha, logsum(alpha[-1])


def scaled_forward(seq):
    # scaling keeps each forward vector near one and stores the removed scale factors.
    values = [initial[s] * emission[s][seq[0] - 1] for s in range(2)]
    log_probability = 0
    scale = sum(values)
    values = [x / scale for x in values]
    log_probability += math.log(scale)
    for roll in seq[1:]:
        values = [emission[s][roll - 1] * sum(
            values[old] * transition[old][s] for old in range(2))
            for s in range(2)]
        scale = sum(values)
        values = [x / scale for x in values]
        log_probability += math.log(scale)
    return log_probability


def backward(seq):
    # beta[t][s] is the probability of observations after t given state s at t.
    # initialise the last row to one, then recurse from the end to the start.
    beta = [[1.0, 1.0] for _ in seq]
    for t in range(len(seq) - 2, -1, -1):
        beta[t] = [sum(transition[s][n] * emission[n][seq[t + 1] - 1] * beta[t + 1][n]
                       for n in range(2)) for s in range(2)]
    probability = sum(initial[s] * emission[s][seq[0] - 1] * beta[0][s]
                       for s in range(2))
    return beta, probability


def viterbi(seq):
    # viterbi replaces the forward sum by a maximum and stores the best predecessor.
    # the back-pointers recover one complete maximum-probability path.
    # viterbi replaces the forward sum by a maximum and stores the best predecessor.
    score = [[math.log(initial[s]) + math.log(emission[s][seq[0] - 1])
              for s in range(2)]]
    back = []
    for roll in seq[1:]:
        previous = []
        row = []
        for s in range(2):
            choices = [score[-1][old] + math.log(transition[old][s]) for old in range(2)]
            previous.append(max(range(2), key=lambda old: choices[old]))
            row.append(math.log(emission[s][roll - 1]) + max(choices))
        back.append(previous)
        score.append(row)
    path = [max(range(2), key=lambda s: score[-1][s])]
    for previous in reversed(back):
        path.append(previous[path[-1]])
    return list(reversed(path)), max(score[-1])


def report():
    # task 1b: state t separates the earlier states and observations from later ones.
    print("\n1. model and generation")
    print("-" * 30)
    print("row sums:", [sum(initial), *[sum(row) for row in transition],
                              *[sum(row) for row in emission]])
    print("conditional independence: state t separates earlier states and observations from later ones")

    rolls, path = generate(300)
    stationary_loaded = transition[0][1] / (transition[0][1] + transition[1][0])
    print(f"loaded proportion:             {sum(path) / 300:.4f}")
    print(f"stationary loaded probability: {stationary_loaded:.4f}")
    print(f"sixes in supplied sequence:    {observations.count(6)}")
    print("sixes expected for fair die:   5")
    # task 1d: the count suggests a loaded period, but it cannot locate the period
    # because counting removes the order and position of the rolls.

    print("\n2. forward algorithm")
    print("-" * 30)
    alpha, probability = forward(observations)
    print(f"probability of observations: {probability:.12e}")
    short = observations[:5]
    print(f"forward, length 5:           {forward(short)[1]:.16f}")
    print(f"brute force, length 5:        {brute_force(short):.16f}")
    # task 2e and 2h: the forward algorithm reuses previous work and costs O(T*S^2),
    # while direct enumeration costs O(S^T).
    print("\noperation counts")
    print("length        brute force        forward")
    for length in (5, 20, 300):
        print(f"{length:<14}{2 ** length:<20,}{length * 4}")

    print("\n3. numerical underflow")
    print("-" * 30)
    print("length        smallest alpha       probability")
    for length in (50, 100, 200, 300, 400, 500, 800):
        seq, _ = generate(length, length)
        values, probability = forward(seq)
        print(f"{length:<14}{min(min(row) for row in values):<21.6e}{probability:.6e}")
    print(f"\nlog-space result: {log_forward(observations)[1]:.12f}")
    print(f"scaled result:    {scaled_forward(observations):.12f}")
    # task 3l: log-space is robust and general; scaling is convenient for the
    # backward pass because the forward scale factors can be reused directly.

    print("\n4. backward pass, filtering and smoothing")
    print("-" * 45)
    beta, backward_probability = backward(observations)
    print(f"backward probability: {backward_probability:.12e}")
    posterior = []
    filtering = []
    for t, row in enumerate(alpha):
        total = sum(row[s] * beta[t][s] for s in range(2))
        posterior.append([row[s] * beta[t][s] / total for s in range(2)])
        filtering.append([row[s] / sum(row) for s in range(2)])
    print("\nposition       filter loaded       smoother loaded")
    for t in range(len(observations)):
        print(f"{t + 1:<15}{filtering[t][1]:<20.6f}{posterior[t][1]:.6f}")
    difference = max(range(len(observations)), key=lambda t: abs(filtering[t][1] - posterior[t][1]))
    print(f"\nlargest difference: position {difference + 1}")
    # task 4q: the smoother also knows future rolls, while the filter only knows
    # rolls up to the current position.

    print("\n5. viterbi algorithm")
    print("-" * 30)
    best_path, best_log_probability = viterbi(observations)
    print("viterbi path:")
    print(" ".join(states[x] for x in best_path))
    print(f"viterbi log probability: {best_log_probability:.12f}")
    posterior_path = [max(range(2), key=lambda s: posterior[t][s]) for t in range(len(observations))]
    print(f"decoding disagreements: {sum(a != b for a, b in zip(best_path, posterior_path))}")
    # task 5t: back-pointers are needed to recover one consistent best path.
    # choosing the best state independently can join incompatible transitions.

    v_accuracy, p_accuracy, disagreements = [], [], []
    for seed in range(10):
        seq, true_path = generate(300, seed)
        v_path, _ = viterbi(seq)
        post = log_forward(seq)[0]
        post_path = [max(range(2), key=lambda s: post[t][s]) for t in range(300)]
        v_accuracy.append(sum(a == b for a, b in zip(v_path, true_path)) / 300)
        p_accuracy.append(sum(a == b for a, b in zip(post_path, true_path)) / 300)
        disagreements.append(sum(a != b for a, b in zip(v_path, post_path)))
    print("\n6. viterbi versus posterior decoding")
    print("-" * 40)
    print(f"mean viterbi accuracy:    {sum(v_accuracy) / 10:.4f}")
    print(f"mean posterior accuracy:  {sum(p_accuracy) / 10:.4f}")
    print(f"mean disagreements:       {sum(disagreements) / 10:.2f}")
    print("viterbi optimises whole-path probability.")
    print("posterior decoding optimises each position independently.")
    print("a zero transition can make a posterior-decoded path impossible.")
    # task 6x: viterbi maximises the probability of one complete sequence;
    # posterior decoding maximises accuracy at each individual position.


if __name__ == "__main__":
    report()
