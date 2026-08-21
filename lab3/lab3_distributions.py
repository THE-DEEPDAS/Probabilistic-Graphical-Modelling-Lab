import math
import random

# A fixed seed, so your own runs repeat. It does NOT make your numbers match
# your neighbour's -- as soon as two people draw a different number of samples,
# the streams diverge. Anything you estimate from samples will differ a little
# from everyone else's, and that is the subject of T3. Compare your sampled
# numbers against the formulas, never against each other.
random.seed(353)


# ======================================================================
# THE DATA  --  both of these are printed on your handout.
# ======================================================================

# Twenty flips of a coin, in order. 1 = heads.
COIN = "11000100100000101000"

# A loaded six-sided die: P(1), P(2), ..., P(6). Note the last one.
DIE = [0.25, 0.20, 0.20, 0.15, 0.18, 0.02]


# ======================================================================
# GIVEN TO YOU  --  use these, do not edit them.
# ======================================================================

def mean_of(xs):
    """Average of a list."""
    return sum(xs) / len(xs)


def var_of(xs):
    """Variance of a list, as E[X^2] - (E[X])^2."""
    m = mean_of(xs)
    return sum(x * x for x in xs) / len(xs) - m * m


def ascii_hist(values):
    """Draw a histogram of a list of whole numbers."""
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    top = max(counts.values())
    for v in range(min(values), max(values) + 1):
        c = counts.get(v, 0)
        print(f"    {v:>4} | {'#' * round(54 * c / top):<54} {c}")


def check(label, got, want, tol=1e-9):
    """Compare a number with what it should be, and say so."""
    ok = abs(got - want) <= tol
    print(f"[{'  ok  ' if ok else ' FAIL '}] {label:<34} "
          f"got {got:>10.4f}   want {want:>10.4f}")


# ----------------------------------------------------------------------
# WORKED EXAMPLE -- nothing to write.
#
# A Bernoulli draw: 1 with probability p, else 0. This is the one-line
# pattern that every sampler below is built out of.
# ----------------------------------------------------------------------

def sample_bernoulli(p):
    return 1 if random.random() < p else 0


print("=" * 74)
print("WORKED EXAMPLE  --  a Bernoulli draw")
print("=" * 74)

draws = [sample_bernoulli(0.3) for _ in range(10000)]
check("fraction of 1s in 10000 draws", mean_of(draws), 0.3, tol=0.02)
print()


# ======================================================================
# PART A  --  SAMPLING
# ======================================================================

# T1. Write sample_die(). It should return one roll of the loaded die
#     above, as an index 0..5 (so face 1 is 0, face 6 is 5).
#
#     The method is the INVERSE CDF, and it is four lines:
#
#         draw u = random.random(), a number in [0, 1)
#         keep a running total, adding DIE[0], then DIE[1], and so on
#         the moment the running total goes past u, return that index
#
#     Picture the line from 0 to 1 cut into six pieces, of lengths 0.25,
#     0.20, 0.20, 0.15, 0.18, 0.02. The question is: which piece did u
#     land in? This is how every discrete distribution in this course is
#     sampled.
#
#     Then draw 10000 rolls and print the fraction of each face. They
#     should match DIE.
# ----------------------------------------------------------------------

def sample_die():
    # t1 is here
    cumulative_die = [sum(DIE[:i+1]) for i in range(len(DIE))]
    u = random.random()

    for i, threshold in enumerate(cumulative_die):
        if u < threshold:
            return i

print("=" * 74)
print("T1  sampling the die")
print("=" * 74)

rolls = [sample_die() for _ in range(10000)]
for i in range(6):
    check(f"fraction of {i+1}s in 10000 rolls", mean_of([x == i for x in rolls]), DIE[i], tol=0.02)
print()


# T2. Write sample_binomial(n, p): the number of 1s in n Bernoulli draws.
#     Do NOT use a formula with factorials. Call sample_bernoulli(p) n
#     times and add up what you get -- that is what "the Binomial is a
#     sum of Bernoullis" means, and T5 depends on your having seen it
#     that way.
#
#     Draw 10000 values of sample_binomial(10, 0.3) and check that their
#     mean is about n*p = 3 and their variance about n*p*(1-p) = 2.1.
# ----------------------------------------------------------------------

def sample_binomial(n, p):
    total = 0
    for _ in range(n):
        total += sample_bernoulli(p)
    return total

print("=" * 74)
print("T2  the Binomial is a sum of Bernoullis")
print("=" * 74)

binomial_draws = [sample_binomial(10, 0.3) for _ in range(10000)]
mean = mean_of(binomial_draws)
variance = var_of(binomial_draws)

check("mean of 10000 draws of Binomial(10, 0.3)", mean, 3, tol=0.1)
check("variance of 10000 draws of Binomial(10, 0.3)", variance, 2.1, tol=0.1)

# ======================================================================
# PART B  --  EXPECTATION AND VARIANCE
# ======================================================================

# T3. Deck 3 gives the mean and variance of the die by formula:
#
#         E[X]   = sum over k of  k * DIE[k]
#         Var(X) = sum over k of  k*k * DIE[k]  -  E[X]^2
#
#     (k is the 0-based index, matching what sample_die returns.)
#
#     Compute both from the formula and print them. Then draw n rolls
#     and estimate both with mean_of() and var_of(), for
#     n = 100, 1000, 10000, 100000. Print a small table of the errors.
#
#     Do the two formula values on paper first -- it is twelve
#     multiplications -- then check the computer agrees.
#
#     One line in a comment: each time n goes up by a factor of 100,
#     what happens to the error? Deck 3 predicted this.
# ----------------------------------------------------------------------

E_X = sum(k * DIE[k] for k in range(len(DIE)))
Var_X = sum(k * k * DIE[k] for k in range(len(DIE))) - E_X * E_X

print("=" * 74)
print("T3  mean and variance: formula against data")
print("=" * 74)

# calling it for n = 100, 1000, 10000, 100000. print small table of errors
for n in [100, 1000, 10000, 100000]:
    rolls = [sample_die() for _ in range(n)]
    mean_estimate = mean_of(rolls)
    variance_estimate = var_of(rolls)

    mean_error = abs(mean_estimate - E_X)
    variance_error = abs(variance_estimate - Var_X)

    print(f"n={n:<7} Mean Error: {mean_error:.6f}, Variance Error: {variance_error:.6f}")

# Each time n goes up by a factor of 100, the typical error decreases
# by roughly a factor of 10.


# T4. Variance of a sum. Draw 20000 pairs (x, y), twice over:
#
#         independent:  x and y each a separate sample_bernoulli(0.3)
#         coupled:      x a sample_bernoulli(0.3), and then y = x
#
#     Both times, estimate Var(X + Y) with var_of(). Compare each against
#     Var(X) + Var(Y) = 0.21 + 0.21 = 0.42.
#
#     One agrees and one does not. In a comment: what is the extra term
#     in the general formula, what is its value in the coupled case, and
#     when does it vanish?
# ----------------------------------------------------------------------

X = [sample_bernoulli(0.3) for _ in range(20000)]
Y_independent = [sample_bernoulli(0.3) for _ in range(20000)]
Y_dependent = [x for x in X]  # Y is the same as X

print("=" * 74)
print("T4  variance of a sum")
print("=" * 74)

var_x = var_of(X)
var_y = var_of(Y_independent)
var_x_y = var_of(X + Y_independent)

print(f"Variance of X is {var_x}, that of Y is {var_y} and X + Y is {var_x_y} in independent case.")

var_xy_dependent = var_of(X + Y_dependent)
var_y_depdendent = var_of(Y_dependent)

print(f"Variance of X is {var_x}, that of Y is {var_y_depdendent} and X + Y is {var_xy_dependent} in dependent case.")


# ANSWER:
# The general formula is Var(X + Y) = Var(X) + Var(Y) + 2Cov(X, Y).
# In the coupled case Y = X, so Cov(X, Y) = Var(X).
# The covariance term vanishes when X and Y are uncorrelated,
# in particular when they are independent.


# ======================================================================
# PART C  --  WHY GAUSSIANS ARE EVERYWHERE
# ======================================================================

# T5. Deck 3 claimed that adding up many independent things gives you a
#     bell curve, whatever you started from. Check it.
#
#     For k = 1, 5, 30, 100: build a list of 5000 values, each one the
#     sum of k draws of sample_bernoulli(0.1), and ascii_hist() it.
#
#     Bernoulli(0.1) is about as far from a bell as a distribution gets
#     -- at k = 1 your histogram is two bars. Look at k = 100.
#
#     Hint: sample_binomial(k, 0.1) already IS the sum of k draws of
#     sample_bernoulli(0.1). You wrote it in T2.
#
#     For each k, also print the mean and variance of your 5000 sums
#     beside k*0.1 and k*0.09, which is what Deck 3 predicts for a sum of
#     k independent copies.
#
#     Two lines in a comment. The shape changed completely between k = 1
#     and k = 100, but those two predictions held at every k, including
#     k = 1 where nothing is remotely bell-shaped. So what did the
#     central limit theorem actually give you -- the mean, the variance,
#     or the shape?
# ----------------------------------------------------------------------

print("=" * 74)
print("T5  sums of many things")
print("=" * 74)

for k in [1, 5, 30, 100]:
    values = [
        sample_binomial(k, 0.1)
        for _ in range(5000)
    ]

    actual_mean = mean_of(values)
    actual_variance = var_of(values)

    expected_mean = k * 0.1
    expected_variance = k * 0.09

    print(f"\nk = {k}")
    print(
        f"mean     = {actual_mean:.4f} "
        f"(expected {expected_mean:.4f})"
    )
    print(
        f"variance = {actual_variance:.4f} "
        f"(expected {expected_variance:.4f})"
    )

    ascii_hist(values)

# ANSWER:
# The mean and variance come from the sum properties, while the central
# limit theorem mainly predicts the shape: the sum becomes approximately
# bell-shaped as k becomes large.


# ======================================================================
# PART D  --  MAXIMUM LIKELIHOOD
# ======================================================================

# T6. The twenty coin flips at the top of the file. Read them with
#         flips = [int(c) for c in COIN]
#
#     Write likelihood(theta, flips): the probability of getting exactly
#     these flips if the coin comes up heads with probability theta. It
#     is theta for each head and (1 - theta) for each tail, all
#     multiplied together.
#
#     Then try every theta from 0.00 to 1.00 in steps of 0.01, and print
#     the one that gives the largest value. Compare it with
#     (number of heads) / 20.
#
#     You are not being asked to trust the lecture's formula. You are
#     checking every possible theta by brute force, and finding that the
#     best one is where the lecture said it would be.
# ----------------------------------------------------------------------

def likelihood(theta, flips):
    result = 1.0

    for flip in flips:
        if flip == 1:
            result *= theta
        else:
            result *= (1 - theta)

    return result


print("=" * 74)
print("T6  maximum likelihood, by brute force")
print("=" * 74)

flips = [int(c) for c in COIN]

best_theta = 0.0
best_value = -1.0

for i in range(101):
    theta = i / 100
    value = likelihood(theta, flips)

    if value > best_value:
        best_value = value
        best_theta = theta

heads = sum(flips)
mle_theta = heads / len(flips)

print(f"best theta by brute force = {best_theta:.2f}")
print(f"heads / 20                = {mle_theta:.2f}")
print(f"maximum likelihood        = {best_value}")


# T7. Print likelihood(0.3, flips * 65). That is the same twenty flips
#     repeated 65 times: 1300 flips, which is a small dataset.
#
#     Look at what comes out. Then write log_likelihood(theta, flips),
#     which adds up math.log(...) instead of multiplying, and print that
#     for the same 1300 flips.
#
#     One line in a comment: what happened to the first number, and why
#     does every library in this course work in logs?
# ----------------------------------------------------------------------

def log_likelihood(theta, flips):
    result = 0.0

    for flip in flips:
        if flip == 1:
            result += math.log(theta)
        else:
            result += math.log(1 - theta)

    return result


print("=" * 74)
print("T7  why everything is done in logs")
print("=" * 74)

large_flips = flips * 65

print(
    "likelihood(0.3, flips * 65) =",
    likelihood(0.3, large_flips)
)

print(
    "log_likelihood(0.3, flips * 65) =",
    log_likelihood(0.3, large_flips)
)

# ANSWER:
# The likelihood becomes extremely small because many probabilities are
# multiplied together. Logs turn multiplication into addition and prevent
# numerical underflow, which is why likelihood calculations are done in logs.


# T8. Maximum likelihood for the die is just counting: the estimate of
#     P(face k) is (number of times face k came up) / (total rolls).
#
#     Write fit_die(rolls) returning a list of six such estimates.
#     Check it on 100000 rolls -- you should get DIE back.
#
#     NOW BREAK IT. Face 6 has probability 0.02, so in a short run it
#     often does not appear at all. Fit the die on just 30 rolls, and do
#     that 100 times over. Count how many of the 100 estimates give face
#     6 a probability of exactly 0. Compare with 0.98^30, which you can
#     work out with a pen.
#
#     Then take one of those estimates and use it to score a NEW
#     sequence of ten rolls in which face 6 appears once -- multiply the
#     ten probabilities together. Print the result.
#
#     Two or three lines in a comment. Your estimate says face 6 is
#     impossible. It is not: you were told at the top of this file that
#     its probability is 0.02, and the die does not care what you saw in
#     thirty rolls. Is "I have never seen it, so it cannot happen" a
#     reasonable thing for an estimate to say? If not, what would you
#     rather it did -- and where could that information come from, given
#     that it is definitely not in your thirty rolls?
#
#     You are not expected to know the fix. The next lecture is the
#     answer.
# ----------------------------------------------------------------------

def fit_die(rolls):
    counts = [0] * 6

    for roll in rolls:
        counts[roll] += 1

    total = len(rolls)

    return [
        count / total
        for count in counts
    ]


print("=" * 74)
print("T8  where maximum likelihood breaks")
print("=" * 74)

rolls = [sample_die() for _ in range(100000)]

fitted = fit_die(rolls)

print("Fit from 100000 rolls:")
for i in range(6):
    print(
        f"face {i + 1}: "
        f"estimated={fitted[i]:.4f}, "
        f"actual={DIE[i]:.4f}"
    )


zero_face_6 = 0
estimates = []

for _ in range(100):
    short_rolls = [
        sample_die()
        for _ in range(30)
    ]

    estimate = fit_die(short_rolls)
    estimates.append(estimate)

    if estimate[5] == 0:
        zero_face_6 += 1


print()
print(
    "Number of estimates with P(face 6) = 0:",
    zero_face_6
)

print(
    "Observed fraction:",
    zero_face_6 / 100
)

print(
    "Theoretical probability 0.98^30:",
    0.98 ** 30
)


# Take one of the estimates and score a new sequence of ten rolls
# in which face 6 appears once.

estimated_die = estimates[0]

new_rolls = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3]

score = 1.0

for roll in new_rolls:
    score *= estimated_die[roll]

print()
print("Estimated probabilities:", estimated_die)
print("New sequence:", new_rolls)
print("Probability of new sequence:", score)


# ANSWER:
# No, "I have never seen it, so it cannot happen" is not reasonable.
# The estimate is based only on the small sample, so it can assign zero
# probability to an event that is actually possible.
# We would rather incorporate information from outside the thirty rolls,
# such as a prior or smoothing method. The next lecture gives the fix.


# ======================================================================
# IF YOU FINISH EARLY
# ======================================================================

# 1. Uncorrelated is not independent. Let X be equally likely to be -1,
#    0 or 1, and let Y = X * X. Estimate their covariance from samples:
#        cov = mean_of([x*y ...]) - mean_of(xs) * mean_of(ys)
#    You will get 0. But Y is computed from X, so they are about as
#    dependent as two things can be. Deck 3 asserted this; build the
#    counterexample yourself.

# 2. Sample a Gaussian. Box-Muller turns two uniforms into one standard
#    normal draw:
#        z = math.sqrt(-2 * math.log(1 - u1)) * math.cos(2 * math.pi * u2)
#    Then mu + sigma * z has mean mu and variance sigma^2. Check it with
#    mean_of() and var_of(). (Why 1 - u1 and not u1?)

# 3. In T5, how good is the bell at k = 100 in the middle, and how good
#    is it at the far right edge? The answer to the second half is why
#    Unit 4 needs sampling methods at all.