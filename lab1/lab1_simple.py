"""
AI453 Probabilistic Graphical Models  --  Practical #1: Introduction to Probability Theory
SVNIT Surat, Department of Artificial Intelligence

You are given ONE joint distribution over three binary variables A, B, C.
Everything you compute today comes out of that one table. Nothing is loaded,
downloaded, or estimated from data.

Run:  python3 lab1_simple.py
Dependencies: NONE. Plain Python 3.
"""

# ----------------------------------------------------------------------
# THE JOINT DISTRIBUTION P(A, B, C)
#
# Three binary variables A, B, C, each 0 or 1.  Eight combinations, eight
# numbers.  The key (a, b, c) means "A=a and B=b and C=c".
#
#       P[(1, 0, 1)]  is  P(A=1, B=0, C=1)  =  0.06
# ----------------------------------------------------------------------
P = {
    #  A  B  C        probability
    (0, 0, 0): 0.06,
    (0, 0, 1): 0.24,
    (0, 1, 0): 0.04,
    (0, 1, 1): 0.16,
    (1, 0, 0): 0.09,
    (1, 0, 1): 0.06,
    (1, 1, 0): 0.21,
    (1, 1, 1): 0.14,
}


# ----------------------------------------------------------------------
# WORKED EXAMPLE  --  read this carefully, every task below is this loop again
#
#   P(A=1)  =  sum of P(A=1, B=b, C=c)  over every b and every c
#
# In words: walk through all eight rows, and add up the ones where A is 1.
# ----------------------------------------------------------------------
total = 0.0
for (a, b, c), p in P.items():
    if a == 1:
        total += p
print("P(A=1) =", total)

# That is the whole idea. A marginal is a sum over the rows that match.
# A conditional is one such sum divided by another.



# ----------------------------------------------------------------------
# T1. Check that the table is a valid distribution
# ----------------------------------------------------------------------

total = 0.0
for p in P.values():
    total += p

print("T1: Total probability =", total)



# ----------------------------------------------------------------------
# T2.  Compute and print P(B=1).
#      Same loop as the worked example, different condition.
# ----------------------------------------------------------------------
total = 0.0
for (a, b, c), p in P.items():
    if b == 1:
        total += p

print("T2: P(B=1) =", total)


# ----------------------------------------------------------------------
# T3.  Compute and print P(C=1).
# ----------------------------------------------------------------------
total = 0.0
for (a, b, c), p in P.items():
    if c == 1:
        total += p

print("T3: P(C=1) =", total)


# ----------------------------------------------------------------------
# T4.  Compute and print the joint P(A=1, B=1).
#      Now the condition has two parts.
# ----------------------------------------------------------------------
total = 0.0
for (a, b, c), p in P.items():
    if a == 1 and b == 1:
        total += p

print("T4: P(A=1, B=1) =", total)


# ----------------------------------------------------------------------
# T5.  Compute and print the conditional P(C=1 | A=1).
#
#                        P(A=1, C=1)
#      P(C=1 | A=1)  =  --------------
#                          P(A=1)
#
#      Two sums, one divided by the other. Compute the top and the bottom
#      in the same loop if you like.
# ----------------------------------------------------------------------

numerator = 0.0
denominator = 0.0

for (a, b, c), p in P.items():
    if a == 1 and c == 1:
        numerator += p

    if a == 1:
        denominator += p

conditional = numerator / denominator

print("T5: P(C=1 | A=1) =", conditional)


# ----------------------------------------------------------------------
# T6.  Compute and print P(B=1 | A=0, C=1).
#      Two things known, one thing asked. Same pattern.
# ----------------------------------------------------------------------
numerator = 0.0
denominator = 0.0

for (a, b, c), p in P.items():

    if a == 0 and c == 1:
        denominator += p

        if b == 1:
            numerator += p

conditional = numerator / denominator

print("T6: P(B=1 | A=0, C=1) =", conditional)

# ----------------------------------------------------------------------
# T7.  THE CHAIN RULE.  In class we showed that for any three variables
#
#          P(A,B,C)  =  P(A) * P(B|A) * P(C|A,B)
#
#      Check it numerically. For every one of the eight rows (a,b,c):
#        - look up P(A=a, B=b, C=c) straight from the table
#        - separately compute P(A=a), then P(B=b|A=a), then P(C=c|A=a,B=b)
#          and multiply the three together
#        - print both numbers side by side and say whether they match
#          (allow a tiny difference, e.g. 1e-9, for floating point)
#
#      Then answer in a comment: does the chain rule hold only for THIS
#      table, or for every joint distribution? Why?
# ----------------------------------------------------------------------
print("\nT7: Chain Rule")

for (a, b, c), joint in P.items():

    # P(A=a)
    p_a = 0.0
    for (aa, bb, cc), p in P.items():
        if aa == a:
            p_a += p

    # P(B=b | A=a)
    numerator_b = 0.0
    denominator_a = 0.0

    for (aa, bb, cc), p in P.items():
        if aa == a:
            denominator_a += p

            if bb == b:
                numerator_b += p

    p_b_given_a = numerator_b / denominator_a

    # P(C=c | A=a, B=b)
    numerator_c = 0.0
    denominator_ab = 0.0

    for (aa, bb, cc), p in P.items():
        if aa == a and bb == b:
            denominator_ab += p

            if cc == c:
                numerator_c += p

    p_c_given_ab = numerator_c / denominator_ab

    # Chain rule
    calculated = p_a * p_b_given_a * p_c_given_ab

    matches = abs(joint - calculated) < 1e-9

    print(
        (a, b, c),
        "joint =", joint,
        "chain =", calculated,
        "match =", matches
    )

# The chain rule holds for every joint distribution because it follows
# directly from the definition of conditional probability:
# P(A,B,C) = P(A) * P(B|A) * P(C|A,B).

# ----------------------------------------------------------------------
# T8.  BAYES' RULE.  You know P(A=1) already -- that was the worked
#      example. Now suppose you are told that C = 1. Compute
#
#          P(A=1 | C=1)
#
#      and compare it with P(A=1). Did learning C=1 make A=1 more likely
#      or less likely? Write ONE line saying by how much, and in which
#      direction.
# ----------------------------------------------------------------------
numerator = 0.0
denominator = 0.0

for (a, b, c), p in P.items():

    if a == 1 and c == 1:
        numerator += p

    if c == 1:
        denominator += p

p_a_given_c = numerator / denominator

print("\nT8: P(A=1 | C=1) =", p_a_given_c)

p_a = 0.0

for (a, b, c), p in P.items():
    if a == 1:
        p_a += p

difference = p_a_given_c - p_a

print(
    "Learning C=1 makes A=1 less likely by",
    abs(difference)
)
