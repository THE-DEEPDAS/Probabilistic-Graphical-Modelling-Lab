"""
AI453  --  Practical #2: Conditional Independence
Two joint distributions, printed on the handout. Plain Python 3, no imports.
Run:  python3 lab2_independence.py
"""

# TABLE 1 -- P(A, B, C).  Key (a, b, c) means A=a, B=b, C=c.
from tkinter import ROUND


P1 = {
    #  A  B  C
    (0, 0, 0): 0.36,
    (0, 0, 1): 0.04,
    (0, 1, 0): 0.01,
    (0, 1, 1): 0.09,
    (1, 0, 0): 0.09,
    (1, 0, 1): 0.01,
    (1, 1, 0): 0.04,
    (1, 1, 1): 0.36,
}

# TABLE 2 -- P(R, S, W).  R = it rained, S = sprinkler was on, W = grass is wet.
P2 = {
    #  R  S  W
    (0, 0, 0): 0.27,
    (0, 0, 1): 0.03,
    (0, 1, 0): 0.12,
    (0, 1, 1): 0.18,
    (1, 0, 0): 0.08,
    (1, 0, 1): 0.12,
    (1, 1, 0): 0.02,
    (1, 1, 1): 0.18,
}


# ----------------------------------------------------------------------
# Last week's loop, wrapped up. Position 0 is the first variable, 1 the
# second, 2 the third.
#
#   prob(P1, {0: 1})           P(A=1)
#   prob(P1, {0: 1, 2: 1})     P(A=1, C=1)
#   cond(P1, {2: 1}, {0: 1})   P(C=1 | A=1)     query first, then given
# ----------------------------------------------------------------------
def prob(table, conditions):
    total = 0.0
    for row, p in table.items():
        if all(row[i] == v for i, v in conditions.items()):
            total += p
    return total


def cond(table, query, given):
    both = dict(given)
    both.update(query)
    return prob(table, both) / prob(table, given)


print("P(A=1) =", prob(P1, {0: 1}))


# ======================================================================
# TABLE 1
# ======================================================================

# T1.  Print the total of each table. Both should be 1.
print("P1 total:", sum(P1.values()))
print("P2 total:", sum(P2.values()))


# T2.  Print P(A=1, C=1) and P(A=1) * P(C=1).
#      Equal? If not, A and C are dependent.
print("P(A=1, C=1) =", prob(P1, {0: 1, 2: 1}))
print("P(A=1) * P(C=1) =", prob(P1, {0: 1}) * prob(P1, {2: 1}))

# T3.  Print P(C=1 | A=1) and P(C=1 | A=0). How far apart?
print("P(C=1 | A=1) =", cond(P1, {2: 1}, {0: 1}))
print("P(C=1 | A=0) =", cond(P1, {2: 1}, {0: 0}))

print("P(C=1 | A=1) - P(C=1 | A=0) =", cond(P1, {2: 1}, {0: 1}) - cond(P1, {2: 1}, {0: 0}))

# T4.  Print these three:
#          P(C=1 | B=1)      P(C=1 | B=1, A=1)      P(C=1 | B=1, A=0)
#      Once B is known, does A still change anything?
print("P(C=1 | B=1) =", cond(P1, {2: 1}, {1: 1}))
print("P(C=1 | B=1, A=1) =", cond(P1, {2: 1}, {1: 1, 0: 1}))
print("P(C=1 | B=1, A=0) =", cond(P1, {2: 1}, {1: 1, 0: 0}))
print("Once B is known, A does not change anything. P(C=1 | B=1) = P(C=1 | B=1, A=1) = P(C=1 | B=1, A=0)")

# T5.  Same three for B=0. Then finish this line:
#          "A and C are ____________, but ____________ given B."
# your code here
#
# ANSWER: "A and C are dependent, but independent given B."
print("P(C=1 | B=0) =", cond(P1, {2: 1}, {1: 0}))
print("P(C=1 | B=0, A=1) =", cond(P1, {2: 1}, {1: 0, 0: 1}))
print("P(C=1 | B=0, A=0) =", cond(P1, {2: 1}, {1: 0, 0: 0}))

# now i can check p(a , b) = p(a) * p(b) to see if they are independent. 
lhs = prob(P1, {0: 1, 2: 1})
rhs = prob(P1, {0: 1}) * prob(P1, {2: 1})

if (lhs == rhs):
    print("A and C are independent")
else:
    print("A and C are dependent")

# A and C but conditioned on B. for this we check p(a, c | b) = p(a | b) * p(c | b)
lhs = prob(P1, {0: 1, 2: 1, 1: 1})
rhs = prob(P1, {0: 1, 1: 1}) * prob(P1, {2: 1, 1: 1}) / prob(P1, {1: 1})
print(lhs)
print(rhs)
if (round(lhs, 2) == round(rhs, 2)):
    print("A and C are independent given B \n")
else:
    print("A and C are dependent given B \n")

# ======================================================================
# TABLE 2
# ======================================================================
print("TABLE 2 Tasks \n")

# T6.  Print P(R=1, S=1) and P(R=1) * P(S=1). These should agree --
#      rain and sprinklers are unrelated.
print("P(R=1, S=1) =", prob(P2, {0: 1, 1: 1}))
print("P(R=1) * P(S=1) =", prob(P2, {0: 1}) * prob(P2, {1: 1}))

print("As one prob is 0.19999999999999998 and the other is 0.2, they are equal within rounding error. \n Rain and sprinklers are unrelated or independent.")

# T7.  Print these four:
#          P(R=1)              P(R=1 | W=1)
#          P(R=1 | W=1, S=1)   P(R=1 | W=1, S=0)
print("P(R=1) =", prob(P2, {0: 1}))
print("P(R=1 | W=1) =", cond(P2, {0: 1}, {2: 1}))
print("P(R=1 | W=1, S=1) =", cond(P2, {0: 1}, {2: 1, 1: 1}))
print("P(R=1 | W=1, S=0) =", cond(P2, {0: 1}, {2: 1, 1: 0}))

# T8.  In T5, conditioning REMOVED a dependence. In T7 it CREATED one.
#      Two or three lines: why does learning the sprinkler was on make
#      rain less likely, when the grass is just as wet either way?
#
# ANSWER:
print("Learning that the sprinkler was on makes rain less likely because if the grass is wet, it could be due to either rain or the sprinkler. If we know the sprinkler was on, so now we have atleast one evidence for wet grass and both occuring togather is naturally rare as human would shut the sprinkler off, thus reducing the probability of rain given that information. \n")