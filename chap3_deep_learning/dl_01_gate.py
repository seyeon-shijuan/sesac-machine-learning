
def AND(x1, x2):
    return 1 if (x1 and x2) else 0


def and_gate(x1, x2):
    theta = -1.5
    return 1 if (x1 + x2 + theta > 0) else 0

def NAND(x1, x2):
    return 1 if not(x1 and x2) else 0


def nand_gate(x1, x2):
    theta = -1.5
    return 1 if not (x1 + x2 + theta > 0) else 0

def OR(x1, x2):
    return 1 if (x1 or x2) else 0


def or_gate(x1, x2):
    theta = -0.5
    return 1 if (x1 + x2 + theta > 0) else 0


def XOR(x1, x2):
    p = NAND(x1, x2)
    q = OR(x1, x2)
    r = AND(p, q)
    return r


def xor_gate(x1, x2):
    p = nand_gate(x1, x2)
    q = or_gate(x1, x2)
    r = and_gate(p, q)
    return r


def perceptron(w1, w2, b, x1, x2):
    wgt_sum = x1 * w1 + x2 * w2 + b
    return 1 if wgt_sum > 0 else 0


def half_adder(x1, x2):
    s = xor_gate(x1, x2)
    c = and_gate(x1, x2)
    return s, c


# def full_adder(x1, x2, ):



# print(f"{and_gate(1, 1) = }")
# print(f"{nand_gate(1, 1) = }")
# print(f"{or_gate(1, 0) = }")
# print(f"{xor_gate(1, 0) = }")
#
# print(f"and: {perceptron(0.5, 0.5, -0.7, 1, 1) = }")
# print(f"nand: {perceptron(0.5, 0.5, -0.3, 0, 1) = }")
# print(f"or: {perceptron(0.5, 0.5, 0, 0, 0) = }")


half_adder(1, 0)
half_adder(1, 1)