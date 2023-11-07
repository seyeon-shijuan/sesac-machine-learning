def test():
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


    def full_adder(A, B, Cin):
        P = xor_gate(A, B)
        S = xor_gate(P, Cin)
        Q = and_gate(P, Cin)
        R = and_gate(A, B)
        Cout = or_gate(Q, R)
        return S, Cout

    def adder4(A, B):
        S0, C0 = half_adder(A[-1], B[-1])
        S1, C1 = full_adder(A[-2], B[-2], C0)
        S2, C2 = full_adder(A[-3], B[-3], C1)
        S3, Cout = full_adder(A[-4], B[-4], C2)
        S = [S3, S2, S1, S0]
        return Cout, S


    print(f"{and_gate(1, 1) = }")
    print(f"{nand_gate(1, 1) = }")
    print(f"{or_gate(1, 0) = }")
    print(f"{xor_gate(1, 0) = }")

    print(f"and: {perceptron(0.5, 0.5, -0.7, 1, 1) = }")
    print(f"nand: {perceptron(0.5, 0.5, -0.3, 0, 1) = }")
    print(f"or: {perceptron(0.5, 0.5, 0, 0, 0) = }")


    half_adder(1, 0)
    half_adder(1, 1)


def AND(x1, x2):
    if x2 > -x1 + 1.5: y = 1
    else: y = 0
    return y

def OR(x1, x2):
    if x2 > -x1 + 0.5: y = 1
    else: y = 0
    return y

def NAND(x1, x2):
    if x2 < -x1 + 1.5: y = 1
    else: y = 0
    return y

def XOR(x1, x2):
    p = NAND(x1, x2)
    q = OR(x1, x2)
    y = AND(p, q)
    return y

def half_adder(A, B):
    S = XOR(A, B)
    C = AND(A, B)
    return S, C

def full_adder(A, B, Cin):
    P = XOR(A, B)
    S = XOR(P, Cin)
    Q = AND(P, Cin)
    R = AND(A, B)
    Cout = OR(Q, R)
    return S, Cout

def adder4(A, B):
    S0, C0 = half_adder(A[-1], B[-1])
    S1, C1 = full_adder(A[-2], B[-2], C0)
    S2, C2 = full_adder(A[-3], B[-3], C1)
    S3, Cout = full_adder(A[-4], B[-4], C2)
    S = [S3, S2, S1, S0]
    return Cout, S


## test code
print(adder4(A=[1, 0, 0, 1], B=[1, 1, 1, 0]))