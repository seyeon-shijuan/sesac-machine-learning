def datatype_test():
    test_int = 10
    test_float = 3.14

    test_list = {1, 2, 3}
    test_dict = {'a': 10, 'b': 20}

    print(f"test_int: {type(test_int)}")
    print(f"test_int: {type(test_float)}")
    print(f"test_int: {type(test_list)}")
    print(f"test_int: {type(test_dict)}")


def class_test1():
    class TestClass:
        pass


    object1 = TestClass()
    object2 = TestClass()

    print(f"object1: {type(object1)}")
    print(f"object2: {type(object2)}")


def class_test2():
    class Person:
        def say_hello(self, name):
            print("Hello!", name)

        def say_bye(self, name):
            print("GoodBye!", name)


    person = Person()
    person.say_bye("Kim")
    person.say_bye("Yang")


def class_test3():
    class Person:
        def set_name(self, name):
            self.name = name

        def say_hello(self):
            print(f"Hello! I'm {self.name}")

    person1, person2 = Person(), Person()

    person1.set_name('Kim')
    person2.set_name('Yang')

    print(person1.name)
    print(person2.name)

    person1.say_hello()
    person2.say_hello()


def class_test4():
    class Person:
        def set_name(self, name):
            self.name = name

        def say_hello(self):
            print(f"Hello! I'm {self.name}")

        def get_name(self):
            return self.name

        def get_family_name(self):
            return self.name[0]

        def get_personal_name(self):
            return self.name[1:]


    person = Person()
    person.set_name("김철수")
    print(person.get_name())
    print(person.get_family_name())
    print(person.get_personal_name())


def class_test5():
    class Person:
        def __init__(self, name):
            self.name = name
            self.say_hello()

        def say_hello(self):
            print(f"Hello! I'm {self.name}")

    person1 = Person("Yang")
    person2 = Person("Shin")


class LogicGate:
    def __init__(self, w1, w2, b):
        self.w1 = w1
        self.w2 = w2
        self.b = b

    def __call__(self, x1, x2):
        return 1 if (self.w1 * x1 + self.w2 * x2 + self.b) > 0 else 0


class ANDGate:
    def __init__(self):
        self.gate = LogicGate(0.5, 0.5, -0.7)

    def __call__(self, x1, x2):
        return self.gate(x1, x2)

class NANDGate:
    def __init__(self):
        self.gate = LogicGate(-0.5, -0.5, 0.7)

    def __call__(self, x1, x2):
        return self.gate(x1, x2)


class ORGate:
    def __init__(self):
        self.gate = LogicGate(0.5, 0.5, -0.2)

    def __call__(self, x1, x2):
        return self.gate(x1, x2)


class NORGate:
    def __init__(self):
        self.gate = LogicGate(-0.5, -0.5, 0.2)

    def __call__(self, x1, x2):
        return self.gate(x1, x2)


class XORGate:
    def __init__(self):
        self.nand_gate = NANDGate()
        self.or_gate = ORGate()
        self.and_gate = ANDGate()

    def __call__(self, x1, x2):
        p = self.nand_gate(x1, x2)
        q = self.or_gate(x1, x2)
        r = self.and_gate(p, q)
        return r


class XNORGate:
    def __init__(self):
        self.nor_gate = NORGate()
        self.and_gate = ANDGate()
        self.or_gate = ORGate()

    def __call__(self, x1, x2):
        p = self.nor_gate(x1, x2)
        q = self.and_gate(x1, x2)
        r = self.or_gate(p, q)
        return r



if __name__ == '__main__':
    and_gate = ANDGate()
    nand_gate = NANDGate()
    or_gate = ORGate()
    nor_gate = NORGate()
    xor_gate = XORGate()
    xnor_gate = XNORGate()

    print("========== 1 LAYER (LINEAR) ==========")
    print(f"{and_gate(1, 1) = }")
    print(f"{nand_gate(1, 1) = }")
    print(f"{or_gate(0, 0) = }")
    print(f"{nor_gate(0, 0) = }")

    print("========== XOR ==========")
    print(f"{xor_gate(0, 0) = }")
    print(f"{xor_gate(0, 1) = }")
    print(f"{xor_gate(1, 0) = }")
    print(f"{xor_gate(1, 1) = }")

    print("========== XNOR ==========")
    print(f"{xnor_gate(0, 0) = }")
    print(f"{xnor_gate(0, 1) = }")
    print(f"{xnor_gate(1, 0) = }")
    print(f"{xnor_gate(1, 1) = }")

    print("========== VALIDATION TEST ==========")
    to_test = [(0, 0), (0, 1), (1, 0), (1, 1)]
    xor_target = [0, 1, 1, 0]
    xnor_target = [1, 0, 0, 1]

    for test, xor, xnor in zip(to_test, xor_target, xnor_target):
        xor_val = xor_gate(*test)
        xnor_val = xnor_gate(*test)

        print(f"{test=} {xor_val=} ({xor_val == xor}), {xnor_val=} ({xnor_val == xnor})")



