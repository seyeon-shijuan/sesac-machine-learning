

class Animal:

    def __init__(self):
        self.legs = 4
        self.head = 1

    def speak(self, new_leg):
        print(f"{self.legs=}")
        print(f"{new_leg=}")




class Dog(Animal):

    def __init__(self):
        super().__init__()
        self.finger = 8




a = Animal()
a.speak(new_leg=20)
#
# dog = Dog()
#
# print('here')