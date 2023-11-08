import collections
import pandas as pd

SAVED_PATH = "./data/save_file.csv"


class GameObject:
    def __init__(self):
        self.hp = 0
        self.damage = 0

    def get_hp(self):
        return self.hp

    def set_hp(self, new_hp):
        self.hp = new_hp


class Character(GameObject):

    def __init__(self):
        super().__init__()
        base_attr = {
            'lv': 1,
            'exp': 0,
            'to_lv_up': 100,
            'hp': 100,
            'max_hp': 100,
            'damage': 10,
            'money': 100
        }
        base_attr = collections.OrderedDict(base_attr)

        for k, v in base_attr.items():
            setattr(self, k, v)

    def print_states(self):
        states = vars(self)
        states['to_lv_up'] = states['to_lv_up'] - states['exp']
        # to_print = ("--------------- \n "
        #             "현재 레벨: {} \n "
        #             "현재 경험치: {} \n "
        #             "다음 레벨을 위한 경험치: {} \n "
        #             "HP: {} \n "
        #             "HP 최대치: {} \n "
        #             "공격력: {} \n "
        #             "돈: {} \n "
        #             "---------------").format(*states.values())
        to_print = ("--------------- \n "
                    f"현재 레벨: {self.lv} \n "
                    f"현재 경험치: {self.exp} \n "
                    f"다음 레벨을 위한 경험치: {self.to_lv_up} \n "
                    f"HP: {self.hp} \n "
                    f"HP 최대치: {self.max_hp} \n "
                    f"공격력: {self.damage} \n "
                    f"돈: {self.money} \n "
                    "---------------")

        print(to_print)

    def save_states(self):
        states = vars(self)
        del (states['to_lv_up'])
        df = pd.DataFrame(states, index=[0])
        df.to_csv(SAVED_PATH, index=False)
        print("saved")

    def load_states(self):
        df = pd.read_csv(SAVED_PATH)
        states = df.iloc[0, :].to_dict()
        for k, v in states.items():
            setattr(self, k, v)

        self.to_lv_up = self.lv * 100

    def attack_monster(self, slime):
        # 사람 공격
        slime_hp = slime.get_hp()
        slime_hp -= self.damage
        slime.set_hp(slime_hp)

        # 슬라임 반격
        char_hp = self.get_hp()
        char_hp -= slime.damage
        self.set_hp(char_hp)

        return slime_hp


class Monster(GameObject):
    def __init__(self):
        super().__init__()
        base_attr = {
            'hp': 0,
            'damage': 0,
            'kill_exp': 0,
            'kill_money': 0
        }
        base_attr = collections.OrderedDict(base_attr)

        for k, v in base_attr.items():
            setattr(self, k, v)



class Slime(Monster):
    def __init__(self):
        super().__init__()
        base_attr = {
            'hp': 30,
            'damage': 2,
            'kill_exp': 50,
            'kill_money': 10
        }
        base_attr = collections.OrderedDict(base_attr)

        for k, v in base_attr.items():
            setattr(self, k, v)




m = Character()
print('here')