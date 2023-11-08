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
            # 'to_lv_up': 100,
            'hp': 100,
            'max_hp': 100,
            'damage': 10,
            'money': 100,
            'n_potion': 10
        }

        for k, v in base_attr.items():
            setattr(self, k, v)

    def print_states(self):
        print("\n----------")
        print("현재 레벨:", self.lv)
        print("현재 경험치:", self.exp)
        print("다음 레벨을 위한 경험치:", self.lv * 100)
        print(f"HP: {self.hp}")
        print(f"HP 최대치: {self.max_hp}")
        print(f"공격력: {self.damage}")
        print(f"돈: {self.money}")
        print('----------\n')
        # states = vars(self)
        # states['to_lv_up'] = states['to_lv_up'] - states['exp']
        # to_print = ("--------------- \n "
        #             f"현재 레벨: {self.lv} \n "
        #             f"현재 경험치: {self.exp} \n "
        #             f"다음 레벨을 위한 경험치: {self.lv * 100} \n "
        #             f"HP: {self.hp} \n "
        #             f"HP 최대치: {self.max_hp} \n "
        #             f"공격력: {self.damage} \n "
        #             f"돈: {self.money} \n "
        #             f"포션: {self.n_potion} \n "
        #             "---------------")

        # print(to_print)

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

        # self.to_lv_up = self.lv * 100

    def check_level_up(self):
        if self.exp >= self.lv * 100:
            self.lv += 1
            print(f"LEVEL UP -> Lv.{self.lv}")

            self.max_hp += 10
            self.set_hp(self.max_hp)
            self.damage += 3

    def attack_monster(self, slime):
        print(f"Before Attack >> char: {self.get_hp()}, slime: {slime.get_hp()}")
        # 사람 공격
        slime_hp = slime.get_hp()
        slime_hp -= self.damage
        slime.set_hp(slime_hp)

        # 슬라임을 잡았으면
        if slime_hp <= 0:
            kill_exp = slime.get_kill_exp()
            kill_money = slime.get_kill_money()
            self.exp += kill_exp
            self.money += kill_money
            print(f"경험치: {kill_exp}, 돈: {kill_money} 획득!")
            slime.__init__()
            self.check_level_up()
            return

        # 슬라임 반격
        char_hp = self.get_hp()
        char_hp -= self.damage
        self.set_hp(char_hp)
        print(f"After Attack >> char: {self.get_hp()}, slime: {slime.get_hp()}")

    def buy_potion(self):
        self.n_potion += 1
        self.money -= 30
        print(f"잔액: {self.money}, 현재 물약: {self.n_potion}개")

    def drink_potion(self):
        new_hp = self.get_hp() + 50
        self.set_hp(min(new_hp, self.max_hp))
        self.n_potion -= 1
        print(f"물약을 마셨습니다. 잔여 체력: {self.get_hp()}")





class Monster(GameObject):
    def __init__(self):
        super().__init__()
        base_attr = {
            'hp': 0,
            'damage': 0,
            'kill_exp': 0,
            'kill_money': 0
        }

        for k, v in base_attr.items():
            setattr(self, k, v)

    def get_kill_exp(self):
        return self.kill_exp

    def get_kill_money(self):
        return self.kill_money

class Slime(Monster):
    def __init__(self):
        super().__init__()
        base_attr = {
            'hp': 30,
            'damage': 2,
            'kill_exp': 50,
            'kill_money': 10
        }

        for k, v in base_attr.items():
            setattr(self, k, v)

