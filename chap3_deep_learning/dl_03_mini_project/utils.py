from chap3_deep_learning.dl_03_mini_project.game_objects import Character
from chap3_deep_learning.dl_03_mini_project.game_objects import Slime


class Utils:
    def __init__(self, char=Character()):
        self.char = char
        self.slime = Slime()

        self.moves = {
            "1": self.char.attack_monster,
            "2": self.char.print_states,
            "3": lambda x: print("2 pass"),
            "4": self.char.save_states,
            "0": self.exit_game
        }

    @staticmethod
    def exit_game():
        print("============ 게임 종료 ===========")
        print("====== 메인화면으로 이동합니다. =====")
        return -1

    def game_main(self, char=None):
        if char is not None:
            self.char = char

        while True:
            print("======================================")
            next_move = input("1. 몬스터 잡기 \n"
                              "2. 현재 상태 확인 \n"
                              "3. 물약 사기(30원) \n"
                              "4. 게임 저장하기 \n"
                              "0. 게임 종료 \n"
                              "다음 중 어떤 것을 하시겠습니까?"
                              )

            # 몬스터 잡기인 경우 현재 몬스터의 상태를 param으로 전달
            if next_move == "1":
                print(f"공격 전: {self.char.get_hp()=}, {self.slime.get_hp()=}")
                char_hp = self.moves[next_move](self.slime)
                print(f"공격 후: {self.char.get_hp()=}, {char_hp=}")

                # if slime.get_hp()

                if char_hp <= 0:
                    print("here")




                continue

            result = self.moves[next_move]()

            if result == -1:
                break
