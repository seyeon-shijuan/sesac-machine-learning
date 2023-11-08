from chap3_deep_learning.dl_03_mini_project.game_objects import Character
from chap3_deep_learning.dl_03_mini_project.game_objects import Slime


class Utils:
    def __init__(self, char=Character()):
        self.char = char
        self.slime = Slime()

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
                              "4. 물약 마시기 \n"
                              "5. 게임 저장하기 \n"
                              "0. 게임 종료 \n"
                              "다음 중 어떤 것을 하시겠습니까?"
                              )

            moves = {
                "1": lambda x=self.slime: self.char.attack_monster(x),
                "2": self.char.print_states,
                "3": self.char.buy_potion,
                "4": self.char.drink_potion,
                "5": self.char.save_states,
                "0": self.exit_game
            }
            result = moves[next_move]()

            if result == -1:
                break

