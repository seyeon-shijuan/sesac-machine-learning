import os

from chap3_deep_learning.dl_03_mini_project.utils import Utils
from game_objects import Character
from game_objects import SAVED_PATH


class GameLauncher:
    def __init__(self):
        self.char = None
        self.utils = None

    def __call__(self):
        while True:
            print("======================================")
            # next_move = int(input("SeSAC 온라인에 오신 것을 환영합니다.\n"
            #                       "1. 새로운 게임 시작하기 \n"
            #                       "2. 지난 게임 불러오기 \n"
            #                       "3. 게임 종료하기 \n"
            #                       "다음 중 어떤 것을 하시겠습니까?"
            #                       ))
            next_move = 1

            if next_move == 1:
                print("새로운 캐릭터를 생성합니다.")
                self.char = Character()
                self.char.print_states()
                self.utils = Utils()
                self.utils.game_main(self.char)

            elif next_move == 2:
                if os.path.exists(SAVED_PATH):
                    print("저장된 파일을 불러옵니다.")
                    self.char = Character()
                    self.char.load_states()
                    self.char.print_states()
                    self.utils = Utils()
                    self.utils.game_main(self.char)

                print("저장된 파일이 없습니다. 메인 화면으로 돌아갑니다.")
                continue

            elif next_move == 3:
                print("게임을 종료합니다.")
                break


if __name__ == '__main__':
    launcher = GameLauncher()
    launcher()





