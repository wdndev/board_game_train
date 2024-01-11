import json
import copy


class BoardState:
    """ 棋盘状态
    """
    def __init__(self, board : list, pi : list, q : float, z : float, 
                 curr_player : int, stage : int, board_size = 19) -> None:
        """ 初始化
            - board : 二维棋盘
            - pi : 当前棋盘概率，二维
            - q : MCTS给出的q
            - z : 胜负信息
        """
        self.board = board
        self.pi = pi
        self.q = q
        self.z = z
        self.curr_player = curr_player
        self.stage = stage
        self.board_size = board_size

    def get_feature(self):
        feature = []

        curr_feature = [[0 for _ in range(self.board_size)] for i in range(self.board_size)]
        oppo_feature = [[0 for _ in range(self.board_size)] for i in range(self.board_size)]
        available_feature = [[0 for _ in range(self.board_size)] for i in range(self.board_size)]
        stage_feature = [[0 for _ in range(self.board_size)] for i in range(self.board_size)]

        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 1 :
                    curr_feature[i][j] = 1
                elif self.board[i][j] == -1 :
                    oppo_feature[i][j] = 1
                elif self.board[i][j] == 0:
                    available_feature[i][j] = 1

                if self.stage == 1:
                    stage_feature[i][j] = 1

        feature.append(curr_feature)
        feature.append(oppo_feature)
        feature.append(available_feature)
        feature.append(stage_feature)

        return feature

    def print_board(self):
        for i in range(self.board_size):
            for j in range(self.board_size):
                self.print_color_char(self.board[i][j])
            print("", end='\n')

    def print_pi(self):
        for i in range(self.board_size):
            for j in range(self.board_size):
                print("{:>{width}} ".format(self.pi[i][j], width=8), end='')
                # print("{} ".format(self.pi[i][j]), end='')
            print("", end='\n')

    def print_color_char(self, value : int):
        # ANSI转义码，用于控制终端文本颜色
        ANSI_RESET = "\x1B[0m"
        ANSI_YELLOW = "\x1B[33m"
        ANSI_GREEN = "\x1B[32m"
        ANSI_BLACK = "\x1B[30m"
        ANSI_WHITE = "\x1B[37m"

        if value == -1:
            print(ANSI_YELLOW + " w " + ANSI_RESET, end='')
        elif value == 1:
            print(ANSI_GREEN + " b " + ANSI_RESET, end='')
        else:
            print(ANSI_WHITE + " - " + ANSI_RESET, end='')


class GameState:
    """ 一局游戏状态
    """
    def __init__(self, game: list, game_win_state: int, random_step: int) -> None:
        """ 初始
            - game : 保存一局游戏所有棋盘, 三维坐标
            - game_win_state : 胜负信息
            - random_step : 随即开局的步数, 训练时不要
        """
        # 一局游戏
        self.game = game
        # 棋局胜负状态
        # 黑 1，空 0，白 -1
        self.game_win_state = game_win_state
        # 随机步数
        self.random_step = random_step

"""
{
	"game":	[{
			"a":	160,
			"q":	1,
			"ap":	[[160, 0.999261]]
		}, {
			"a":	142,
			"q":	1,
			"ap":	[[142, 0.999949], [233, 0.037037]]
		}],
	"game_win_state":	-1,
	"random_step":	8
}
"""
def deserialize_from_json(file_name: str) -> GameState:
    # 自定义反序列化函数
    def deserialize(game_data):
        # game_data = json.loads(json_data)
        board_size = 19
        # print(game_data)
        game_list = []
        game_win_state = game_data["game_win_state"]
        random_step = game_data["random_step"]
        game_arr = game_data["game"]
        board = [[0 for _ in range(board_size)] for i in range(board_size)]

        for i in range(len(game_arr)):
            curr_player = 0
            stage = 1 if i % 2 == 0 else 0
            board_data = game_arr[i]
            
            action = board_data["a"]
            q = board_data["q"]
            ap_list = board_data["ap"]

            x = action // board_size
            y = action % board_size
            if ((i + 1) // 2) % 2 == 0:
                board[x][y] = 1
                curr_player = 1
            else:
                board[x][y] = -1
                curr_player = -1

            pi = [[0 for _ in range(board_size)] for i in range(board_size)]
            for ap in ap_list:
                ap_a = ap[0]
                ap_pi = ap[1]
                xa = ap_a // board_size
                ya = ap_a % board_size
                pi[xa][ya] = ap_pi
            
            z = game_win_state if curr_player == game_win_state else -game_win_state
            z = (z + q) / 2

            board_state = BoardState(copy.deepcopy(board), pi, q, z, curr_player, stage)
            game_list.append(board_state)
        
        game_state = GameState(game_list, game_win_state, random_step)
        return game_state

    # 从本地 json 文件反序列化数据到 game_state_t 数据结构
    with open(file_name, 'r') as file:
        return deserialize(json.load(file))


# game_state = deserialize_from_json("conn6_20231224193935193.json")
# print(game_state.game[9].print_board())
# print(game_state.game[9].print_pi())
# print(game_state.game[9].curr_player)

# print(game_state.game[9].get_feature())

# print(game_state.game_win_state)
# print(game_state.random_step)

# for game in game_state.game:
#     print(game.print_board())
#     print(game.curr_player)
#     print(game.q)

# # 解析 JSON 数据
# game_state = parse_json(data)
# print(game_state.game[1].print_board())
# print(game_state.game[1].print_pi())
# print(game_state.game[1].curr_player)
# for i in range(2):
#     if ((i + 1) // 2) % 2 == 0:
#         print("B")
#     else:
#         print("W")

