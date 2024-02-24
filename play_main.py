import sys
import os
import time
import torch
import random

from backbone.resnet import resnet_10b128c, resnet_8b96c, resnet_6b128c, resnet_4b128c
from backbone.mobilevit_v1 import mobile_vit
from backbone.chess_transformer import chess_transformer_88_112
from connect6.board import Conn6Board
from connect6.alpha_zero_mcts import AlphaZeroMCTS

def load_pt_mode_weight(model_path, num_id : str):
    """ 加载神经网络和优化器参数
        - model_path : 模型路径
        - num_id : 模型id
    """
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    file_path = os.path.join(model_path + "/", "model_" + num_id + ".pth")
    if not os.path.exists(file_path):
        print ("No model in path {}".format(file_path))
        return None, None, None, None
    params = torch.load(file_path)
    return params["state_dict"], params["optimizer"], params["lr_param"], params["train_step_count"]

def load_jit_model_weight(model_path, num_id : str):
    pass

def pk_one_game(mcts_old:AlphaZeroMCTS, mcts_new:AlphaZeroMCTS):
    first_point = random.randint(0, 360)
    board = Conn6Board(board_size=19, n_feature_planes=18, first_point=first_point)
    win_flag = False
    while not win_flag:
        action_11 = mcts_old.get_action(board)
        board.do_action(action_11)
        win_flag, winner = board.is_game_over()
        if win_flag:
            return 1 if winner == board.WHITE else -1
        action_12 = mcts_old.get_action(board)
        board.do_action(action_12)
        win_flag, winner = board.is_game_over()
        if win_flag:
            return 1 if winner == board.WHITE else -1
        action_21 = mcts_new.get_action(board)
        board.do_action(action_21)
        win_flag, winner = board.is_game_over()
        if win_flag:
            return 1 if winner == board.WHITE else -1
        action_22 = mcts_new.get_action(board)
        board.do_action(action_22)
        win_flag, winner = board.is_game_over()
        if win_flag:
            return 1 if winner == board.WHITE else -1
        
def computer_model(model_old:torch.nn.Module(), model_new:torch.nn.Module(), pk_num:int):
    mcts_old = AlphaZeroMCTS(net_work = model_old, c_puct=1.0, n_iters=1)
    mcts_new = AlphaZeroMCTS(net_work = model_new, c_puct=1.0, n_iters=1)
    
    net_old_win = 0
    net_new_win = 0
    num = int(pk_num / 2)

    for i in range(num):
        result = pk_one_game(mcts_old, mcts_new)
        if result == 1:
            net_old_win += 1
        elif result == -1:
            net_new_win += 1

    for j in range(num):
        result = pk_one_game(mcts_new, mcts_old)
        if result == 1:
            net_new_win += 1
        elif result == -1:
            net_old_win += 1

    return net_old_win, net_new_win

def load_model_weight(model:torch.nn.Module, model_path:str):
    if not os.path.exists(model_path):
        print ("No model in path {}".format(model_path))
        return None
    model.load_state_dict(torch.load(model_path)["state_dict"])
    return model

def get_file_list(folder_path:str):
    # 获取文件夹中的文件名
    file_names = os.listdir(folder_path)

    # 筛选出以 'model_' 开头，以 '.pth' 结尾，且文件名中包含数字的文件
    selected_files = [file_name for file_name in file_names if file_name.startswith('model_') and file_name.endswith('.pth') and any(char.isdigit() for char in file_name)]

    # 按照顺序获取两两配对的文件的路径和文件名
    file_paths_and_names = []
    sorted_files = sorted(selected_files, key=lambda x: int(''.join(filter(str.isdigit, x))))
    for i in range(0, len(sorted_files), 5):
        file_name = sorted_files[i]
        file_path = os.path.join(folder_path, file_name)
        file_paths_and_names.append([file_name, file_path])
        # 每次取两个文件
        # pair_files = sorted_files[i:i + 2]
        # if len(pair_files) == 2:
        #     file_path_1 = os.path.join(folder_path, pair_files[0])
        #     file_path_2 = os.path.join(folder_path, pair_files[1])
        #     file_paths_and_names[(file_path_1, file_path_2)] = pair_files
            
    return file_paths_and_names

def save_csv_data(type_name : str, data: list, suffix:str='csv'):
    """ 记录损失
    """
    file_path = "."
    file_name = os.path.join(file_path, type_name + '.' + suffix)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    ans = ''
    with open(file_name, "a") as file_point:
        for i in data:
            ans = ans + str(i) + ','
        file_point.write(ans[:-1] + '\n')
        ans = ''

if __name__ == '__main__':
    in_chans = 18
    board_size = 19
    file_paths_and_names = get_file_list("logs/chess_transformer_88_112_inc18_softlabel/model")
    for i in range(21, len(file_paths_and_names)):
        file_name_1 = file_paths_and_names[i-1][0]
        file_path_1 = file_paths_and_names[i-1][1]
        file_name_2 = file_paths_and_names[i][0]
        file_path_2 = file_paths_and_names[i][1]
        # print(file_name_1 + "  " + file_name_2)

        model_old = chess_transformer_88_112(in_chans=in_chans, board_size=board_size) 
        model_new = chess_transformer_88_112(in_chans=in_chans, board_size=board_size) 

        model_old = load_model_weight(model_old, file_path_1)
        model_new = load_model_weight(model_new, file_path_2)

        net_old_win, net_new_win = computer_model(model_old, model_new, 20)
        print("{0} vs {1} : {2}-{3}".format(file_name_1, file_name_2, net_old_win, net_new_win))
        save_csv_data("elo_compute/chess_transformer_88_112_inc18_softlabel", [file_name_1, file_name_2, net_old_win, net_new_win])


    

if __name__ == '__main__2':
    in_chans = 18
    board_size = 19
    net_work = resnet_10b128c(in_chans=in_chans, board_size=board_size)
    model_weight, _, _ , _= load_pt_mode_weight("logs/resnet_10b128c_inc18_softlabel/model", "500")
    if model_weight is not None:
        net_work.load_state_dict(model_weight)
        print("Loading model parameters succeeded!")
    else:
        print("Failed to load model parameters.")

    conn6_board = Conn6Board(board_size=board_size, n_feature_planes=in_chans, first_point=180)
    search = AlphaZeroMCTS(net_work = net_work, c_puct=1.0, n_iters=1)

    conn6_board.print_board()

    win_flag = False
    while not win_flag:
        action_1 = search.get_action(conn6_board)
        conn6_board.do_action(action_1)
        flag, winner = conn6_board.is_game_over()
        if flag:
            win_flag = True
            print("wwww : ", winner, win_flag)
            break


        # action_2 = search.get_action(conn6_board)
        # conn6_board.do_action(action_2)

        conn6_board.print_board()
        
        # flag, winner = conn6_board.is_game_over()
        # if flag:
        #     win_flag = True
        #     print("wwww : ", winner, win_flag)
        #     break

    print("11111111111")

    