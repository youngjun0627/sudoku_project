import torch


def CF(batch):
    input_list, sudoku_list = [], []
    for (_input, _sudoku) in batch:
        input_list.append(_input)
        sudoku_list.append(torch.tensor(_sudoku, dtype=torch.long))
    input_list = torch.stack(input_list)
    sudoku_list = torch.stack(sudoku_list)
    return input_list, sudoku_list
