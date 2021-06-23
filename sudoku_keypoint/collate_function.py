import torch


def CF(batch):
    input_list, mask_list, mask2_list, key_list = [], [], [], []
    for (_input, _mask, _mask2, _key) in batch:
        input_list.append(_input)
        mask_list.append(_mask)
        mask2_list.append(_mask2)
        key_list.append(_key)

    input_list = torch.stack(input_list)
    mask_list = torch.stack(mask_list)
    mask2_list = torch.stack(mask2_list)
    key_list = torch.stack(key_list)
    return input_list, mask_list, mask2_list, key_list
