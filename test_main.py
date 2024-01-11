import torch
import torch.nn as nn

for i in range(2):
    print(19 // (3 * (i + 1)))
    print(19 // (3 * 2))
    print(19 // (3 * (i + 1)))
    print("----------")

for i in range(4):
    print(224 // (2 ** (i + 2)))
    print(224 // (2 ** (4 + 1)))
    print(224 // (2 ** (i + 2)))
    print("========")