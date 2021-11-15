# coding:utf-8
"""
Filename: device.py
Author: @DvdNss

Created on 11/15/2021
"""

import torch

if __name__ == "__main__":
    print("GPU" if torch.cuda.is_available() else "CPU")
