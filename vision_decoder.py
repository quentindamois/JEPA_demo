import torch
import torch.nn as nn
import numpy as np
from functools import reduce

class vision_decoder(torch.nn.Module):
    def __init__(self, input_dim:tuple[int], output_dim:tuple[int], dropout:float=0.1) -> None:
        super().__init__()
        self.has_one_channel = len(input_dim) == 2
        kernel_l1 = (int(input_dim[-2]//4), int(input_dim[-1]//4))
        self.cnn_layer_l1 = nn.ConvTranspose2d(16, 1 if self.has_one_channel else input_dim[0], kernel_size=kernel_l1)
        list_h_out_l1 = tuple(list(map(lambda a: a[0] - a[1] + 1 , zip(input_dim[-2:], kernel_l1))))
        kernel_l2 = (int(list_h_out_l1[-2]//4), int(list_h_out_l1[-1]//4))
        self.cnn_layer_l2 = nn.ConvTranspose2d(8, 16, kernel_size=kernel_l2)
        list_h_out_l2 = tuple(list(map(lambda a: a[0] - a[1] + 1 , zip(list_h_out_l1, kernel_l2))))
        kernel_l3 = (int(list_h_out_l2[-2]//4), int(list_h_out_l2[-1]//4))
        self.cnn_layer_l3 = nn.ConvTranspose2d(4, 8, kernel_size=kernel_l3)
        list_h_out_l3 = tuple(list(map(lambda a: a[0] - a[1] + 1 , zip(list_h_out_l2, kernel_l3))))
        cnn_out_dim = reduce(lambda a, b: a * b, list_h_out_l3, 4)
        self.ln_out = nn.Linear(output_dim, cnn_out_dim)
        self.act_fun = nn.ReLU()
        self.do = nn.Dropout(dropout)
        self.cnn_out_dim = cnn_out_dim
        self.img_dim = input_dim


    def forward(self, x):
        x_size = x.size()
        

        tem = self.ln_out(x)
        tem = self.act_fun(tem)
        dim_input = int(np.sqrt(self.cnn_out_dim/4))
        tem = tem.view([x_size[0] * x_size[1]] + [4, dim_input, dim_input])
        tem = self.cnn_layer_l3(tem)
        tem = self.act_fun(tem)
        tem = self.do(tem)

        tem = self.cnn_layer_l2(tem)
        tem = self.act_fun(tem)
        tem = self.do(tem)

        tem = self.cnn_layer_l1(tem)
        res = self.act_fun(tem)
        res = res.view([x_size[0],  x_size[1]] + list(self.img_dim))
        return res