import torch
import torchvision
import torch.nn as nn
from functools import reduce

class vision_encoder(torch.nn.Module):
    def __init__(self, input_dim:tuple[int], output_dim:tuple[int], dropout:float=0.1) -> None:
        super().__init__()
        self.has_one_channel = len(input_dim) == 2
        kernel_l1 = (int(input_dim[-2]//4), int(input_dim[-1]//4))
        self.cnn_layer_l1 = nn.Conv2d(1 if self.has_one_channel else input_dim[0], 16, kernel_size=kernel_l1)
        list_h_out_l1 = tuple(list(map(lambda a: a[0] - a[1] + 1 , zip(input_dim[-2:], kernel_l1))))
        kernel_l2 = (int(list_h_out_l1[-2]//4), int(list_h_out_l1[-1]//4))
        self.cnn_layer_l2 = nn.Conv2d(16, 8, kernel_size=kernel_l2)
        list_h_out_l2 = tuple(list(map(lambda a: a[0] - a[1] + 1 , zip(list_h_out_l1, kernel_l2))))
        kernel_l3 = (int(list_h_out_l2[-2]//4), int(list_h_out_l2[-1]//4))
        self.cnn_layer_l3 = nn.Conv2d(8, 4, kernel_size=kernel_l3)
        list_h_out_l3 = tuple(list(map(lambda a: a[0] - a[1] + 1 , zip(list_h_out_l2, kernel_l3))))
        cnn_out_dim = reduce(lambda a, b: a * b, list_h_out_l3, 4)
        self.ln_out = nn.Linear(cnn_out_dim, output_dim)
        self.act_fun = nn.ReLU()
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        if self.has_one_channel:
            x = torch.unsqueeze(x, -3)
        x_size = x.size()
        x = x.view([x_size[0] *x_size[1]] + list(x_size[2:]))
        tem = self.cnn_layer_l1(x)
        tem = self.act_fun(tem)
        tem = self.do(tem)
        tem = self.cnn_layer_l2(tem)
        tem = self.act_fun(tem)
        tem = self.do(tem)
        tem = self.cnn_layer_l3(tem)
        tem = self.act_fun(tem)
        tem = self.do(tem)
        tem = self.ln_out(torch.flatten(tem, -3, -1))
        res = self.act_fun(tem)
        res = res.view(list(x_size[:2]) + [-1])
        return res


class embedding_predictor(torch.nn.Module):
    def __init__(self, dim_embedded_seq:int) -> None:
        super().__init__()
        self.predictor = nn.GRU(dim_embedded_seq, dim_embedded_seq, batch_first=True)
    def forward(self, x):
        res, _ = self.predictor(x)
        return res