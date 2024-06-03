import torch
import torch.nn.functional as F
from torch import nn
from models.vis_dict import VisualDict


class PrototypeLearner(nn.Module):
    def __init__(self, num_tokens=2048, token_dim=768, hidden_dim=256, decay=0.1, max_decay=0.99):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.token_dim = token_dim
        self.prototype = VisualDict(num_tokens=num_tokens, token_dim=token_dim, decay=decay)
        self.conv_head = nn.Conv2d(hidden_dim, token_dim, kernel_size=1, bias=False)
        self.conv_tail = nn.Conv2d(token_dim, hidden_dim, kernel_size=1, bias=False)
        self.pos_align = True
        if self.pos_align:
            self.pos_line = nn.Linear(token_dim, token_dim)

        # 处理
        self.gate_fc = nn.Conv2d(token_dim * 2, 2, kernel_size=1, stride=1, bias=False)

    def forward(self, src: torch.tensor, h: int, w: int):
        num_visu_token, bs, hidden_dim = src.size()
        src = src.permute(1, 2, 0).contiguous().view(bs, -1, h, w)
        xq = self.conv_head(src)
        inputs_flatten = xq.view(-1, self.token_dim)
        xq_img = xq

        quantized_pt, indices = self.prototype(inputs_flatten)
        if self.pos_align:
            quantized_pt = self.pos_line(quantized_pt)
        embedded_pt = quantized_pt.view(bs, num_visu_token, quantized_pt.size(-1))
        embedded_pt = embedded_pt.permute(0, 2, 1).contiguous().view(bs, -1, h, w)

        tmp_feat = torch.cat([embedded_pt, xq_img], dim=1)
        tmp_score = F.softmax(self.gate_fc(tmp_feat), dim=1)

        emb_score = tmp_score[:, 0, :, :].unsqueeze(dim=1)
        img_score = tmp_score[:, 1, :, :].unsqueeze(dim=1)
        embedded_pt = embedded_pt * emb_score + xq_img * img_score

        embedded_pt = self.conv_tail(embedded_pt)
        return {"embedded_pt": embedded_pt.flatten(2).permute(2, 0, 1).contiguous(), "indices": indices}