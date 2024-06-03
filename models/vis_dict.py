import torch
import torch.nn as nn
import torch.distributed as dist


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def sum_inplace(sum_data, new):
    sum_data.data.add_(new)


def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


def ema_tensor_inplace(moving_avg, new, decay):
    new_out = torch.mul(new, 1.0 - decay)
    moving_avg.data.mul_(decay).add_(new_out.detach())


class VisualDict(nn.Module):
    def __init__(self, num_tokens, token_dim, decay=0.1, max_decay=0.99, eps=1e-5) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.decay = decay
        self.cur_decay = decay
        self.max_decay = max_decay
        self.eps = eps

        embed = torch.randn(num_tokens, token_dim)
        self.register_buffer("embed", embed)
        nn.init.normal_(self.embed)
        # embed = torch.normal(num_tokens, token_dim)
        # self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(num_tokens))
        self.register_buffer("cluster_sum", torch.zeros(num_tokens))
        self.register_buffer("embed_avg", torch.zeros(num_tokens, token_dim))

    def set_decay_updates(self, num_update) -> None:
        self.cur_decay = min(self.cur_decay * num_update, self.max_decay)

    def forward(self, inputs_flatten: torch.Tensor):
        distances = (torch.sum(inputs_flatten**2, dim=1, keepdim=True) + torch.sum(self.embed.data**2, dim=1) -
                     2 * torch.matmul(inputs_flatten, self.embed.data.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_tokens, dtype=torch.float, device=inputs_flatten.device)
        encodings.scatter_(1, encoding_indices, 1)
 
        if self.training:
            tmp_sum = torch.sum(encodings, dim=0, keepdim=True)
            encoding_sum = torch.sum(concat_all_gather(tmp_sum), dim=0)

            # encoding_sum = tmp_sum.squeeze()
            sum_inplace(self.cluster_sum, encoding_sum)
            ema_tensor_inplace(self.cluster_size, encoding_sum, self.cur_decay)
            embed_sum_tmp = torch.matmul(encodings.t(), inputs_flatten)

            embed_sum = torch.sum(concat_all_gather(embed_sum_tmp.unsqueeze(dim=0)), dim=0)
            # embed_sum = embed_sum_tmp
            ema_tensor_inplace(self.embed_avg, embed_sum, self.cur_decay)

            # 下一步加的拉普拉斯平滑就是可能会出现维度灾难
            cluster_size = laplace_smoothing(self.cluster_size, self.num_tokens, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)

            # self.embed.data.copy_(embed_normalized)
            world_size = dist.get_world_size()
            dist.all_reduce(embed_normalized.div_(world_size))
            self.embed.data.copy_(embed_normalized)

        # 这步相乘起到了选择的作用，encodings是用encoding_indices
        # 构造出来的mask,其与self.embed相乘选择出了mask所标的embed
        # 其实就是式（3）（soho论文编号）
        quantize = torch.matmul(encodings, self.embed)
        # 这步是实现论文中式（5）（soho论文编号）
        quantize = (quantize - inputs_flatten).detach() + inputs_flatten

        return quantize, encoding_indices
