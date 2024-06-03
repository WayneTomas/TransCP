import torch
import torch.nn as nn
import torch.nn.functional as F
from util.misc import NestedTensor


class LSTMBert(nn.Module):
    def __init__(self,
                 lstm_cfg=dict(type='gru',
                               num_layers=1,
                               dropout=0.,
                               hidden_size=512,
                               bias=True,
                               bidirectional=True,
                               batch_first=True),
                 output_cfg=dict(type="max"),
                 train_bert=True,
                 enc_num=12,
                 bert_model=None):
        super(LSTMBert, self).__init__()
        self.enc_num = enc_num
        self.fp16_enabled = False

        assert lstm_cfg.pop('type') in ['gru']
        self.lstm = nn.GRU(**lstm_cfg, input_size=768)

        output_type = output_cfg.pop('type')
        assert output_type in ['mean', 'default', 'max']
        self.output_type = output_type

        self.bert = bert_model

        if not train_bert:
            for parameter in self.bert.parameters():
                parameter.requires_grad_(False)
        self.num_channels = 768
        # num_dirs指的是lstm的方向，=2就是双向
        if lstm_cfg["bidirectional"]:
            self.num_dirs = 2
        else:
            self.num_dirs = 1
        self.sub_attn = PhraseAttention_Liu(self.lstm.hidden_size * self.num_dirs)

    def forward(self, ref_expr_inds: NestedTensor):
        """Args:
            ref_expr_inds (tensor): [batch_size, max_token], 
                integer index of each word in the vocabulary,
                padded tokens are 0s at the last.
        Returns:
            y (tensor): [batch_size, 1, C_l].
            y_word (tensor): [batch_size, max_token, C_l].
            y_mask (tensor): [batch_size, max_token], dtype=torch.bool, 
                True means ignored position.
        """
        # Eq. (2)
        sequence_output, pooled_output = self.bert(ref_expr_inds.tensors, attention_mask=ref_expr_inds.mask)
        xs = sequence_output[self.enc_num - 1]

        y_mask = torch.abs(ref_expr_inds.tensors) == 0

        # Eq. (6)
        y_word, h = self.lstm(xs)

        # Eq. (7) ~ Eq. (8)
        # 测试使用phrase attn
        sub_attn, sub_phrase_emb = self.sub_attn(y_word, xs, y_mask)
        y = sub_phrase_emb
        
        return  (y, NestedTensor(xs, y_mask))


class PhraseAttention_Liu(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, context, embedded, mask):
        cxt_scores = self.fc(context).squeeze(2)

        attn = F.softmax(cxt_scores, dim=-1)

        attn = attn * ((~mask).float())  # (batch, seq_len)

        attn = attn / attn.sum(1).view(attn.size(0), 1).expand(attn.size(0), attn.size(1))  # (batch, seq_len)

        attn3 = attn.unsqueeze(1)
        weighted_emb = torch.bmm(attn3, embedded)

        return attn, weighted_emb


def build_LSTMBert(args, bert_model):
    # 注意，该函数里除了本注释外，皆为原作者所加
    train_bert = args.lr_bert > 0
    lstm_bert = LSTMBert(enc_num=args.bert_output_layers, train_bert=train_bert, bert_model=bert_model)
    return lstm_bert