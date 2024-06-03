import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_pretrained_bert.modeling import BertModel
from transformers import BertModel
from util.misc import (NestedTensor, get_world_size, is_dist_avail_and_initialized)
# from transformers import AutoConfig
# from transformers import AutoConfig, AutoModel
# from transformers.models import BertModel



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

        # assert len(word_emb) > 0
        # lstm_input_ch = word_emb.shape[-1]
        # self.embedding = nn.Embedding.from_pretrained(
        #     torch.from_numpy(word_emb),
        #     freeze=freeze_emb,
        # )

        assert lstm_cfg.pop('type') in ['gru']
        self.lstm = nn.GRU(**lstm_cfg, input_size=768)

        output_type = output_cfg.pop('type')
        assert output_type in ['mean', 'default', 'max']
        self.output_type = output_type

        # self.config = AutoConfig.from_pretrained("./checkpoints/bert-base-uncased.tar.gz")
        # self.config.hidden_dropout_prob = 0.
        # self.config.attention_probs_dropout_prob = 0.
        self.bert = bert_model
        # for v in self.bert.pooler.parameters():
        #     v.requires_grad_(False)
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
        sequence_output, pooled_output = self.bert(ref_expr_inds.tensors, attention_mask=ref_expr_inds.mask)
        xs = sequence_output[self.enc_num - 1]
        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        # lstm_output, (h,c) = self.lstm(sequence_output) ## extract the 1st token's embeddings
        # shidden = torch.cat((lstm_output[:,-1, :256],lstm_output[:,0, 256:]),dim=-1)

        y_mask = torch.abs(ref_expr_inds.tensors) == 0
        y_word, h = self.lstm(xs)

        # 测试使用phrase attn
        sub_attn, sub_phrase_emb = self.sub_attn(y_word, xs, y_mask)
        y = sub_phrase_emb
        
        # if self.output_type == "mean":
        #     y = torch.cat(list(map(lambda feat, mask: torch.mean(feat[mask, :], dim=0, keepdim=True), y_word,
        #                            ~y_mask))).unsqueeze(1)
        # elif self.output_type == "max":
        #     y = torch.cat(list(map(lambda feat, mask: torch.max(feat[mask, :], dim=0, keepdim=True)[0], y_word,
        #                            ~y_mask))).unsqueeze(1)
        # elif self.output_type == "default":
        #     h = h.transpose(0, 1)
        #     y = h.flatten(1).unsqueeze(1)
        return  (y, NestedTensor(xs, y_mask))


class PhraseAttention_Liu(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, context, embedded, mask):
        cxt_scores = self.fc(context).squeeze(2)

        attn = F.softmax(cxt_scores, dim=-1)

        # is_not_zero = (mask != 0).float()  # (batch, seq_len)
        attn = attn * ((~mask).float())  # (batch, seq_len)

        attn = attn / attn.sum(1).view(attn.size(0), 1).expand(attn.size(0), attn.size(1))  # (batch, seq_len)

        attn3 = attn.unsqueeze(1)
        weighted_emb = torch.bmm(attn3, embedded)
        # weighted_emb = weighted_emb.squeeze(1)

        return attn, weighted_emb


def build_LSTMBert(args, bert_model):
    # 注意，该函数里除了本注释外，皆为原作者所加
    # position_embedding = build_position_encoding(args)
    train_bert = args.lr_bert > 0
    lstm_bert = LSTMBert(enc_num=args.bert_output_layers, train_bert=train_bert, bert_model=bert_model)
    # model = Joiner(bert, position_embedding)
    # model.num_channels = bert.num_channels
    return lstm_bert