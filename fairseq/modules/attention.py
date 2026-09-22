
import torch
import torch.nn as nn

class ScaledAttention(nn.Module):
    # attention type can be LBL or dot
    def __init__(self, dim, scale=1.0, att_type='lbl'):
        super(ScaledAttention, self).__init__()
        self.attention_type = att_type
        if self.attention_type == 'lbl':
            self.trans = nn.Linear(dim, dim, bias=False)
        print('attention type is', self.attention_type)
        self.softmax = nn.LogSoftmax(dim=-1)
        # self.scale = dim**-0.5
        self.scale = scale

    def forward(self, input, context, mask=None):
        """
        input: batch x targetL x dim
        context: batch x sourceL x dim
        """
        if self.attention_type == 'lbl':
            target = self.trans(input)  # batch x targetL x dim
        else:
            target = input

        target = target * self.scale

        # Get attention
        attn_ = torch.bmm(target, context.transpose(1, 2))  # batch x targetL x sourceL
        if mask is not None:
            attn_.masked_fill_(mask.unsqueeze(1), float(-100))
        attn = self.softmax(attn_) # batch x targetL x sourceL
        weighted_context = torch.bmm(attn.exp(), context) # batch x targetL x dim

        return weighted_context, attn_
