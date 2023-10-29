import torch
from torch import nn
from torch.nn import functional as F
import math



class SelfAttention(nn.Module):
    """
     Computes self attention between all pixels in image
    """
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # This combines the Wq, Wk and Wv matrices into one matrix
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias) # *3 for k, q, v
        # This one represents the Wo matrix
        self.out_proj = nn.Linear(d_embed,  d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads # dimension of 1 head = tot size embedding / n_heads


    def forward(self, x: torch.Tensor, causal_mask=False):
        # x: (Batch_size, Seq_len, Dim) 
        # where seq_len is h * w in convolution and dim is the numbear of feature of a conv block

        input_shape = x.shape
        
        batch_size, sequence_length, d_embed = input_shape
        
        intermim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # create q, k, v by applying a nn.linear of dim 3 * input dim
        # then split in chunks 
        # (batch_size, Seq_len, Dim) -> (
        #   (batch_size, Seq_len, Dim)
        #   (batch_size, Seq_len, Dim),
        #   (batch_size, Seq_len, Dim),
        # )
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # reshape k, q and v to prepare attention computation,  
        # by splitting by head (just after batch position)
        # (Batch_size, seq_len, Dim) - -> (Batch_size, d_head, seq_len,  d_head) 
        #    can be written as       - -> (Batch_size, H, seq_len, Dim/H)
        q = q.view(intermim_shape).transpose(1, 2)
        k = k.view(intermim_shape).transpose(1, 2)
        v = v.view(intermim_shape).transpose(1, 2)

        # (batch_size, H, Seq_len, Seq_len)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            # Mask where the upper triangle (above principal diag) is made up of ones
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        # (batch_size, H, Seq_len, Seq_len) @ (Batch_size, H, seq_len, Dim/H)
        # -> (Batch_size, H, seq_len, Dim/H)
        output = weight @ v

        # -> (Batch_size, seq_len,  H, Dim/H)
        output = output.transpose(1, 2)
        
        # reconcatenate heads
        # -> input shape, ie (Batch_size, Seq_len, Dim) 
        output = output.reshape(input_shape)

        # reproject with trainable w0 weigths
        output = self.out_proj(output)

        # same as input shape -> (Batch_size, Seq_len, Dim)
        return output