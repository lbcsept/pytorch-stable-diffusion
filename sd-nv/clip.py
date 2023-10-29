import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):

    def __init__(self, n_vocab:int=49408, n_embd:int=768, 
                 n_tokens:int=77):
        """Clip embeddings, including postion embeddings which where 
        learnt and will be loaded when loading CLIP weights 

        Args:
            n_vocab (int, optional): Vocabulary size. Defaults to 49408.
            n_embd (int, optional): dimension of the tokens. Defaults to 768.
            n_tokens (int, optional): length of a sequence. Defaults to 77.
    
       """
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        # here position embedding is learnt, not like in attention is all you need
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embd))

    def forward(self, tokens: torch.LongTensor) :

         # tokens:(batch_size, n_tokens) -> (batch_size, n_tokens, n_embd) 
        x = self.token_embedding(tokens)
        
        x += self.position_embedding

        return x

class CLIPLayer(nn.Module):

    def __init__(self, n_head:int=12, n_embd:int=768):
        """One CLIP layer (basically a kind of transformer block)

        Args:
            n_head (int, optional): number of heads. Defaults to 12.
            n_embd (int, optional): dimension of the tokens. Defaults to 768.
        """
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.linear(4 * n_embd, n_embd)
    
    def forward(self, x:torch.tensor) -> torch.Tensor:
        #x: (batch_size, n_tokens, n_embd)

        residue = x

        ## Self Attention

        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residue


        ## FEEDFORWAD LAYER

        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        # QuickGELU activation function
        x = x * torch.sigmoid(1.702 * x)
        x = self.linear_2(x)

        x += residue

        return x

class CLIP(nn.Module):

    def __init__(self, n_vocab:int=49408, n_embd:int=768, 
                 n_tokens:int=77, n_head:int=12, nb_clip_layers=12):
        """Contrastive Language–Image Pre-training modele
            https://openai.com/research/clip
        Args:
            n_vocab (int, optional): Vocabulary size. Defaults to 49408.
            n_embd (int, optional): dimension of the tokens. Defaults to 768.
            n_tokens (int, optional): length of a sequence. Defaults to 77.
            n_head (int, optional): number of heads. Defaults to 12.
            nb_clip_layers (int, optional): number of CLIP layers. Defaults to 12.
        """
        self.embedding = CLIPEmbedding(n_vocab, n_embd, n_tokens)

        self.layers = nn.Module([
            CLIPLayer(n_head, n_embd) for i in range(nb_clip_layers)
        ])

        self.layernorm = nn.LayerNorm(n_embd)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        
        tokens = tokens.type(torch.long)
        
        # (batch_size, n_tokens) -> (batch_size, n_tokens, n_embd) 
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)
    
        # (batch_size, n_tokens, n_embd)
        output = self.layernorm(state)

        return output