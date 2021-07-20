import torch as t
class GAN(t.nn.Module):
    def __init__(self, input,hidden,out):
        super(GAN, self).__init__()
        self.transformer = t.nn.Transformer(d_model=768,nhead=8,num_encoder_layers=6,num_decoder_layers=6,dim_feedforward=2048)

    def forward(self,input,seed):
        self.transformer = t.nn.Transformer(input,seed)
        return x