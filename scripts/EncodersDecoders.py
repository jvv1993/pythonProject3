from transformers import BertTokenizer,BertModel
import torch as t
from torch.nn.utils import weight_norm

class Chomp1d(t.nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(t.nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(t.nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = t.nn.ReLU()
        self.dropout1 = t.nn.Dropout(dropout)

        self.conv2 = weight_norm(t.nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = t.nn.ReLU()
        self.dropout2 = t.nn.Dropout(dropout)

        self.net = t.nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = t.nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = t.nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        outp = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(outp + res)


class TemporalConvNet(t.nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = t.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x),self.network[:3](x)
class PoseEncoder(t.nn.Module):
    def __init__(self):
        super(PoseEncoder,self).__init__()
        self.encoderlayer = t.nn.TransformerEncoderLayer(d_model=38, nhead=2).cuda()
        self.encoder = t.nn.TransformerEncoder(self.encoderlayer, 6).cuda()
        self.linear = t.nn.Linear(38,72)


    def forward(self, pose):
        x = self.encoder(pose)
        x = self.linear(x)
        return x


class PoseDecoder(t.nn.Module):
    def __init__(self):
        super(PoseDecoder, self).__init__()
        self.decoderlayer = t.nn.TransformerDecoderLayer(d_model=72, nhead=9).cuda()
        self.decoder = t.nn.TransformerDecoder(self.decoderlayer, 6).cuda()

    def forward(self, pose_embedding,real_poses):
        return self.decoder(pose_embedding,real_poses)


class EmotionDecoder(t.nn.Module):
    def __init__(self):
        super(EmotionDecoder, self).__init__()
        self.pose_emo_embeddinglayer = t.nn.TransformerEncoderLayer(d_model=72, nhead=9).cuda()
        self.pose_emo_embedding = t.nn.TransformerEncoder(self.pose_emo_embeddinglayer, 6).cuda()
        self.Linear = t.nn.Linear(72,11)
        self.Relu = t.nn.ReLU()
        self.Softmax = t.nn.Softmax(dim=1)

    def forward(self, pose_embedding,real_emotions):
        x = self.pose_emo_embedding(pose_embedding)
        x = self.Linear(x)
        x = self.Relu(x)
        x = self.Softmax(x)
        x = t.transpose(t.transpose(x,0,1),1,2)
        return x

class BERT(t.nn.Module):
    def __init__(self,sentences):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        BERT_sents = []
        for sent in sentences:
            input = self.tokenizer(sent,return_tensors="pt")
            output = self.model(**input)
            outp = output.last_hidden_state
            BERT_sents.append(t.squeeze(outp,0))
        self.BERT_sents_padded = t.transpose(t.nn.utils.rnn.pad_sequence(BERT_sents).float(),0,1)
