import GAN as g
import torch as t
import EncodersDecoders
import MPI_preprocessing
from EncodersDecoders import BERT,TemporalBlock,TemporalConvNet,Chomp1d
from torch.utils.data import Dataset,DataLoader,IterableDataset,sampler,SequentialSampler
import math
import os
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm




class DataSet(Dataset):
  def __init__(self, list_IDs,path):
        self.list_IDs = list_IDs
        self.path = path

  def __len__(self):
        return len(self.list_IDs)

  def __getitem__(self, index):
        ID = self.list_IDs[index]
        X = t.load(self.path + ID)
        return X
def Train_TCN_Embeddings(bvh_dataloader):
    tcn_autoencoder = EncodersDecoders.TemporalConvNet(72,[72,56,38,56,72],3).cuda()
    optimizer_tcn_autoencoder = t.optim.Adam(tcn_autoencoder.parameters())
    tcn_reconstruction_l = t.nn.MSELoss().cuda()
    for d in bvh_dataloader:
        optimizer_tcn_autoencoder.zero_grad()
        bvh_pose = d.cuda()
        tcn_embedding = tcn_autoencoder(bvh_pose)[0]
        tcn_reconstruction_loss = tcn_reconstruction_l(bvh_pose,tcn_embedding)
        tcn_reconstruction_loss.backward()
        optimizer_tcn_autoencoder.step()
    return tcn_autoencoder

def Train_PoseEncoder(batch_size,epochs,tcn_dataloader,bvh_10,seq_l,embed_size,real_emotions):
    pose_encoder = EncodersDecoders.PoseEncoder().cuda()
    pose_decoder = EncodersDecoders.PoseDecoder().cuda()
    emotion_decoder = EncodersDecoders.EmotionDecoder().cuda()
    optimizer_pose_encoder = t.optim.Adam(pose_encoder.parameters())
    optimizer_pose_decoder = t.optim.Adam(pose_decoder.parameters())
    optimizer_emotion_decoder = t.optim.Adam(emotion_decoder.parameters())
    pose_reconstruction_l = t.nn.MSELoss().cuda()
    emotion_classification_l = t.nn.CrossEntropyLoss().cuda()
    #pose_encoder_loss = pose_reconstruction_l + emotion_classification
    seq_index = 0

    for d in tcn_dataloader:
        bvh_pose = t.transpose(d, 0, 1)
        bvh_pose = t.transpose(bvh_pose, 0, 2).cuda()
        bvh_pose.requires_grad = True
        for ii in range(0,len(bvh_10)-batch_size,batch_size):
            real_pose = t.transpose(bvh_10[ii:(ii+batch_size)], 0, 1)
            real_pose = t.transpose(real_pose, 0, 2)
            bvh_pose2 = bvh_pose.clone()
            bvh_pose3 = bvh_pose.clone()
            tot_seq_l = bvh_pose.shape[1]
            for n in range(0,tot_seq_l,batch_size):
                for i in range(math.ceil(tot_seq_l/seq_l)):
                    seq_index += seq_l
                    if seq_index > tot_seq_l:
                        seq_index = tot_seq_l
                    optimizer_pose_decoder.zero_grad()
                    optimizer_emotion_decoder.zero_grad()
                    optimizer_pose_encoder.zero_grad()
                    pose_embedding = pose_encoder(bvh_pose)
                    pose_embedding2 = pose_embedding.clone()
                    pose_reconstruction = Train_PoseDecoder(pose_embedding,real_pose,pose_decoder)
                    pose_reconstruction_loss = pose_reconstruction_l(pose_reconstruction, real_pose)
                    pose_reconstruction_loss.backward(retain_graph=True)
                    optimizer_pose_decoder.step()
                    real_emotions2 = real_emotions[n:n + batch_size, :].clone()
                    real_emotions3 = real_emotions2.clone()
                    emotion_output = Train_EmotionDecoder(pose_embedding2,real_emotions2,emotion_decoder)
                    #emotion_classification = t.max(emotion_output,dim=-1,keepdim=True)[0]
                    #emotion_classification2 = emotion_classification.clone()
                    emotion_classification_loss = emotion_classification_l(emotion_output,real_emotions3)
                    emotion_classification_loss.backward()
                    #pose_encoder_loss.backward()
                    #optimizer_pose_encoder.step()
                    optimizer_emotion_decoder.step()
    asds = 2
    return pose_encoder,pose_decoder
def Train_PoseDecoder(pose_embedding,real_poses,pose_decoder,*args):
    if args:
        word_embeddings = args[0]
        outp = pose_decoder(word_embeddings,real_poses)
    else:
        outp = pose_decoder(pose_embedding,real_poses)
    return outp
def Train_EmotionDecoder(pose_embedding,real_emotions,emotion_decoder,*args):
    if args:
        word_embeddings = args[0]
        outp = emotion_decoder(word_embeddings,real_emotions)
    else:
        outp = emotion_decoder(pose_embedding,real_emotions)

    return outp

def Train_GAN(batch_size,epochs,BERT_sents_padded,mocap_data_padded,seq_l,embed_size):
    model = g.GAN(input=seq_l, hidden=mocap_l,out=mocap_coords).cuda()
    optimizer = t.optim.Adam(model.parameters())
    loss = t.nn.MSELoss()
    for n in range(0,epochs,batch_size):
            optimizer.zero_grad()
            seed = t.randn((seq_l,batch_size,embed_size)).cuda()
            outp = model.transformer(BERT_sents_padded[:seq_l,n:(n+batch_size),:embed_size],seed)
            loss2 = loss(outp, mocap_data_padded[:seq_l,n:(n+batch_size),:embed_size])
            print(loss2.item())
            loss2.backward()
            optimizer.step()
            seed = outp
    return model
device = t.cuda.device
write_sentence = False
write_BERT_embedding = False
write_BVH = False
write_TCN = False
pred_bool = True
n_samples = 355
seq_l = 37
batch_size = 2
total_seq_l = 1258
feat_n = 72
seq_n = int(total_seq_l/seq_l)
bvh_files_n = n_samples*seq_n
bvh_path = "C:\\Users\\jordy\\Documents\\MPI\\bvh\\"
animation_path = "C:\\Users\\jordy\\Documents\\MPI\\AnimationEmbeddings\\"
sentence_path = "C:\\Users\\jordy\\PycharmProjects\\sentstrain.txt"
t_path = "C:\\Users\\jordy\\Documents\\MPI\\text_MPI.txt"
BERT_path = "C:\\Users\\jordy\\Documents\\MPI\\BERT\\"
tcn_path = "C:\\Users\\jordy\\Documents\\MPI\\TCN\\"
pred_path = "C:\\Users\\jordy\\Documents\\MPI\\Predictions\\"
pre = MPI_preprocessing.Preprocessing()
data = pre.PrepareMPI(t_path).train_dict
sents = pre.ListToString()
IDx = ["ID-"+str(i) for i in range(bvh_files_n)]
IDx2 = ["ID-"+str(i) for i in range(26*seq_n)]
if write_BVH == True:
        mocap_data_padded = pre.BVHdata(bvh_path, n_samples,768,seq_l)
        pre.WriteEmbeddingPT(animation_path,mocap_data_padded, IDx,seq_l,n_samples,seq_n)
if write_sentence == True:
    pre.WriteSentencetxt(sentence_path,sents)
seq_l = 37
sample_n = n_samples
embed_size = 768
if write_BERT_embedding ==  True:
    with open(sentence_path,"r") as f:
        BERT_sents_padded = BERT(f.read().splitlines()).BERT_sents_padded
        pre.WriteEmbeddingPT(BERT_path,BERT_sents_padded,IDx)
emo = [data["Intended emotion"][0:n_samples] for x in range(seq_l)]
real_emotions = t.transpose(t.LongTensor(emo).cuda(),0,1)
if __name__ == '__main__':
    params = {'batch_size': 2,
              'shuffle': True,
              'num_workers': 4}
    bvh_dataset = DataSet(IDx,animation_path)
    bvh_dataloader = DataLoader(bvh_dataset,**params)
    tcn_embedding_model = Train_TCN_Embeddings(bvh_dataloader)
    if write_TCN == True:

        tcn_embeddings = t.zeros(26,38,total_seq_l).float()
        tt = 0
        for b_sample in bvh_dataloader:
            tt += 1
            if tt >= 26:
                break
            for i in range(1,8+1):
                    batch_next_batch = slice((i - 1) * batch_size,(i * batch_size))
                    for k in range(1, seq_n):
                        tcn_embeddings[batch_next_batch,:feat_n,(k-1)*seq_l:k*seq_l] = tcn_embedding_model(b_sample.cuda())[1].cpu()
            aw = 2
        pre.WriteEmbeddingPT(tcn_path, tcn_embeddings, IDx2,seq_l,26 ,seq_n)
    tcn_dataset = DataSet(IDx2,tcn_path)
    tcn_dataloader = DataLoader(tcn_dataset,**params)
    if pred_bool == True:
        bvh_10 = t.zeros(10,72,37)
        for n in range(10):
            bvh_10[n,:,:] = bvh_dataset.__getitem__(n)
        bvh_10 = bvh_10.float().cuda()
        pose_encoder,pose_decoder = Train_PoseEncoder(2,10,tcn_dataloader,bvh_10,seq_l,embed_size,real_emotions)
        tcn_embed = t.transpose(tcn_embedding_model(bvh_10)[1],1,2)
        pose_encode = pose_encoder(tcn_embed)
        pose_decode = pose_decoder(pose_encode,bvh_10)
        t.save(pose_decode.state_dict(),pred_path)
    pose_decode_model = t.load(pred_path)
    pose_predictions = pose_reconstruction(bvh_10)
    abc = 2
    writer = pre.BVHWriter()
    pose_sequence = writer.write(pred_path)
