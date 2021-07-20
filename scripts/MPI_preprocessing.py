import numpy as np
import os
from collections import defaultdict
from pymo.parsers import BVHParser
import torch as t
from pymo.preprocessing import *
from sklearn.pipeline import Pipeline


class Preprocessing():
    def __init__(self):
        self.n_samples = 1451
        self.t_examples = []
        self.emotion_list = ["neutral","sadness","anger","pride","fear","amusement","relief","surprise","shame","joy","disgust"]
        self.e2i = defaultdict(lambda:len(self.e2i)) #Intended emotion
        self.p2i = defaultdict(lambda:len(self.p2i)) #Intended polarity
        self.ep2i = defaultdict(lambda:len(self.ep2i)) #Perceived emotion
        self.pp2i = defaultdict(lambda:len(self.pp2i)) #Perceived polarity
        self.w2i = defaultdict(lambda:len(self.w2i)) #Vocabulary
        self.a2i = defaultdict(lambda:len(self.a2i)) #actors
        self.i2w = dict()
        self.train_dict = {"Intended emotion": [], "Intended polarity": [], "Perceived emotion": [], "Perceived polarity": [], "Vocabulary": [], "Sentences": [[] for x in range(self.n_samples)], "Actors": [],"Duration": [], "Peaks": [], "Speed": [], "Span": [], "Gender": []}
        self.emo_pol = ["Intended emotion", "Intended polarity", "Perceived emotion", "Perceived polarity"]
        self.sent_intervals = ["Duration", "Peaks", "Speed", "Span"]
        self.intended_emo = []
        self.perceived_emo = []
        self.intended_pola = []
        self.perceive_pola = []
    def PrepareMPI(self,t_path):
        with open(t_path,"r") as f:
            feats = f.readlines()
            for j,f in enumerate(feats):
                f = f.split()
                f = f[3:]
                values = f[:4]
                i_list = []
                f = f[4:]
                f.insert(0,values)
                del f[1]
                del f[1]
                intervals = f[1:5]
                del f[1]
                del f[1]
                del f[1]
                del f[1]
                f.insert(1, intervals)
                del f[2]
                del f[2]
                del f[4]
                del f[4]
                del f[4]
                maxi = None
                mini = None
                for i,w in enumerate(f):
                    if ',' in w and w[:-1] in self.emotion_list:
                        i_list.append(i)
                        f[i] = f[i][:-1] #removing the comma's
                maxi = max(i_list) + 2
                mini = min(i_list)
                f[mini:maxi] = [f[mini:maxi]] #turning responses into a list of responses
                del[f[5]]
                self.train_dict["Sentences"][j] = " ".join(f[5:]).lower().split()
                f[mini+2:] = [" ".join(f[mini+2:]).lower().split()]#mini+2 is always the index where the sentence starts
                if f[3] == f:  #male is 0, female is 1
                    self.train_dict["Gender"].append(1)
                else:
                    self.train_dict["Gender"].append(0)
                self.train_dict["Actors"].append(self.a2i[f[2]])
                intervals = f[1]
                self.train_dict["Duration"].append(float(intervals[0]))
                self.train_dict["Peaks"].append(float(intervals[1]))
                self.train_dict["Speed"].append(float(intervals[2]))
                self.train_dict["Span"].append(float(intervals[3]))
                self.intended_emo.append(f[0][0])
                self.intended_pola.append(f[0][1])
                self.perceived_emo.append(f[0][2])
                self.perceive_pola.append(f[0][3])

        for x in self.intended_emo:
            self.train_dict["Intended emotion"].append(self.e2i[x])
        for x in self.intended_pola:
            self.train_dict["Intended polarity"].append(self.p2i[x])
        for x in self.perceived_emo:
            self.train_dict["Perceived emotion"].append(self.ep2i[x])
        for x in self.perceive_pola:
            self.train_dict["Perceived polarity"].append(self.pp2i[x])
        for s in self.train_dict["Sentences"]:
            for w in s:
                self.i2w[self.w2i[w]] = w
        self.train_dict["Vocabulary"] = self.i2w
        return self

    def ListToString(self):
        Sent_txt = ""
        for s in self.train_dict["Sentences"]:
            Sent_txt += " ".join(s) + "\n"
        self.Senttxt = Sent_txt
        return self.Senttxt

    @staticmethod
    def WriteSentencetxt(path,sents):
        with open (path, "w") as wr:
            wr.write(sents)
    @staticmethod
    def WriteEmbeddingPT(path,embeddings,IDx,seq_l,sample_n,seq_n):
        k = 0
        for n in range(sample_n):
             for i in range(1,seq_n+1):
                t.save(embeddings[n,:,(i-1)*seq_l:(i*seq_l)].detach(), path + IDx[k])
                k+=1

    def BVHdata(self,bvh_path,samples,embed_size,seq_l):
        data_pipes = []
        parser = BVHParser()
        for i,file in enumerate(os.listdir(bvh_path)[0:samples]):
                parsed_data = parser.parse(bvh_path+file)
                data_pipe = Pipeline([
                    ('param', MocapParameterizer('euler')),
                    ('delta', RootTransformer('pos_rot_deltas')),
                    ('np', Numpyfier())])
                piped_data = data_pipe.fit_transform([parsed_data])
                if piped_data.shape[0] > seq_l:
                    piped_data = piped_data[:seq_l,:]
                data_pipes.append(t.squeeze(t.from_numpy(piped_data),0))
        mocap_data_padded = t.transpose(t.nn.utils.rnn.pad_sequence(data_pipes).float(), 1, 2)
        mocap_data_padded = t.transpose(mocap_data_padded,0,2)
        return mocap_data_padded

    class BVHWriter():
        def __init__(self):
            pass

        def write(self, X, ofile):

            # Writing the skeleton info
            ofile.write('HIERARCHY\n')

            self.motions_ = []
            self._printJoint(X, X.root_name, 0, ofile)

            # Writing the motion header
            ofile.write('MOTION\n')
            ofile.write('Frames: %d\n' % X.values.shape[0])
            ofile.write('Frame Time: %f\n' % X.framerate)

            # Writing the data
            self.motions_ = np.asarray(self.motions_).T
            lines = [" ".join(item) for item in self.motions_.astype(str)]
            ofile.write("".join("%s\n" % l for l in lines))

        def _printJoint(self, X, joint, tab, ofile):

            if X.skeleton[joint]['parent'] == None:
                ofile.write('ROOT %s\n' % joint)
            elif len(X.skeleton[joint]['children']) > 0:
                ofile.write('%sJOINT %s\n' % ('\t' * (tab), joint))
            else:
                ofile.write('%sEnd site\n' % ('\t' * (tab)))

            ofile.write('%s{\n' % ('\t' * (tab)))

            ofile.write('%sOFFSET %3.5f %3.5f %3.5f\n' % ('\t' * (tab + 1),
                                                          X.skeleton[joint]['offsets'][0],
                                                          X.skeleton[joint]['offsets'][1],
                                                          X.skeleton[joint]['offsets'][2]))
            channels = X.skeleton[joint]['channels']
            n_channels = len(channels)

            if n_channels > 0:
                for ch in channels:
                    self.motions_.append(np.asarray(X.values['%s_%s' % (joint, ch)].values))

            if len(X.skeleton[joint]['children']) > 0:
                ch_str = ''.join(' %s' * n_channels % tuple(channels))
                ofile.write('%sCHANNELS %d%s\n' % ('\t' * (tab + 1), n_channels, ch_str))

                for c in X.skeleton[joint]['children']:
                    self._printJoint(X, c, tab + 1, ofile)

            ofile.write('%s}\n' % ('\t' * (tab)))




