import os.path

import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import math
from itertools import groupby
from tqdm import tqdm
from torch.utils.data.dataset import T_co
from pathos.multiprocessing import ProcessingPool as Pool
from Bio import SeqIO
import gzip
from p_tqdm import p_map
from utils.download_RNA_family_data import download_family_data
from utils.path_utils import base_dir


def one_hot(seq):
    """ convert RNA seq into one-hot matrix
    :param seq: string
    :return: one hot matrix (len(seq), 4)
    """
    RNN_seq = seq
    BASES = 'AUCG'
    bases = np.array([base for base in BASES])
    feat = np.concatenate(
        [[(bases == base.upper()).astype(int)] if str(base).upper() in BASES else np.array([[0] * len(BASES)]) for base
         in RNN_seq])

    return feat


def gen_contact_mat_from_pair_list(pair_list: list):
    """
    :param pair_list: list of length seq_len
    :return: np.ndarray (seq_len, seq_len), upper triangle matrix
    """
    seq_len = len(pair_list)
    real_pair = [(i, pair_list[i] - 1) for i in range(seq_len) if pair_list[i] > 0]
    id_from = np.array([int(i) for i, _ in real_pair])
    id_to = np.array([int(j) for _, j in real_pair])
    contact = np.zeros([seq_len, seq_len])
    contact[id_from, id_to] = 1
    assert np.sum(np.abs(contact.T - contact) < 1e-5) == np.prod(contact.shape)
    return contact


def create_prob_feat_image(data):
    def Gaussian(x):
        return math.exp(-0.5 * (x * x))

    def paired(x, y):
        if x == [1, 0, 0, 0] and y == [0, 1, 0, 0]:
            return 2
        elif x == [0, 0, 0, 1] and y == [0, 0, 1, 0]:
            return 3
        elif x == [0, 0, 0, 1] and y == [0, 1, 0, 0]:
            return 0.8
        elif x == [0, 1, 0, 0] and y == [1, 0, 0, 0]:
            return 2
        elif x == [0, 0, 1, 0] and y == [0, 0, 0, 1]:
            return 3
        elif x == [0, 1, 0, 0] and y == [0, 0, 0, 1]:
            return 0.8
        else:
            return 0

    mat = np.zeros([len(data), len(data)])
    for i in range(len(data)):
        for j in range(len(data)):
            coefficient = 0
            for add in range(30):
                if i - add >= 0 and j + add < len(data):
                    score = paired(list(data[i - add]), list(data[j + add]))
                    if score == 0:
                        break
                    else:
                        coefficient = coefficient + score * Gaussian(add)
                else:
                    break
            if coefficient > 0:
                for add in range(1, 30):
                    if i + add < len(data) and j - add >= 0:
                        score = paired(list(data[i + add]), list(data[j - add]))
                        if score == 0:
                            break
                        else:
                            coefficient = coefficient + score * Gaussian(add)
                    else:
                        break
            mat[[i], [j]] = coefficient
    return mat


class RNASeqDataset(Dataset):
    def __init__(self, seq_data):
        super(RNASeqDataset, self).__init__()
        # seq_data: list of RNA data (filename, str_seq, pair_list)
        self.data = seq_data
        self.max_seq_len = max([len(self.data[i][1]) for i in range(len(self.data))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class BPseqDataset(Dataset):
    def __init__(self, bpseq_lst_file_path):
        super(BPseqDataset, self).__init__()
        self.data = []
        with open(bpseq_lst_file_path) as f:
            for l in f:
                l = l.rstrip('\n').split()
                self.data.append(self.read(l[0]))
        self.max_seq_len = max([len(self.data[i][1]) for i in range(len(self.data))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def read(self, filename):
        with open(filename) as f:
            structure_is_known = True
            pair_dict = []  # padding to make index start from 1 if needed
            s = ['']
            for l in f:
                if not l.startswith('#'):
                    l = l.rstrip('\n').split()
                    if len(l) == 3:
                        if not structure_is_known:
                            raise ('invalid format: {}'.format(filename))
                        idx, c, pair = l
                        pos = 'x.<>|'.find(pair)
                        if pos >= 0:
                            idx, pair = int(idx), -pos
                        else:
                            idx, pair = int(idx), int(pair)
                        s.append(c)
                        pair_dict.append(pair)
                    elif len(l) == 4:
                        structure_is_known = False
                        idx, c, nll_unpaired, nll_paired = l
                        s.append(c)
                        nll_unpaired = math.nan if nll_unpaired == '-' else float(nll_unpaired)
                        nll_paired = math.nan if nll_paired == '-' else float(nll_paired)
                        pair_dict.append([nll_unpaired, nll_paired])
                    else:
                        raise ('invalid format: {}'.format(filename))

        if structure_is_known:
            seq = ''.join(s)
            return filename, seq, pair_dict
        else:
            seq = ''.join(s)
            pair_dict.pop(0)
            return filename, seq, pair_dict


class FAseqDataset(Dataset):
    def __init__(self, index_pd, fa_file_dir='data/family_data', num_each_class=150, seed=1234, seq_len_threshold=800):
        super(FAseqDataset, self).__init__()
        self.data = []
        self.dic = {arch: i for i, arch in enumerate(index_pd['architecture'].unique())}
        self.num_each_class = num_each_class
        self.read(index_pd, fa_file_dir, seed, seq_len_threshold)
        self.max_seq_len = max([len(self.data[i][1]) for i in range(len(self.data))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def read(self, index_pd, fa_file_dir, seed=1234, seq_len_threshold=800):
        np.random.seed(seed)
        rfam_id_list = index_pd['RFAM_ID'].tolist()
        arch_list = index_pd['architecture'].tolist()
        arch_data_dic = {}
        for key in self.dic.keys():
            arch_data_dic[key] = []

        for i in range(len(rfam_id_list)):
            rfam_id = rfam_id_list[i]
            label = self.dic[arch_list[i]]
            fa_file = os.path.join(fa_file_dir, f'{rfam_id}.fa.gz')
            with gzip.open(fa_file, "rt") as f:
                for index, record in enumerate(SeqIO.parse(f, 'fasta')):
                    if len(str(record.seq)) > seq_len_threshold:
                        continue
                    file_name = fa_file.split('.fa.gz')[0] + f'_{index}'
                    arch_data_dic[arch_list[i]].append((file_name, str(record.seq), label))
        # sampling
        for key in self.dic.keys():
            if len(arch_data_dic[key]) <= self.num_each_class:
                print(f"Warning: number of seq in {key} is less than {self.num_each_class}")
                self.data.extend(arch_data_dic[key])
            else:
                choice = np.random.choice(len(arch_data_dic[key]), self.num_each_class, replace=False)
                data_selected = [arch_data_dic[key][i] for i in choice]
                self.data.extend(data_selected)
                len_data = [len(data[1]) for data in data_selected]
                print(f"{key}: total number {len(arch_data_dic[key]):.0f}, "
                      f"average len {np.mean(len_data)}, max len {np.max(len_data)}, "
                      f"min len {np.min(len_data)}, std len {np.std(len_data): .0f}, "
                      f"75% quantile {np.quantile(len_data, 0.75): .0f}, "
                      f"50% quantile {np.quantile(len_data, 0.50): .0f}, "
                      f"25% quantile {np.quantile(len_data, 0.25): .0f}")


class ClassImageSeqDataset(Dataset):
    def __init__(self, faseq_data, max_seq_len=None, cal_prob_feat_image=True):
        super(ClassImageSeqDataset, self).__init__()
        self.faseq_data = faseq_data
        if max_seq_len is None:
            self.max_seq_len = faseq_data.max_seq_len
        else:
            self.max_seq_len = max_seq_len
        self.max_seq_len = get_cut_len(self.max_seq_len)
        self.cal_prob_feat_image = cal_prob_feat_image

    def save_prob_feat_images(self, parallel=False, num_cpus=4, use_tqdm=False):
        def save_single_prob_feat(index):
            file_name, rna_seq, _ = self.faseq_data[index]
            feat_path = file_name.split('.fa.gz')[0] + '.npy'
            seq_len = len(rna_seq)
            one_hot_mat = one_hot(rna_seq)
            data_fcn_prob = create_prob_feat_image(one_hot_mat).reshape((1, seq_len, seq_len))
            np.save(feat_path, data_fcn_prob)

        if parallel:
            start_time = time.time()
            print("start save_prob_feat_images...")
            if use_tqdm:
                p_map(save_single_prob_feat, range(len(self.faseq_data)), num_cpus=num_cpus)
            else:
                with Pool(num_cpus) as p:
                    ret = p.map(save_single_prob_feat, range(len(self.faseq_data)))
                    print(f"len {len(ret)}")
            print(f"Time {(time.time() - start_time) / 60:.2f} mins: ImageSeqDataset save_prob_feat_image finished!")
        else:
            for index in tqdm(range(len(self.faseq_data)), desc='ImageSeqDataset save_prob_feat_image'):
                save_single_prob_feat(index)

    def __len__(self):
        return len(self.faseq_data)

    def __getitem__(self, index):
        # start_time = time.time()
        file_name, rna_seq, label = self.faseq_data[index]
        seq_len = len(rna_seq)
        one_hot_mat = one_hot(rna_seq)  # np.ndarray
        data_fcn = np.kron(one_hot_mat.T, one_hot_mat.T).reshape((16, seq_len, seq_len))
        if self.cal_prob_feat_image:
            data_fcn_prob = create_prob_feat_image(one_hot_mat).reshape((1, seq_len, seq_len))
        else:
            feat_path = file_name.split('.bpseq')[0] + '.npy'
            data_fcn_prob = np.load(feat_path)
        feature = torch.Tensor(np.concatenate((data_fcn, data_fcn_prob), axis=0)).float()  # (17, seq_len, seq_len)

        l = min(seq_len, self.max_seq_len)
        feature_extend = torch.zeros(feature.shape[0], self.max_seq_len, self.max_seq_len).float()
        feature_extend[:, :l, :l] = feature[:, :l, :l]
        # print(f"finish get item: {time.time() - start_time:.4f}s\n")
        return feature_extend, label, seq_len


class ContactImageSeqDataset(Dataset):
    def __init__(self, bpseq_data, max_seq_len=None, cal_prob_feat_image=True):
        super(ContactImageSeqDataset, self).__init__()
        self.bpseq_data = bpseq_data
        if max_seq_len is None:
            self.max_seq_len = bpseq_data.max_seq_len
        else:
            self.max_seq_len = max_seq_len
        self.max_seq_len = get_cut_len(self.max_seq_len)
        self.cal_prob_feat_image = cal_prob_feat_image

    def save_prob_feat_images(self, parallel=False, num_cpus=4, use_tqdm=False):
        def save_single_prob_feat(index):
            file_name, rna_seq, pair_list = self.bpseq_data[index]
            feat_path = file_name.split('.bpseq')[0] + '.npy'
            seq_len = len(rna_seq)
            one_hot_mat = one_hot(rna_seq)
            data_fcn_prob = create_prob_feat_image(one_hot_mat).reshape((1, seq_len, seq_len))
            np.save(feat_path, data_fcn_prob)

        if parallel:
            start_time = time.time()
            print("start save_prob_feat_images...")
            if use_tqdm:
                p_map(save_single_prob_feat, range(len(self.bpseq_data)), num_cpus=num_cpus)
            else:
                with Pool(num_cpus) as p:
                    ret = p.map(save_single_prob_feat, range(len(self.bpseq_data)))
                    print(f"len {len(ret)}")
            print(f"Time {(time.time() - start_time) / 60:.2f} mins: ImageSeqDataset save_prob_feat_image finished!")
        else:
            for index in tqdm(range(len(self.bpseq_data)), desc='ImageSeqDataset save_prob_feat_image'):
                save_single_prob_feat(index)

    def __len__(self):
        return len(self.bpseq_data)

    def __getitem__(self, index):
        # start_time = time.time()
        file_name, rna_seq, pair_list = self.bpseq_data[index]
        seq_len = len(rna_seq)
        one_hot_mat = one_hot(rna_seq)  # np.ndarray
        contact = torch.Tensor(gen_contact_mat_from_pair_list(pair_list)).float()  # (seq_len, seq_len)
        data_fcn = np.kron(one_hot_mat.T, one_hot_mat.T).reshape((16, seq_len, seq_len))
        if self.cal_prob_feat_image:
            data_fcn_prob = create_prob_feat_image(one_hot_mat).reshape((1, seq_len, seq_len))
        else:
            feat_path = file_name.split('.bpseq')[0] + '.npy'
            data_fcn_prob = np.load(feat_path)
        feature = torch.Tensor(np.concatenate((data_fcn, data_fcn_prob), axis=0)).float()  # (17, seq_len, seq_len)

        l = min(seq_len, self.max_seq_len)
        contact_extend = torch.zeros(self.max_seq_len, self.max_seq_len).float()
        contact_extend[:l, :l] = contact[:l, :l]
        feature_extend = torch.zeros(feature.shape[0], self.max_seq_len, self.max_seq_len).float()
        feature_extend[:, :l, :l] = feature[:, :l, :l]
        one_hot_extend = torch.zeros(self.max_seq_len, 4).float()
        one_hot_extend[:l, :] = torch.Tensor(one_hot_mat[:l, :]).float()
        # print(f"finish get item: {time.time() - start_time:.4f}s\n")
        return feature_extend, contact_extend, seq_len, one_hot_extend


def load_contact_data(data_dir, train_lst_name, test_lst_name=None, batch_size=64, cal_prob_feat_image=False, seed=1):
    if test_lst_name is not None:
        train_bpseq = BPseqDataset(os.path.join(data_dir, train_lst_name))
        test_bpseq = BPseqDataset(os.path.join(data_dir, test_lst_name))
        train_len = len(train_bpseq)
        test_len = len(test_bpseq)
        data_train = ContactImageSeqDataset(train_bpseq, cal_prob_feat_image=cal_prob_feat_image)
        data_test = ContactImageSeqDataset(test_bpseq, cal_prob_feat_image=cal_prob_feat_image)
        data = {
            'train_bpseq': train_bpseq,
            'test_bpseq': test_bpseq,
            'train_loader': DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=1),
            'test_loader': DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=1)
        }
    else:
        full_bpseq = BPseqDataset(os.path.join(data_dir, train_lst_name))
        train_len = int(len(full_bpseq) * 0.8)
        test_len = int(len(full_bpseq) - train_len)
        full_data = ContactImageSeqDataset(full_bpseq, cal_prob_feat_image=cal_prob_feat_image)
        data_train, data_test = random_split(
            full_data, [train_len, test_len], generator=torch.Generator().manual_seed(seed)
        )
        data = {
            'train_seq': full_bpseq,
            'test_seq': full_bpseq,
            'train_loader': DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=1),
            'test_loader': DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=1)
        }
    print(f"data len: train {train_len}, test {test_len}")
    return data


def load_class_data(data_dir='data/family_data', batch_size=64, cal_prob_feat_image=False, seed=1):
    index_pd = download_family_data()
    full_faseq = FAseqDataset(index_pd, os.path.join(base_dir, data_dir))
    train_len = int(len(full_faseq) * 0.8)
    test_len = int(len(full_faseq) - train_len)
    full_data = ClassImageSeqDataset(full_faseq, cal_prob_feat_image=cal_prob_feat_image)
    data_train, data_test = random_split(
        full_data, [train_len, test_len], generator=torch.Generator().manual_seed(seed)
    )
    data = {
        'train_seq': full_faseq,
        'test_seq': full_faseq,
        'train_loader': DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=1),
        'test_loader': DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=1)
    }
    print(f"data len: train {train_len}, test {test_len}")
    return data


def get_cut_len(seq_len, base=32):
    return int((((seq_len - 1) // base) + 1) * base)
