import os

from utils.data_utils import BPseqDataset, ContactImageSeqDataset, FAseqDataset, ClassImageSeqDataset
from utils.path_utils import base_dir
from utils.download_RNA_family_data import download_family_data


def main_bpseq():
    data_dir = 'data'
    train_lst_name = 'TrainSetA.lst'
    test_lst_name = 'TestSetA.lst'
    train_bpseq = BPseqDataset(os.path.join(data_dir, train_lst_name))
    test_bpseq = BPseqDataset(os.path.join(data_dir, test_lst_name))
    data_train = ContactImageSeqDataset(train_bpseq)
    data_test = ContactImageSeqDataset(test_bpseq)

    num_cpus = 24
    data_train.save_prob_feat_images(parallel=True, num_cpus=num_cpus)
    data_test.save_prob_feat_images(parallel=True, num_cpus=num_cpus)


def main_faseq():
    data_dir = 'data/family_data'
    index_pd = download_family_data()
    full_faseq = FAseqDataset(index_pd, os.path.join(base_dir, data_dir))
    full_data = ClassImageSeqDataset(full_faseq)
    num_cpus = 24
    full_data.save_prob_feat_images(parallel=True, num_cpus=num_cpus)


if __name__ == '__main__':
    main_faseq()
