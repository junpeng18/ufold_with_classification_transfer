import _pickle as pickle
import sys
import os

import torch
import torch.optim as optim
from torch.utils import data

from Network import UNet as FCNNet

from ufold.utils import *
from ufold.config import process_config
import time
from ufold.data_generator import RNASSDataGenerator, Dataset
from ufold.data_generator import Dataset_Cut_concat_new_canonicle as Dataset_FCN
import collections

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

args = get_args()
if args.nc:
    from ufold.postprocess import postprocess_new_nc as postprocess
else:
    from ufold.postprocess import postprocess_new as postprocess


def get_seq(contact):
    seq = None
    seq = torch.mul(contact.argmax(axis=1), contact.sum(axis=1).clamp_max(1))
    seq[contact.sum(axis=1) == 0] = -1
    return seq


def seq2dot(seq):
    idx = np.arange(1, len(seq) + 1)
    dot_file = np.array(['_'] * len(seq))
    dot_file[seq > idx] = '('
    dot_file[seq < idx] = ')'
    dot_file[seq == 0] = '.'
    dot_file = ''.join(dot_file)
    return dot_file


def get_ct_dict(predict_matrix, batch_num, ct_dict):
    for i in range(0, predict_matrix.shape[1]):
        for j in range(0, predict_matrix.shape[1]):
            if predict_matrix[:, i, j] == 1:
                if batch_num in ct_dict.keys():
                    ct_dict[batch_num] = ct_dict[batch_num] + [(i, j)]
                else:
                    ct_dict[batch_num] = [(i, j)]
    return ct_dict


def get_ct_dict_fast(predict_matrix, batch_num, ct_dict, dot_file_dict, seq_embedding, seq_name):
    seq_tmp = torch.mul(predict_matrix.cpu().argmax(axis=1),
                        predict_matrix.cpu().sum(axis=1).clamp_max(1)).numpy().astype(int)
    seq_tmp[predict_matrix.cpu().sum(axis=1) == 0] = -1
    # seq = (torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1)).numpy().astype(int).reshape(predict_matrix.shape[-1]), torch.arange(predict_matrix.shape[-1]).numpy())
    dot_list = seq2dot((seq_tmp + 1).squeeze())
    seq = ((seq_tmp + 1).squeeze(), torch.arange(predict_matrix.shape[-1]).numpy() + 1)
    letter = 'AUCG'
    ct_dict[batch_num] = [(seq[0][i], seq[1][i]) for i in np.arange(len(seq[0])) if seq[0][i] != 0]
    seq_letter = ''.join([letter[item] for item in np.nonzero(seq_embedding)[:, 1]])
    dot_file_dict[batch_num] = [(seq_name, seq_letter, dot_list[:len(seq_letter)])]
    return ct_dict, dot_file_dict


# randomly select one sample from the test set and perform the evaluation

def model_eval_all_test(contact_net, test_generator):
    contact_net.train()  # ?
    result_no_train = list()
    batch_n = 0
    run_time = []
    for contacts, seq_embeddings, _, seq_lens, seq_ori, _, _, _ in test_generator:
        if seq_lens.item() > 1500:
            continue
        if batch_n % 1000 == 0:
            print('Batch number: ', batch_n)
        batch_n += 1
        contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
        seq_ori = torch.Tensor(seq_ori.float()).to(device)

        ## test
        tik = time.time()
        with torch.no_grad():
            pred_contacts = contact_net(seq_embedding_batch)

        # only post-processing without learning
        u_no_train = postprocess(pred_contacts, seq_ori, 0.01, 0.1, 100, 1.6, True, 1.5)  # 1.6
        # seq_ori, 0.01, 0.1, 100, 1.6, True) ## 1.6
        map_no_train = (u_no_train > 0.5).float()
        tok = time.time()
        t0 = tok - tik
        run_time.append(t0)

        result_no_train_tmp = list(map(lambda i: evaluate_exact_new(map_no_train.cpu()[i],
                                                                    contacts_batch.cpu()[i]),
                                       range(contacts_batch.shape[0])))
        result_no_train += result_no_train_tmp

    nt_exact_p, nt_exact_r, nt_exact_f1 = zip(*result_no_train)
    # pdb.set_trace()
    print('Average testing F1 score with pure post-processing: ', np.average(nt_exact_f1))
    print('Average testing precision with pure post-processing: ', np.average(nt_exact_p))
    print('Average testing recall with pure post-processing: ', np.average(nt_exact_r))


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.cuda.set_device(2)

    # pdb.set_trace()

    config_file = args.config
    test_file = args.test_files

    config = process_config(config_file)
    if test_file not in ['TS1', 'TS2', 'TS3']:
        MODEL_SAVED = 'models/ufold_train.pt'
    else:
        MODEL_SAVED = 'models/ufold_train_pdbfinetune.pt'


    # os.environ["CUDA_VISIBLE_DEVICES"]= config.gpu

    d = config.u_net_d
    BATCH_SIZE = config.batch_size_stage_1
    OUT_STEP = config.OUT_STEP
    LOAD_MODEL = config.LOAD_MODEL
    data_type = config.data_type
    model_type = config.model_type
    model_path = '/data2/darren/experiment/ufold/models_ckpt/'.format(model_type, data_type, d)
    epoches_first = config.epoches_first

    # if gpu is to be used
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    seed_torch()

    # for loading data
    # loading the rna ss data, the data has been preprocessed
    # 5s data is just a demo data, which do not have pseudoknot, will generate another data having that

    # train_data = RNASSDataGenerator('/home/yingxic4/programs/e2efold/data/{}/'.format(data_type), 'train', True)
    # val_data = RNASSDataGenerator('/home/yingxic4/programs/e2efold/data/{}/'.format(data_type), 'val')
    ##test_data = RNASSDataGenerator('./data/{}/'.format(data_type), 'test_no_redundant.pickle')
    # test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/rnastralign_all/', 'test_no_redundant.pickle')
    # test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/rnastralign_all/', 'test_no_redundant_600.pickle')
    print('Loading test file: ', test_file)
    if test_file == 'RNAStralign' or test_file == 'ArchiveII':
        test_data = RNASSDataGenerator('data/', test_file + '.pickle')
    else:
        test_data = RNASSDataGenerator('data/', test_file + '.cPickle')
    # test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'all_1800_archieveII.pickle')
    # test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'all_600_archieveII.pickle')
    # test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA12_test_generate.cPickle')
    # test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA12_RF00001_similarfamilys_test.cPickle')
    # test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA_TestSetA.cPickle')
    # test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA12_allfamily_generate_test.cPickle')
    # test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA12_38family_generate_test.cPickle')
    ##test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA_new_generate.cPickle')
    # test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA_new20201015.cPickle')
    # test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'archieveII_contacts_pred.cPickle')
    # test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA_pdbnewgenerate_yingxc.cPickle')
    # test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA_TS0_ori.cPickle')
    # test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA_TR0_leavefortest1000.cPickle')
    # test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA_TR0_andsim_mutate_extract_train.cPickle')
    seq_len = test_data.data_y.shape[-2]
    print('Max seq length ', seq_len)
    # pdb.set_trace()

    # using the pytorch interface to parallel the data generation and model training
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 6,
              'drop_last': True}
    # # train_set = Dataset(train_data)
    # train_set = Dataset_FCN(train_data)
    # train_generator = data.DataLoader(train_set, **params)

    # # val_set = Dataset(val_data)
    # val_set = Dataset_FCN(val_data)
    # val_generator = data.DataLoader(val_set, **params)

    # test_set = Dataset(test_data)
    test_set = Dataset_FCN(test_data)
    test_generator = data.DataLoader(test_set, **params)

    '''
    test_merge = Dataset_FCN_merge(test_data,test_data2)
    test_merge_generator = data.DataLoader(test_merge, **params)
    pdb.set_trace()
    '''

    contact_net = FCNNet(img_ch=17)

    # pdb.set_trace()
    print('==========Start Loading==========')
    contact_net.load_state_dict(torch.load(MODEL_SAVED, map_location='cuda:1'))
    print('==========Finish Loading==========')
    # contact_net = nn.DataParallel(contact_net, device_ids=[3, 4])
    contact_net.to(device)
    model_eval_all_test(contact_net, test_generator)

    # if LOAD_MODEL and os.path.isfile(model_path):
    #     print('Loading u net model...')
    #     contact_net.load_state_dict(torch.load(model_path))

    # u_optimizer = optim.Adam(contact_net.parameters())


if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')
    main()
