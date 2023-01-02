import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from ufold.postprocess import postprocess_new as postprocess
from ufold.utils import evaluate_exact_new
from utils.data_utils import BPseqDataset, ContactImageSeqDataset, get_cut_len
from utils.path_utils import base_dir
from Network import UNet, resnet_18, UNetTransfer, FeatureExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def graph_evaluate(pred, true, seq_len, full_graph=True):
    # pred, true: binary matrix
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.cpu().numpy()
    if seq_len < true.shape[0]:
        pred = pred[:seq_len, :seq_len]
        true = true[:seq_len, :seq_len]
    else:
        seq_len = pred.shape[0]
    if not full_graph:
        pred = np.triu(pred, k=1)
        true = np.triu(true, k=1)
        total_num = int(seq_len * (seq_len - 1) / 2)
    else:
        total_num = int(seq_len**2)

    eps = 1e-20
    tp = int(np.sum(pred * true))
    p = int(np.sum(true))
    pp = int(np.sum(pred))
    fp = pp - tp
    fn = p - fp
    tn = total_num - tp - fp - fn
    precision = tp / (tp + fp + eps)
    recall = tp / (p + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


def prepare_contact_data(data, target, one_hot, seq_lens=None):
    # data: (B, C, L, L)
    # target: (B, L, L)
    # one_hot: (B, L, 4)
    if seq_lens is not None:
        max_seq_len = get_cut_len(max(64, seq_lens.max().item()))
        data = data[..., :max_seq_len, :max_seq_len]
        target = target[:, :max_seq_len, :max_seq_len]
        one_hot = one_hot[:, :max_seq_len, :].float()
    return data.to(device), target.to(device), one_hot.to(device)


def load_test_data(args):
    test_bpseq = BPseqDataset(os.path.join(args['data_dir'], 'TestSetA.lst'))
    data_test = ContactImageSeqDataset(test_bpseq, cal_prob_feat_image=False)
    test_loader = DataLoader(data_test, batch_size=args['batch_size'], shuffle=False, num_workers=1)
    return test_loader


def load_ufold(args, model_save_dir):
    model = UNet(img_ch=args['input_dim'])  # default to 17
    model.to(device)

    save_state = torch.load(os.path.join(model_save_dir, 'best_model.pth'))
    model.load_state_dict(save_state['state_dict'])
    return model


def load_ufold_transfer(args, model_save_dir, class_model_log_dir):
    # class model
    class_model = resnet_18(img_ch=17, num_classes=18)
    class_model.to(device)
    saved_state = torch.load(os.path.join(class_model_log_dir, 'best_model.pth'))
    class_model.load_state_dict(saved_state['state_dict'])
    feat_extract_model = FeatureExtractor(class_model, layers=["layer2", "layer3", "layer4"])\

    # unet transfer
    model = UNetTransfer(img_ch=args['input_dim'])  # default to 17
    model.to(device)
    save_state = torch.load(os.path.join(model_save_dir, 'best_model.pth'))
    model.load_state_dict(save_state['state_dict'])
    return model, feat_extract_model


def model_eval(args, model, test_loader, feat_extract_model=None):
    model.train()
    result_no_train = list()
    run_time = []
    pbar = tqdm(total=len(test_loader), desc='model_eval')
    for batch_idx, (data, target, seq_lens, one_hot) in enumerate(test_loader):
        seq_feats, contacts, one_hot = prepare_contact_data(data, target, one_hot, seq_lens)

        ## test
        tik = time.time()
        if feat_extract_model is None:
            with torch.no_grad():
                pred_contacts = model(seq_feats)
        else:
            with torch.no_grad():
                feats = feat_extract_model(seq_feats)
                feat_l2, feat_l3, feat_l4 = feats["layer2"], feats["layer3"], feats["layer4"]
                pred_contacts = model(seq_feats, feat_l2, feat_l3, feat_l4)

        # only post-processing without learning
        u_no_train = postprocess(pred_contacts, one_hot, lr_min=0.01, lr_max=0.1,
                                 num_itr=1000, rho=1, with_l1=True, s=1.5)  # rho=1.6
        map_no_train = (u_no_train > 0.5).float()
        tok = time.time()
        t0 = tok - tik
        run_time.append(t0)
        # print(f"id {batch_idx}: run time {t0:.2f}s")

        result_no_train_tmp = list(map(lambda i: graph_evaluate(map_no_train.cpu()[i], contacts.cpu()[i],
                                                                seq_lens[i].item(), full_graph=args['full_graph']),
                                       range(contacts.shape[0])))
        result_no_train += result_no_train_tmp
        pbar.update()

    nt_exact_p, nt_exact_r, nt_exact_f1 = zip(*result_no_train)
    print('Average testing F1 score with pure post-processing: ', np.average(nt_exact_f1))
    print('Average testing precision with pure post-processing: ', np.average(nt_exact_p))
    print('Average testing sensitivity with pure post-processing: ', np.average(nt_exact_r))


def main(model='ufold_transfer'):
    args = {
        'data_dir': 'data',
        'full_graph': False,
        'batch_size': 8,
        'input_dim': 17
    }
    test_loader = load_test_data(args)
    ufold_model_save_dir = 'models/ufold/20230102112553_Ufold_bs_8_lr_0.0100'
    ufold_transfer_model_save_dir = 'models/ufold_transfer/20230102140525_Ufold_Transfer_bs_4_lr_0.0100'
    class_model_log_dir = 'models/resnet_18/20230101213730_Ufold_bs_8_lr_0.0100'

    print(f"test {model}")
    if model == 'ufold':
        model = load_ufold(args, ufold_model_save_dir)
        model_eval(args, model, test_loader)
    elif model == 'ufold_transfer':
        model, feat_extract_model = load_ufold_transfer(args, ufold_transfer_model_save_dir, class_model_log_dir)
        model_eval(args, model, test_loader, feat_extract_model)


if __name__ == '__main__':
    main()
