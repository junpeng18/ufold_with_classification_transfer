import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bce_with_logits_loss():
    pos_weight = torch.Tensor([300]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def loss(pred, true, seq_lens):
        # pred, true: tensor (B*L*L)
        # seq_lens: (B,)
        assert true.shape[0] == seq_lens.shape[0]
        mask = torch.zeros(true.shape).bool().to(true.device)
        for i in range(seq_lens.shape[0]):
            mask[i, :seq_lens[i], :seq_lens[i]] = True
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
        return criterion(pred, true)

    return loss
