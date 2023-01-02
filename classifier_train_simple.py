import os
import json
import torch
from datetime import datetime

from Trainer import ClassifierTrainer
from utils.data_utils import load_class_data
from utils.TrainInits import init_seed
from utils.logger import get_logger
from Network import resnet_18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_trainer(args, seed=1):
    init_seed(seed)
    data = load_class_data(batch_size=args['batch_size'], seed=1)
    train_loader = data['train_loader']
    val_loader = data['test_loader']

    model = resnet_18(img_ch=args['input_dim'], num_classes=len(data['train_seq'].dic))  # default to 17
    model.to(device)

    from utils.TrainInits import print_model_parameters
    print_model_parameters(model, only_num=True)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args['lr_init'], eps=1.0e-3)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=args['lr_milestones'],
                                                        gamma=args['lr_decay_rate'])
    logger = get_logger(args['log_dir'], name='Resnet_18', debug=args['debug'])
    trainer = ClassifierTrainer(model, loss, optimizer, train_loader, val_loader, args, logger,
                                lr_scheduler=lr_scheduler)
    return trainer


def main(seed=1):
    args = {
        'log_dir': 'models/resnet_18',
        'debug': False,
        'tb_dir': 'runs/resnet_18',
        'data_dir': 'data',
        'batch_size': 32,
        'input_dim': 17,
        'epochs': 100,
        'lr_init': 1e-2,
        'lr_milestones': [20, 30, 40, 50],
        'lr_decay_rate': 0.1,
        'log_step': 10,
        'early_stop': 10,
        'plot': True
    }
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    run_id = f"{current_time}_Ufold_bs_{args['batch_size']}_lr_{args['lr_init']:.4f}"
    args['log_dir'] = f"{args['log_dir']}/{run_id}"
    args['tb_dir'] = f"{args['tb_dir']}/{run_id}"
    if not os.path.isdir(args['log_dir']):
        os.makedirs(args['log_dir'], exist_ok=True)
    with open(os.path.join(args['log_dir'], 'args.txt'), 'w') as f:
        json.dump(args, f, indent=2)

    trainer = prepare_trainer(args, seed)
    trainer.train()


if __name__ == '__main__':
    main()
