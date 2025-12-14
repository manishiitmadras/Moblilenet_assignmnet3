# train.py
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from models import vgg, cfg_vgg6, GetCifar10
import wandb

# ---- helper functions ----
def get_optimizer(optim_name, parameters, lr, momentum=0.9):
    optim_name = optim_name.lower()
    if optim_name == 'sgd':
        return optim.SGD(parameters, lr=lr, momentum=momentum)
    if optim_name == 'nesterov-sgd' or optim_name == 'nesterov':
        return optim.SGD(parameters, lr=lr, momentum=momentum, nesterov=True)
    if optim_name == 'adam':
        return optim.Adam(parameters, lr=lr)
    if optim_name == 'adagrad':
        return optim.Adagrad(parameters, lr=lr)
    if optim_name == 'rmsprop':
        return optim.RMSprop(parameters, lr=lr, momentum=momentum)
    if optim_name == 'nadam':
        # PyTorch >=1.8 has NAdam as optim.NAdam
        try:
            return optim.NAdam(parameters, lr=lr)
        except AttributeError:
            raise ValueError("NAdam not available in installed torch. Upgrade torch or choose another optimizer.")
    raise ValueError(f"Unknown optimizer: {optim_name}")

def eval_model(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    return 100. * correct / total

# ---- training loop ----
def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.init(project=config.get('wandb_project','cifar-vgg-sweep'), config=config)
    cfg = wandb.config

    # build model
    model = vgg(cfg_vgg6, num_classes=10, batch_norm=cfg.batch_norm, activation=cfg.activation)
    model = model.to(device)

    # data
    train_loader, test_loader = GetCifar10(batchsize=cfg.batch_size, num_workers=cfg.num_workers, use_cutout=cfg.use_cutout)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(cfg.optimizer, model.parameters(), lr=cfg.lr, momentum=cfg.momentum)

    best_test = 0.0
    save_dir = config.get('save_dir','checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_acc = eval_model(model, train_loader, device)
        test_acc = eval_model(model, test_loader, device)

        wandb.log({
            'epoch': epoch,
            'train_loss': running_loss/len(train_loader),
            'train_acc': train_acc,
            'test_acc': test_acc,
            'lr': cfg.lr
        }, step=epoch)

        print(f"Epoch {epoch+1}/{cfg.epochs} | loss: {running_loss/len(train_loader):.4f} | train_acc: {train_acc:.2f} | test_acc: {test_acc:.2f}")

        # save best
        if test_acc > best_test:
            best_test = test_acc
            ckpt_path = os.path.join(save_dir, f"best_{wandb.run.id}.pth")
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'config': dict(wandb.config),
                'test_acc': test_acc
            }, ckpt_path)
            print("Saved best model to", ckpt_path)
            # log artifact
            wandb.save(ckpt_path)

    wandb.finish()
    return best_test

# ---- CLI ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_project', type=str, default='cifar-vgg-sweep')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--activation', type=str, default='silu', choices=['relu','sigmoid','tanh','silu','gelu'])
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd','nesterov-sgd','adam','adagrad','rmsprop','nadam'])
    parser.add_argument('--batch_norm', type=lambda x: (str(x).lower() in ['true','1','yes']), default=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_cutout', type=lambda x: (str(x).lower() in ['true','1','yes']), default=True)

    args = parser.parse_args()
    # allow both argparse and wandb sweep to provide config:
    config = vars(args)
    # pass config to train
    train(config)
