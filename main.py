# Copyright (c) Alibaba Group
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import classifier

parser = argparse.ArgumentParser(description='Train Linear Classifier with SeA')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('-b', '--batch-size', default=256, type=int)
parser.add_argument('--lr', default=1.6, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--tau', default=1e-6, type=float)
parser.add_argument('--train-feat-path', type=str)
parser.add_argument('--train-label-path', type=str)
parser.add_argument('--test-feat-path', type=str)
parser.add_argument('--test-label-path', type=str)
parser.add_argument('--alpha', default=0.01, type=float)
parser.add_argument('--delta', default=0, type=float)
parser.add_argument('--eta', default=0.8, type=float)
parser.add_argument('--lamda', default=0.1, type=float)


def main():
    args = parser.parse_args()
    # load extracted features
    xtr = F.normalize(torch.load(args.train_feat_path), dim=1)  # n x d matrix
    xte = F.normalize(torch.load(args.test_feat_path), dim=1)
    ytr = torch.LongTensor(torch.load(args.train_label_path))  # n-dimensional vector
    yte = torch.LongTensor(torch.load(args.test_label_path))
    if torch.cuda.is_available():
        xtr, ytr = xtr.cuda(), ytr.cuda()
        xte, yte = xte.cuda(), yte.cuda()
    acc = train(xtr, ytr, xte, yte, args)
    print('Test accuracy is {}'.format(acc))


def train(xtr, ytr, xte, yte, args):
    num_ins, dim = xtr.shape
    num_class = torch.max(ytr) + 1
    model = classifier.SeA(dim, num_class).cuda()
    model.requires_grad = True
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.tau)
    for epoch in range(args.start_epoch, args.epochs):
        idx = torch.randperm(num_ins).cuda()
        train_epoch(xtr[idx, :], ytr[idx], model, criterion, optimizer, args)
    acc = validate(xte, yte, model)
    return acc


def train_epoch(x, y, model, criterion, optimizer, args):
    idx = torch.arange(args.batch_size).cuda()
    model.train()
    for i in range(0, len(y), args.batch_size):
        batch_x = x[i:i + args.batch_size, :]
        batch_y = y[i:i + args.batch_size]
        cur_size = len(batch_y)
        if cur_size < args.batch_size:
            idx = torch.arange(cur_size).cuda()
        adv_x = get_adv(batch_x, batch_y, model, criterion, idx, args)
        batch_x += args.eta * adv_x
        batch_x = F.normalize(batch_x, dim=1)
        logits = model(batch_x)
        logits[idx, batch_y] -= args.delta
        loss = args.lamda * criterion(logits / args.lamda, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(x, y, model):
    model.eval()
    logits = model(x)
    predict = torch.argmax(logits, dim=1)
    return 100. * torch.sum(predict - y == 0) / len(y)


def get_adv(batch_x, batch_y, model, criterion, idx, args):
    model.eval()
    model.requires_grad = False
    batch_x.requires_grad = True
    logits = model(batch_x)
    logits[idx, batch_y] -= args.delta
    loss = args.lamda * criterion(logits / args.lamda, batch_y)
    loss.backward()
    adv_x = F.normalize(batch_x.grad, dim=1)
    model.train()
    model.requires_grad = True
    batch_x.requires_grad = False
    prob_adv = torch.softmax(adv_x.matmul(batch_x.t()) / args.alpha, dim=1)
    prob_adv -= torch.diag(torch.diag(prob_adv))
    prob_adv /= torch.sum(prob_adv, keepdim=True, dim=1)
    return F.normalize(prob_adv.matmul(batch_x), dim=1)


if __name__ == '__main__':
    main()
