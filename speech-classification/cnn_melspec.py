import argparse
import datetime

import torch.nn as nn
from tensorboardX import SummaryWriter

from model.conv import vgg11_bn
from preprocess.dataset import *

xwriter = SummaryWriter('cnn_melspec_log')
data_feed = DataFeed()


def train(model: torch.nn.Module, device, optimizer, nepoch, metric, spec_fd, nbatch=32, log_interval=10):
    model.train()
    losses = []
    print("start train")
    start = datetime.datetime.now()

    for iepoch in range(nepoch):
        train_iter, val_iter = spec_cvloader(spec_fd, iepoch % len(spec_fd), nbatch)
        # acc = evaluate(model, val_iter)
        for i, (X, Y) in enumerate(train_iter):
            # print(X.shape, Y.shape)
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            Ym = model(X)
            loss = metric(Ym, Y)
            xwriter.add_scalar('train/{}th'.format(iepoch), loss.item() / X.size(0), i)
            losses.append(loss.item() / X.size(0))
            loss.backward()
            optimizer.step()
            if i % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    iepoch, i * len(X), len(train_iter.dataset),
                            100. * i / len(train_iter), loss.item()))
        acc = evaluate(model, val_iter)
        print("Train Epoch: {} Loss: {:.3f} Acc: {:.3f}".format(iepoch, losses[-1], acc))
    end = datetime.datetime.now()
    print(end - start)
    print("train finished")


def evaluate(model: torch.nn.Module, val_iter):
    model.eval()
    acc, tot = 0, 0
    with torch.no_grad():
        for i, (X, Y) in enumerate(val_iter):
            X, Y = X.to(device), Y.to(device)
            Ym = model(X)
            Ym = torch.argmax(Ym, dim=1).view(-1)
            Y = Y.view(-1)
            tot += Ym.size(0)
            acc += (Ym == Y).sum().item()
    return acc / tot


def individual_test(model: torch.nn.Module, stu):
    iter = spec_loader([stu], 32)
    acc = evaluate(model, iter)
    print("outsider test acc: {:.3f}".format(acc))


def outsider_test(model: torch.nn.Module, outsiders):
    for o in outsiders:
        iter = spec_loader([o], 32)
        acc = evaluate(model, iter)
        print("outsider {} test acc: {:.3f}".format(data_feed.stuids[o], acc))


def infer(model: torch.nn.Module, sample_path):
    X = read_sample(sample_path)
    X = X[None, :, :, :]
    X = X.cuda()
    model.eval()
    print(X)
    print(X.shape)
    Ym = model(X)
    print(Ym)
    return data_feed.cates[torch.argmax(Ym, dim=1).item()]


def build_model(load, device=torch.device("cuda"), lr=4e-5):
    model = vgg11_bn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if load:
        checkpoint = torch.load(load)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    model.to(device)
    return model, optimizer


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--name", default="cnn_melspec", type=str)
    argparser.add_argument("--infer", default='', type=str)
    argparser.add_argument("--nepoch", default=5, type=int)
    argparser.add_argument("--save", default="./save/save.ptr", type=str)
    argparser.add_argument("--load", default='./save/save.ptr', type=str)
    args = argparser.parse_args()

    # setting
    torch.manual_seed(1)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model, optimizer = build_model(args.load, device, lr=0.001)
    if args.infer:
        infer(model, args.infer)

    candidates = range(22)
    outsiders = range(32)

    spec_fd = spec_folder(candidates, 10)

    metric = nn.CrossEntropyLoss().to(device)
    epoch = args.nepoch
    train(model, device, optimizer, epoch, metric, spec_fd)
    xwriter.export_scalars_to_json("./test.json")
    xwriter.close()

    # outsider_test(model, outsiders)

    checkpointer = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save(checkpointer, args.save)
