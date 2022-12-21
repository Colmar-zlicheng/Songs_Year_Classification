import os
import time
import torch
import argparse
from lib.utils.etqdm import etqdm
from lib.dataset.Songs import Songs
from lib.utils.misc import bar_perfixes
from torch.utils.tensorboard import SummaryWriter
from lib.utils.logger import logger
from lib.model.Songs_Years import Songs_Years


def ANN_worker(arg, summary):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Use device: {device}")

    train_data = Songs(data_split='train', train_size=arg.train_size, test_size=arg.test_size, device=device, seed=arg.seed)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=arg.batch_size,
                                               shuffle=True,
                                               drop_last=True)
    test_data = Songs(data_split='test', train_size=arg.train_size, test_size=arg.test_size, device=device, seed=arg.seed)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=arg.batch_size,
                                              shuffle=False,
                                              drop_last=False)

    model = Songs_Years(num_years=42).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=arg.learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, arg.decay_step, arg.decay_gamma)

    for epoch_idx in range(arg.epoch_size):
        model.train()
        train_bar = etqdm(train_loader)
        correct = 0
        total = 0
        for bidx, input in enumerate(train_bar):
            step_idx = epoch_idx * len(train_loader) + bidx
            pred, loss = model(input)
            train_bar.set_description(f"{bar_perfixes['train']} Epoch {epoch_idx} Loss {'%.12f' % loss}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step_idx % arg.log_interval == 0:
                summary.add_scalar(f"scalar/loss", loss, global_step=step_idx, walltime=None)
            _, predicted = torch.max(pred.data, 1)
            predicted += 1969
            total += input['year'].size(0)
            correct += (predicted == input['year']).sum().item()
        train_acc = 100 * correct / total
        summary.add_scalar(f"scalar/train_acc", train_acc, global_step=epoch_idx, walltime=None)
        scheduler.step()
        print(f"Current LR: {[group['lr'] for group in optimizer.param_groups]}, Train Accuracy: {train_acc}")

    with torch.no_grad():
        correct = 0
        total = 0
        model.eval()
        test_bar = etqdm(test_loader)
        for bidx, input in enumerate(test_bar):
            pred, loss = model(input)
            test_bar.set_description(f"{bar_perfixes['test']} Loss {'%.12f' % loss}")
            _, predicted = torch.max(pred.data, 1)
            predicted += 1969
            total += input['year'].size(0)
            correct += (predicted == input['year']).sum().item()
            acc = 100 * correct / total
        print('Accuracy on test set: {} %'.format(acc))
        save_acc_path = os.path.join('./exp/ANN', arg.exp_id, 'acc_test_txt')
        with open(save_acc_path, 'w') as ff:
            ff.write("Correct_test:" + str(correct) + '\n')
            ff.write("Accuracy_test:" + str(acc) + '\n')

    logger.info("-----beginning save checkpoints and results-----")
    save_path = os.path.join('./exp/ANN', arg.exp_id, 'model.ckpt')
    torch.save(model.state_dict(), save_path)
    logger.info("-----successfully save checkpoints-----")
    # save_results_ANN(arg=arg, val_acc=val_acc, test_acc=acc, exp=save_dir)
    # print("-----successfully save results-----")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=100)
    parser.add_argument('-e', '--epoch_size', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-ds', '--decay_step', type=int, default=30)
    parser.add_argument('-dg', '--decay_gamma', type=float, default=0.1)
    parser.add_argument('-log', '--log_interval', type=int, default=50)
    parser.add_argument('-eval', '--eval_interval', type=int, default=1)
    parser.add_argument('--train_size', type=int, default=2000)
    parser.add_argument('--test_size', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-exp', '--exp_id', type=str)

    if not os.path.exists('./exp'):
        os.mkdir('./exp')
    if not os.path.exists('./exp/ANN'):
        os.mkdir('./exp/ANN')

    arg = parser.parse_args()

    start = time.time()
    timestamp = time.strftime("%Y_%m%d_%H%M_%S", time.localtime(start))
    if arg.exp_id is not None:
        arg.exp_id = f"{arg.exp_id}_{timestamp}"
    else:
        arg.exp_id = f"Songs_e{arg.epoch_size}_d{arg.decay_step}-{arg.decay_gamma}_{timestamp}"

    exp_path = os.path.join('./exp/ANN', arg.exp_id)
    logger.set_log_file(exp_path, arg.exp_id)

    save_cfg = os.path.join(exp_path, 'Hyperparameters.txt')
    argsDict = arg.__dict__
    with open(save_cfg, 'w') as f:
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ': ' + str(value) + '\n')

    summary_dir = os.path.join(exp_path, 'run')
    os.mkdir(summary_dir)
    summary = SummaryWriter(summary_dir)

    ANN_worker(arg, summary)

    end = time.time()
    logger.info('Running time: %s Seconds' % (end - start))
