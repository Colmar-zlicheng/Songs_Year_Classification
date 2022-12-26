import os
import time
import torch
import argparse
from lib.utils.etqdm import etqdm
from lib.dataset.Songs import Songs, Songs_Total
from lib.utils.misc import bar_perfixes
from torch.utils.tensorboard import SummaryWriter
from lib.utils.logger import logger
from lib.model.Songs_Years import Songs_Years, SY_Baseline
from lib.utils.save_results import save_results_ANN


def ANN_worker(arg, summary):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Use device: {device}")
    if arg.big_dataset is True:
        begin_year = 1922
        num_years = 90
        train_data = Songs_Total(data_split='train', device=device, seed=arg.seed)
        test_data = Songs_Total(data_split='test', device=device, seed=arg.seed)

    else:
        begin_year = 1969
        num_years = 42
        train_data = Songs(data_split='train', train_size=arg.train_size, test_size=arg.test_size, device=device,
                           seed=arg.seed)
        test_data = Songs(data_split='test', train_size=arg.train_size, test_size=arg.test_size, device=device,
                          seed=arg.seed)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=arg.batch_size,
                                               shuffle=True,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=arg.batch_size,
                                              shuffle=False,
                                              drop_last=False)
    if arg.model == 'Songs':
        model = Songs_Years(num_years=num_years, begin_year=begin_year).to(device)
    else:
        model = SY_Baseline(num_years=num_years, begin_year=begin_year, mode=arg.base_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=arg.learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, arg.decay_step, arg.decay_gamma)

    for epoch_idx in range(arg.epoch_size):
        model.train()
        train_bar = etqdm(train_loader)
        correct = 0
        total = 0
        correct_age = 0
        for bidx, inputs in enumerate(train_bar):
            step_idx = epoch_idx * len(train_loader) + bidx
            pred, loss = model(inputs)
            train_bar.set_description(f"{bar_perfixes['train']} Epoch {epoch_idx} Loss {'%.12f' % loss}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step_idx % arg.log_interval == 0:
                summary.add_scalar(f"scalar/loss", loss, global_step=step_idx, walltime=None)
            _, predicted = torch.max(pred.data, 1)
            predicted += begin_year
            total += inputs['year'].size(0)
            correct += (predicted == inputs['year']).sum().item()
            correct_age += ((predicted // arg.age) == (inputs['year'] // arg.age)).sum().item()
        train_acc = 100 * correct / total
        train_age_acc = 100 * correct_age / total
        summary.add_scalar(f"scalar/train_acc", train_acc, global_step=epoch_idx, walltime=None)
        summary.add_scalar(f"scalar/train_age_acc", train_age_acc, global_step=epoch_idx, walltime=None)
        scheduler.step()
        print(f"Current LR: {[group['lr'] for group in optimizer.param_groups]}, Train Accuracy: {'%.2f' % train_acc}%,"
              f" Train Age Accuracy: {'%.2f' % train_age_acc}%")

    with torch.no_grad():
        correct = 0
        correct_age = 0
        total = 0
        model.eval()
        test_bar = etqdm(test_loader)
        for bidx, inputs in enumerate(test_bar):
            pred, loss = model(inputs)
            test_bar.set_description(f"{bar_perfixes['test']} Loss {'%.12f' % loss}")
            _, predicted = torch.max(pred.data, 1)
            predicted += begin_year
            total += inputs['year'].size(0)
            correct += (predicted == inputs['year']).sum().item()
            correct_age += ((predicted // arg.age) == (inputs['year'] // arg.age)).sum().item()
        acc = 100 * correct / total
        age_acc = 100 * correct_age / total
        print('Accuracy on test set: {}%, Age {} Accuracy: {}%'.format('%.2f' % acc, arg.age, '%.2f' % age_acc))
        save_acc_path = os.path.join('./exp/ANN', arg.exp_id, 'acc_test.txt')
        save_age_acc_path = os.path.join('./exp/ANN', arg.exp_id, f"acc_age{arg.age}_test.txt")
        with open(save_acc_path, 'w') as ff:
            ff.write("Correct_test:" + str(correct) + '\n')
            ff.write("Accuracy_test:" + str(acc) + '%' + '\n')
        with open(save_age_acc_path, 'w') as fff:
            fff.write("Correct_age_test:" + str(correct_age) + '\n')
            fff.write("Age Accuracy_test:" + str(age_acc) + '%' + '\n')

    logger.info("-----beginning save checkpoints and results-----")
    if arg.model == 'Songs':
        save_model = 'Songs.ckpt'
    else:
        save_model = 'Baseline.ckpt'
    save_path = os.path.join('./exp/ANN', arg.exp_id, save_model)
    torch.save(model.state_dict(), save_path)
    logger.info("-----successfully save checkpoints-----")
    save_results_ANN(arg=arg, train_acc=train_acc, train_age_acc=train_age_acc, test_acc=acc, test_age_acc=age_acc)
    logger.info("-----successfully save results-----")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--age', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=200)
    parser.add_argument('-e', '--epoch_size', type=int, default=300)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-ds', '--decay_step', type=int, default=50)
    parser.add_argument('-dg', '--decay_gamma', type=float, default=0.5)
    parser.add_argument('-log', '--log_interval', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-exp', '--exp_id', type=str)
    parser.add_argument('-bd', '--big_dataset', action='store_true', help="use big dataset")
    parser.add_argument('--train_size', type=int, default=2000)
    parser.add_argument('--test_size', type=int, default=200)
    parser.add_argument('-m', '--model', type=str, default='Songs',
                        choices=['Songs', 'baseline'],
                        help="run whole model or baseline")
    parser.add_argument('-bt', '--base_type', type=str, default='avg', choices=['avg', 'cov', 'cat'])

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
        if arg.model == 'Songs':
            model_name = 'Songs'
            arg.base_type = '--'
        else:
            model_name = f"Baseline_{arg.base_type}"
        if arg.big_dataset is True:
            dataset_name = 'Big'
            arg.train_size = 463715
            arg.test_size = 51630
        else:
            dataset_name = 'Small'
        arg.exp_id = f"{model_name}_{dataset_name}_{arg.epoch_size}_d{arg.decay_step}-{arg.decay_gamma}_{timestamp}"

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
