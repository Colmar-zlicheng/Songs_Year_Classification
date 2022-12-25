# if u use macOS or linux, you should run follow cmd in your terminal at root path:
# export PYTHONPATH="./:$PYTHONPATH"
# if u use windows, you may need these two lines of code:
# import sys
# sys.path.append('.')
import os
import torch
import joblib
import argparse
from lib.utils.etqdm import etqdm
from lib.dataset.Songs import Songs

ANN_exp_path = './exp/ANN'
SVM_exp_path = './exp/SVM'


def compute_age_SVM(args):
    if not os.path.exists(SVM_exp_path):
        print("no such path: ", SVM_exp_path)
        return
    SVM_exp_dir = os.listdir(SVM_exp_path)
    if '.DS_Store' in SVM_exp_dir:
        pop_id = SVM_exp_dir.index('.DS_Store')  # hide file for mac
        SVM_exp_dir.pop(pop_id)
    for idir in etqdm(SVM_exp_dir):
        dir = os.path.join('./exp/SVM', idir)

        set_dir = ckpt = os.path.join(dir, 'setting.txt')
        with open(set_dir, 'r') as f:
            setting = f.readlines()
        train_size = int(setting[0][12:-1])
        test_size = int(setting[1][11:-1])
        data_type = setting[4][6:-1]
        dataset = Songs(train_size=train_size, test_size=test_size, seed=arg.seed)
        data = dataset.Songs_whole(train_size=train_size, test_size=test_size)

        if data_type == 'avg':
            data_test = data['timbre_avg_test']
        elif data_type == 'cov':
            data_test = data['timbre_cov_test']
        else:
            data_test = torch.cat([data['timbre_avg_test'], data['timbre_cov_test']], dim=1)

        ckpt = os.path.join(dir, 'model.m')
        svc = joblib.load(ckpt)
        predicts = svc.predict(data_test)
        predicts = torch.tensor(predicts)
        correct = ((predicts // args.age) == (data['year_test'] // args.age)).sum().item()
        age_acc = correct / data['year_test'].shape[0]
        print("Age accuracy:", '%.2f' % (age_acc * 100), "%")
        save_acc = os.path.join(dir, 'age_acc.txt')
        with open(save_acc, 'w') as ff:
            ff.write("Correct:" + str(correct) + '\n')
            ff.write("Accuracy:" + str('%.2f' % (age_acc * 100)) + '%' + '\n')


def compute_age_ANN(args):
    # TO DO: compute_age_ANN
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--age', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-t', '--type', type=str, default='SVM', choices=['SVM', 'ANN'])

    arg = parser.parse_args()
    if arg.type == 'SVM':
        compute_age_SVM(arg)
    elif arg.type == 'ANN':
        compute_age_ANN(arg)
