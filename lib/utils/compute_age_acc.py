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
from lib.utils.misc import bar_perfixes
from lib.dataset.Songs import Songs, Songs_Total
from lib.model.Songs_Years import Songs_Years, SY_Baseline

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

    for idir in SVM_exp_dir:
        dir = os.path.join('./exp/SVM', idir)
        save_name = f"age_{args.age}_acc.txt"
        save_acc = os.path.join(dir, save_name)
        if os.path.exists(save_acc):
            print(f"{save_acc} exists")
            continue

        ckpt = os.path.join(dir, 'model.m')
        if not os.path.exists(ckpt):
            print(f"{ckpt} don't exists, consider to clean this exp")
            continue

        print(f"beginning compute age({args.age}) acc for {dir}")
        set_dir = os.path.join(dir, 'setting.txt')
        with open(set_dir, 'r') as f:
            setting = f.readlines()
        train_size = int(setting[0][12:-1])
        test_size = int(setting[1][11:-1])
        data_type = setting[4][6:-1]
        dataset = Songs(train_size=train_size, test_size=test_size, seed=args.seed)
        data = dataset.Songs_whole(train_size=train_size, test_size=test_size)

        if data_type == 'avg':
            data_test = data['timbre_avg_test']
        elif data_type == 'cov':
            data_test = data['timbre_cov_test']
        else:
            data_test = torch.cat([data['timbre_avg_test'], data['timbre_cov_test']], dim=1)

        svc = joblib.load(ckpt)
        predicts = svc.predict(data_test)
        predicts = torch.tensor(predicts)
        correct = ((predicts // args.age) == (data['year_test'] // args.age)).sum().item()
        age_acc = correct / data['year_test'].shape[0]
        print("Age accuracy:", '%.2f' % (age_acc * 100), "%")

        with open(save_acc, 'w') as ff:
            ff.write("Correct:" + str(correct) + '\n')
            ff.write("Accuracy:" + str('%.2f' % (age_acc * 100)) + '%' + '\n')


def compute_age_ANN(args):
    if not os.path.exists(ANN_exp_path):
        print("no such path: ", ANN_exp_path)
        return
    ANN_exp_dir = os.listdir(ANN_exp_path)
    if '.DS_Store' in ANN_exp_dir:
        pop_id = ANN_exp_dir.index('.DS_Store')  # hide file for mac
        ANN_exp_dir.pop(pop_id)

    for idir in ANN_exp_dir:
        dir = os.path.join('./exp/ANN', idir)

        set_dir = os.path.join(dir, 'Hyperparameters.txt')
        with open(set_dir, 'r') as f:
            setting = f.readlines()
        age = int(setting[0][5:-1])
        batch_size = int(setting[1][12:-1])
        seed = int(setting[7][6:-1])
        big_dataset = setting[9][13:-1]
        train_size = int(setting[10][12:-1])
        test_size = int(setting[11][11:-1])
        model_type = setting[12][7:-1]
        base_type = setting[13][11:-1]

        save_name = f"acc_age{age}_test.txt"
        save_acc = os.path.join(dir, save_name)
        if os.path.exists(save_acc):
            print(f"{save_acc} exists")
            continue

        if model_type == 'Songs':
            ckpt = os.path.join(dir, 'Songs.ckpt')
            # model = Songs_Years(num_years=num_years, begin_year=begin_year).to(device)
        else:
            ckpt = os.path.join(dir, 'Baseline.ckpt')
            # model = SY_Baseline(num_years=num_years, begin_year=begin_year, mode=arg.base_type).to(device)
        if not os.path.exists(ckpt):
            print(f"{ckpt} don't exists, consider to clean this exp")
            continue

        print(f"beginning compute age({age}) acc for {dir}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if big_dataset == 'True':
            begin_year = 1922
            num_years = 90
            test_data = Songs_Total(data_split='test', device=device, seed=seed)

        else:
            begin_year = 1969
            num_years = 42
            test_data = Songs(data_split='test', train_size=train_size, test_size=test_size, device=device,
                              seed=seed)
        test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  drop_last=False)

        if model_type == 'Songs':
            model = Songs_Years(num_years=num_years, begin_year=begin_year).to(device)
        else:
            model = SY_Baseline(num_years=num_years, begin_year=begin_year, mode=base_type).to(device)

        with torch.no_grad():
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
                correct_age += ((predicted // age) == (inputs['year'] // age)).sum().item()
            age_acc = 100 * correct_age / total
            print('Age {} Accuracy on test set: {} %'.format(age, '%.2f' % age_acc))
            save_age_acc_path = os.path.join(dir, f"acc_age{age}_test.txt")
            with open(save_age_acc_path, 'w') as fff:
                fff.write("Correct_age_test:" + str(correct_age) + '\n')
                fff.write("Age Accuracy_test:" + str(age_acc) + '%' + '\n')


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
