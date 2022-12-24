import os
import argparse
import torch
import time
import joblib
from sklearn import svm
from lib.utils.logger import logger
from lib.dataset.Songs import Songs
from lib.utils.save_results import save_results_SVM


def SVM_worker(arg):
    dataset = Songs(train_size=arg.train_size, test_size=arg.test_size, seed=arg.seed)
    data = dataset.Songs_whole(train_size=arg.train_size, test_size=arg.test_size)

    svc = svm.SVC(C=arg.C, kernel=arg.kernel_type)
    if arg.type == 'avg':
        data_train = data['timbre_avg_train']
        data_test = data['timbre_avg_test']
    elif arg.type == 'cov':
        data_train = data['timbre_cov_train']
        data_test = data['timbre_cov_test']
    else:
        data_train = torch.cat([data['timbre_avg_train'], data['timbre_cov_train']], dim=1)
        data_test = torch.cat([data['timbre_avg_test'], data['timbre_cov_test']], dim=1)
    print(f"Start training with {arg.type} ...")
    svc.fit(data_train, data['year_train'])
    print("Start testing...")
    test_result = svc.predict(data_test)
    test_result = torch.tensor(test_result)
    correct = (test_result == data['year_test']).sum().item()
    acc = correct / data['year_test'].shape[0]
    print("Accuracy:", '%.2f' % (acc * 100), "%")
    save_results_SVM(arg=arg, acc=acc, correct=correct)
    save_model = os.path.join(save_dir, 'model.m')
    joblib.dump(svc, save_model)
    save_acc = os.path.join(save_dir, 'acc.txt')
    with open(save_acc, 'w') as ff:
        ff.write("Correct:" + str(correct) + '\n')
        ff.write("Accuracy:" + str('%.2f' % (acc * 100))+'%' + '\n')
    print("-----successfully save results-----")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size', type=int, default=2000)
    parser.add_argument('--test_size', type=int, default=200)
    parser.add_argument('-c', '--C', type=float, default=10.0)
    parser.add_argument('-kt', '--kernel_type', type=str, default='rbf',
                        choices=['linear', 'poly', 'rbf', 'sigmoid'])
    parser.add_argument('-t', '--type', type=str, default='avg', choices=['avg', 'cov', 'cat'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-exp', '--exp_id', type=str)

    arg = parser.parse_args()

    start = time.time()
    if not os.path.exists('./exp'):
        os.mkdir('./exp')
    if not os.path.exists('./exp/SVM'):
        os.mkdir('./exp/SVM')

    print("<<<<<Beginning training SVM>>>>>")
    if arg.exp_id is not None:
        arg.exp_id = arg.exp_id
    else:
        timestamp = time.strftime("%Y_%m%d_%H%M_%S", time.localtime(start))
        arg.exp_id = f"SVM_{arg.type}_{arg.kernel_type}_C{arg.C}_{arg.train_size}+{arg.test_size}_{timestamp}"
    save_dir = os.path.join('./exp/SVM', arg.exp_id)
    logger.set_log_file(save_dir, arg.exp_id)
    save_set = os.path.join(save_dir, 'setting.txt')
    argsDict = arg.__dict__
    with open(save_set, 'w') as f:
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ': ' + str(value) + '\n')

    SVM_worker(arg)

    end = time.time()
    logger.info('Running time: %s Seconds' % (end - start))
