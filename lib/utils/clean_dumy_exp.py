import os
import shutil
import csv

ANN_exp_path = './exp/ANN'
ANN_result_path = './results/ANN_results.csv'
SVM_exp_path = './exp/SVM'
SVM_result_path = './results/SVM_results.csv'
KNN_exp_path = './exp/KNN'
KNN_result_path = './results/KNN_results.csv'


def main_ANN():
    if not os.path.exists(ANN_exp_path):
        print("no such path: ", ANN_exp_path)
        return
    ANN_exp_dir = os.listdir(ANN_exp_path)
    if not os.path.exists(ANN_result_path):
        print("no such file: ", ANN_result_path)
        return
    f = open(ANN_result_path, 'r')
    ANN_csv = csv.reader(f)
    exp_list = []
    for row in ANN_csv:
        exp_list.append(row[-1])
    if '.DS_Store' in ANN_exp_dir:
        pop_id = ANN_exp_dir.index('.DS_Store')  # hide file for mac
        ANN_exp_dir.pop(pop_id)
    # print(exp_list)
    for idir in ANN_exp_dir:
        dir = os.path.join('./exp/ANN', idir)
        need_to_clean = False
        if dir not in exp_list:
            need_to_clean = True
        if need_to_clean is True:
            print(f"remove {dir}")
            input("Confirm ?")
            shutil.rmtree(dir)


def main_SVM():
    if not os.path.exists(SVM_exp_path):
        print("no such path: ", SVM_exp_path)
        return
    if not os.path.exists(SVM_result_path):
        print("no such file: ", SVM_result_path)
        return
    SVM_exp_dir = os.listdir(SVM_exp_path)
    ff = open(SVM_result_path, 'r')
    SVM_csv = csv.reader(ff)
    exp_list = []
    for row in SVM_csv:
        exp_list.append(row[-1])
    if '.DS_Store' in SVM_exp_dir:
        pop_id = SVM_exp_dir.index('.DS_Store')  # hide file for mac
        SVM_exp_dir.pop(pop_id)
    # print(exp_list)
    for idir in SVM_exp_dir:
        dir = os.path.join('./exp/SVM', idir)
        need_to_clean = False
        if dir not in exp_list:
            need_to_clean = True
        if need_to_clean is True:
            print(f"remove {dir}")
            input("Confirm ?")
            shutil.rmtree(dir)


def main_KNN():
    if not os.path.exists(KNN_exp_path):
        print("no such path: ", KNN_exp_path)
        return
    if not os.path.exists(KNN_result_path):
        print("no such file: ", KNN_result_path)
        return
    KNN_exp_dir = os.listdir(KNN_exp_path)
    fff = open(KNN_result_path, 'r')
    KNN_csv = csv.reader(fff)
    exp_list = []
    for row in KNN_csv:
        exp_list.append(row[-1])
    if '.DS_Store' in KNN_exp_dir:
        pop_id = KNN_exp_dir.index('.DS_Store')  # hide file for mac
        KNN_exp_dir.pop(pop_id)
    # print(exp_list)
    for idir in KNN_exp_dir:
        dir = os.path.join('./exp/KNN', idir)
        need_to_clean = False
        if dir not in exp_list:
            need_to_clean = True
        if need_to_clean is True:
            print(f"remove {dir}")
            input("Confirm ?")
            shutil.rmtree(dir)


if __name__ == '__main__':
    # clean the dummy exp not in ANN_results.csv
    main_ANN()
    main_SVM()
    main_KNN()
    print('Successfully clean dummy exp!')
