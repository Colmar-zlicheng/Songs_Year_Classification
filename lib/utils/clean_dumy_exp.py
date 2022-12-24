import os
import shutil
import csv

ANN_exp_path = './exp/ANN'
ANN_result_path = './results/ANN_results.csv'
SVM_exp_path = './exp/SVM'
SVM_result_path = './results/SVM_results.csv'


def main_ANN():
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
    if not os.path.exists(SVM_result_path):
        print("no such file: ", SVM_result_path)
        return
    SVM_exp_dir = os.listdir(SVM_exp_path)
    f = open(SVM_result_path, 'r')
    SVM_csv = csv.reader(f)
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

if __name__ == '__main__':
    # clean the dummy exp not in ANN_results.csv
    main_ANN()
    main_SVM()
    print('Successfully clean dummy exp!')
