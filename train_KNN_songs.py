import os
import argparse
import torch
import time
import joblib
from lib.utils.logger import logger
from lib.dataset.Songs import Songs
from lib.utils.save_results import save_results_KNN
from sklearn.neighbors import KNeighborsClassifier


def KNN_worker(arg):
    dataset = Songs(train_size=arg.train_size, test_size=arg.test_size, seed=arg.seed)
    data = dataset.Songs_whole(train_size=arg.train_size, test_size=arg.test_size)

    knn = KNeighborsClassifier(n_neighbors=arg.K)
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
    knn.fit(data_train, data['year_train'])
    if arg.do_val is True:
        train_result = knn.predict(data_train)
        train_result = torch.tensor(train_result)
        correct = (train_result == data['year_train']).sum().item()
        val_acc = correct / data['year_train'].shape[0]
        print("Accuracy:", '%.2f' % (val_acc * 100), "%")
    else:
        val_acc = '--'
    print("Start testing...")
    test_result = knn.predict(data_test)
    test_result = torch.tensor(test_result)
    correct = (test_result == data['year_test']).sum().item()
    acc = correct / data['year_test'].shape[0]
    correct_age = ((test_result // arg.age) == (data['year_test'] // arg.age)).sum().item()
    age_acc = correct_age / data['year_test'].shape[0]
    print("Accuracy:", '%.2f' % (acc * 100), "%")
    print(f"Age {arg.age} Accuracy:", '%.2f' % (age_acc * 100), "%")
    save_results_KNN(arg=arg, acc=acc, correct=correct, val_acc=val_acc, age_acc=age_acc, age_correct=correct_age)
    save_model = os.path.join(save_dir, 'model.m')
    joblib.dump(knn, save_model)
    save_acc = os.path.join(save_dir, 'acc.txt')
    with open(save_acc, 'w') as ff:
        ff.write("Correct:" + str(correct) + '\n')
        ff.write("Accuracy:" + str('%.2f' % (acc * 100)) + '%' + '\n')
    save_age_acc = os.path.join(save_dir, f'acc_age{arg.age}.txt')
    with open(save_age_acc, 'w') as ff:
        ff.write("Correct_age:" + str(correct) + '\n')
        ff.write("Age Accuracy:" + str('%.2f' % (age_acc * 100)) + '%' + '\n')
    print("-----successfully save results-----")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size', type=int, default=2000)
    parser.add_argument('--test_size', type=int, default=200)
    parser.add_argument('-k', '--K', type=int, default=1024)
    parser.add_argument('--age', type=int, default=10)
    parser.add_argument('-t', '--type', type=str, default='avg', choices=['avg', 'cov', 'cat'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-exp', '--exp_id', type=str)
    parser.add_argument('-v', '--do_val', action='store_true', help="do validation with train set")

    arg = parser.parse_args()

    start = time.time()
    if not os.path.exists('./exp'):
        os.mkdir('./exp')
    if not os.path.exists('./exp/KNN'):
        os.mkdir('./exp/KNN')

    print("<<<<<Beginning training KNN>>>>>")
    if arg.exp_id is not None:
        arg.exp_id = arg.exp_id
    else:
        timestamp = time.strftime("%Y_%m%d_%H%M_%S", time.localtime(start))
        arg.exp_id = f"KNN_{arg.type}_K{arg.K}_{arg.train_size}+{arg.test_size}_{timestamp}"
    save_dir = os.path.join('./exp/KNN', arg.exp_id)
    logger.set_log_file(save_dir, arg.exp_id)
    save_set = os.path.join(save_dir, 'setting.txt')
    argsDict = arg.__dict__
    with open(save_set, 'w') as f:
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ': ' + str(value) + '\n')

    KNN_worker(arg)

    end = time.time()
    logger.info('Running time: %s Seconds' % (end - start))

"""
import random
import csv

# read_the_data
with open('data/songs.csv', 'r') as file:
    reader = csv.DictReader(file)
    datas = [row for row in reader]  # store in dictionary format

# years中储存有1965-2010年份数据的个数
years = ["0" for i in range(46)]
year_0 = 1965
year_0 = int(year_0)
for i in range(46):
    for data in datas:
        if data['year'] == str(year_0):
            years[i] = int(years[i]) + 1
    year_0 = int(year_0) + 1

# 每一年份的数据量差异很大，因此每一年选择1050+50个数据进行训练和测试
current_data_train = []
current_data_test = []
year_0 = 1965
year_0 = int(year_0)
for i in range(46):
    num = 1
    num = int(num)  ##计数器
    for data in datas:
        if data['year'] == str(year_0):
            if num <= 1060:
                current_data_train.append(data)
                num += 1
            elif num > 1060 and num <= 1100:
                current_data_test.append(data)
                num += 1
            else:
                break
    year_0 = int(year_0) + 1

# 将训练集和测试集打乱顺序
random.shuffle(current_data_train)
random.shuffle(current_data_test)
print(current_data_train)
assert False
# KNN
# caculate the distance
def distance(d1, d2):
    res = 0
    for key in (
    "timbre_avg_0", "timbre_avg_1", "timbre_avg_2", "timbre_avg_3", "timbre_avg_4", "timbre_avg_5", "timbre_avg_6",
    "timbre_avg_7", "timbre_avg_8", "timbre_avg_9", "timbre_avg_10", "timbre_avg_11",
    "timbre_cov_0", "timbre_cov_1", "timbre_cov_2", "timbre_cov_3", "timbre_cov_4", "timbre_cov_5", "timbre_cov_6",
    "timbre_cov_7", "timbre_cov_8", "timbre_cov_9", "timbre_cov_10", "timbre_cov_11",
    "timbre_cov_12", "timbre_cov_13", "timbre_cov_14", "timbre_cov_15", "timbre_cov_16", "timbre_cov_17",
    "timbre_cov_18", "timbre_cov_19", "timbre_cov_20", "timbre_cov_21", "timbre_cov_22", "timbre_cov_23",
    "timbre_cov_24", "timbre_cov_25", "timbre_cov_26", "timbre_cov_27", "timbre_cov_28", "timbre_cov_29",
    "timbre_cov_30", "timbre_cov_31", "timbre_cov_32", "timbre_cov_33", "timbre_cov_34", "timbre_cov_35",
    "timbre_cov_36", "timbre_cov_37", "timbre_cov_38", "timbre_cov_39", "timbre_cov_40", "timbre_cov_41",
    "timbre_cov_42", "timbre_cov_43", "timbre_cov_44", "timbre_cov_45", "timbre_cov_46", "timbre_cov_47",
    "timbre_cov_48", "timbre_cov_49", "timbre_cov_50", "timbre_cov_51", "timbre_cov_52", "timbre_cov_53",
    "timbre_cov_54", "timbre_cov_55", "timbre_cov_56", "timbre_cov_57", "timbre_cov_58", "timbre_cov_59",
    "timbre_cov_60", "timbre_cov_61", "timbre_cov_62", "timbre_cov_63", "timbre_cov_64", "timbre_cov_65",
    "timbre_cov_66", "timbre_cov_67", "timbre_cov_68", "timbre_cov_69", "timbre_cov_70", "timbre_cov_71",
    "timbre_cov_72", "timbre_cov_73", "timbre_cov_74", "timbre_cov_75", "timbre_cov_76", "timbre_cov_77"):
        res += (float(d1[key]) - float(d2[key])) ** 2
    return res ** 0.5


K = 1024


def KNN(data):
    # 1_distance
    res = [
        {"result": train['year'], "distance": distance(data, train)}
        for train in current_data_train
    ]

    # 2_sort in accending order
    res = sorted(res, key=lambda item: item['distance'])

    # 3_get_K_data
    res2 = res[0:K]

    # 4_Weighted average
    result = {'1965': 0, '1966': 0, '1967': 0, '1968': 0, '1969': 0, '1970': 0, '1971': 0,
              '1972': 0, '1973': 0, '1974': 0, '1975': 0, '1976': 0, '1977': 0, '1978': 0,
              '1979': 0, '1980': 0, '1981': 0, '1982': 0, '1983': 0, '1984': 0, '1985': 0, '1986': 0,
              '1987': 0, '1988': 0, '1989': 0, '1990': 0, '1991': 0, '1992': 0, '1993': 0, '1994': 0,
              '1995': 0, '1996': 0, '1997': 0, '1998': 0, '1999': 0, '2000': 0, '2001': 0, '2002': 0,
              '2003': 0, '2004': 0, '2005': 0, '2006': 0, '2007': 0, '2008': 0, '2009': 0, '2010': 0,
              }

    sum = 0
    for r in res2:
        sum += r['distance']

    for r in res2:
        result[r['result']] += 1 - r['distance'] / sum

    list1 = sorted(result.items(), key=lambda item: item[1], reverse=True)
    # print(list1[0][0])
    return list1[0][0]


correct = 0
for test in current_data_test:
    result = test['year']
    result2 = KNN(test)
    if result[0:3] == result2[0:3]:
        correct += 1
        print("right!")
    else:
        print("wrong!")

print("训练集数量： " + str(len(current_data_train)))
print("测试集数量： " + str(len(current_data_test)))
print("准确率：{:.2f}%".format(100 * correct / len(current_data_test)))
"""
