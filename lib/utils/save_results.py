import os
import csv


def save_results_ANN(arg, train_acc, test_acc):
    if not os.path.exists('./results'):
        os.mkdir('./results')
    log_path = './results/ANN_results.csv'
    file = open(log_path, 'a+', encoding='utf-8', newline='')
    csv_writer = csv.writer(file)

    total = len(open(log_path).readlines())
    if total == 0:
        csv_writer.writerow(['ID', 'Model', 'Base_Type',
                             'Dataset', 'Train_Size', 'Test_Size',
                             'Epoch_size', 'Batch_size', 'learning_rate',
                             'decay_step', 'decay_gamma',
                             'train_acc', 'test_acc', 'exp'])
        total += 1
    if arg.decay_step > arg.epoch_size:
        decay_step = '--'
    else:
        decay_step = arg.decay_step
    if arg.big_dataset is True:
        dataset_name = 'Big'
    else:
        dataset_name = 'Small'

    exp = os.path.join('./exp/ANN', arg.exp_id)
    csv_writer.writerow([str(total), str(arg.model), str(arg.base_type),
                         str(dataset_name), str(arg.train_size), str(arg.test_size),
                         str(arg.epoch_size), str(arg.batch_size), str(arg.learning_rate),
                         str(decay_step), str(arg.decay_gamma),
                         str(train_acc)+'%', str(test_acc)+'%', str(exp)])
    file.close()


def save_results_SVM(arg, acc, correct):
    if not os.path.exists('./results'):
        os.mkdir('./results')
    log_path = './results/SVM_results.csv'
    file = open(log_path, 'a+', encoding='utf-8', newline='')
    csv_writer = csv.writer(file)

    total = len(open(log_path).readlines())
    if total == 0:
        csv_writer.writerow(['ID', 'type', 'train_size', 'test_size',
                             'C', 'kernel',
                             'acc', 'correct', 'exp'])
        total += 1
    exp = os.path.join('./exp/SVM', arg.exp_id)
    csv_writer.writerow([str(total), str(arg.type), str(arg.train_size), str(arg.test_size),
                         str(arg.C), str(arg.kernel_type),
                         str('%.2f' % (acc*100))+'%', str(correct), str(exp)])
    file.close()
