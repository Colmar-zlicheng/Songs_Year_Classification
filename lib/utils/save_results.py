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
        csv_writer.writerow(['ID', 'Epoch_size', 'Batch_size', 'learning_rate',
                             'decay_step', 'decay_gamma',
                             'train_acc', 'test_acc', 'exp'])
        total += 1
    if arg.decay_step > arg.epoch_size:
        decay_step = '--'
    else:
        decay_step = arg.decay_step
    exp = os.path.join('./exp/ANN', arg.exp_id)
    csv_writer.writerow([str(total), str(arg.epoch_size), str(arg.batch_size), str(arg.learning_rate),
                         str(decay_step), str(arg.decay_gamma),
                         str(train_acc)+'%', str(test_acc)+'%', str(exp)])
    file.close()