import csv

import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

labels = ["bee",
          "bumblebee",
          "butterfly",
          "fly",
          "hummingbird",
          "no pollinator"]


def get_label_index(label):
    for i, l in enumerate(labels):
        if l == label:
            return i
    return None


def read_csv(csv_file):
    labels = dict()
    with open(csv_file, mode='r') as file:
        # reading the CSV file
        csvFile = csv.reader(file)
        # displaying the contents of the CSV file
        for lines in csvFile:
            if len(lines) == 2:
                file, label = lines
                label_idx = get_label_index(label)
                if file not in labels:
                    labels[file] = label_idx
                else:
                    print('Somethign wrong...')
            elif len(lines) == 3:
                file, label, visited = lines
                label = int(label[1:])
                visited = visited[:-1]
                if visited == 'yes':
                    if file not in labels:
                        labels[file] = label
                    else:
                        print('Something wrong...')
                if visited == 'no':
                    if file not in labels:
                        labels[file] = 5
                    else:
                        print('Something wrong...')
    return labels


true_labels = read_csv('results/true_labels.csv')
pred_labels = read_csv('results/full_results.csv')

Y_true = []
Y_pred = []

for key in pred_labels:
    true_key = key+'.mp4'
    if true_key in true_labels:
        Y_pred.append(pred_labels[key])
        Y_true.append(true_labels[true_key])
    else:
        print(f'Not there {key}??')

Y_true = np.array(Y_true)
Y_pred = np.array(Y_pred)

cm = confusion_matrix(Y_true, Y_pred)

report = classification_report(Y_true, Y_pred, labels=np.arange(6), target_names=labels)
print(report)

print(cm)


