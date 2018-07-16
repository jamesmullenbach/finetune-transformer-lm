import os
import json
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix

from datasets import _rocstories, read_pws

def rocstories(data_dir, pred_path, log_path, **kwargs):
    preds = pd.read_csv(pred_path, delimiter='\t')['prediction'].values.tolist()
    _, _, _, labels = _rocstories(os.path.join(data_dir, 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv'))
    test_accuracy = accuracy_score(labels, preds)*100.
    logs = [json.loads(line) for line in open(log_path)][1:]
    best_validation_index = np.argmax([log['va_acc'] for log in logs])
    valid_accuracy = logs[best_validation_index]['va_acc']
    print('ROCStories Valid Accuracy: %.2f'%(valid_accuracy))
    print('ROCStories Test Accuracy:  %.2f'%(test_accuracy))

def pw(data_dir, pred_path, log_path, test=False, ordinal=False):
    preds = pd.read_csv(pred_path, delimiter='\t')['prediction'].values.tolist()
    fold = 'test' if test else 'dev'
    _, _, labels = read_pws(os.path.join(data_dir, f'snli_style_{fold}_feats.jsonl'), ordinal)
    acc = accuracy_score(labels, preds)*100.
    logs = [json.loads(line) for line in open(log_path)][1:]
    best_validation_index = np.argmax([log['va_acc'] for log in logs])
    val_acc = logs[best_validation_index]['va_acc']
    print(f"PW valid acc: {val_acc: %2.2f}")
    print(f"PW {fold} acc: {val_acc: %2.2f}")
    print(confusion_matrix(labels, preds))
