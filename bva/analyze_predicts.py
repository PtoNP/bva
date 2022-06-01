import numpy as np

def find_best_targets(classes, targets_probas, threshold):
    predicts = []

    for target_probas in targets_probas:
        selected_idxs = []
        selected_values = []
        count = 0
        for proba in target_probas:
            if proba > threshold:
                selected_idxs.append(count)
                selected_values.append(proba)
            count += 1
        if len(selected_values) > 0:
            max_value = max(selected_values)
            max_index = selected_values.index(max_value)
            idx_class = selected_idxs[max_index]
            predicts.append(classes[idx_class])
        else:
            predicts.append('no_hit')

    return predicts

if __name__ == '__main__':
    test_classes = ['a', 'b', 'c', 'd', 'e', 'f']
    test_targets_probas = [
        [0.1,0.2,0.3,0.4,0.5,0.6],
        [0.1,0.2,0.3,0.4,0.75,0.76],
        [0.1,0.2,0.3,0.74,0.76,0.75]
    ]

    print(find_best_targets(test_classes, test_targets_probas, 0.7))
