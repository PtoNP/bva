import numpy as np
import pandas as pd
import params

def set_hit(hit_proba, threshold):
    if hit_proba >= threshold:
        return 1
    return 0

def set_final_hit(x, idxs):
    if x-6 in idxs:
        return 1
    return 0

def find_final_predict_from_hitnet(hitnet_predict_path, proba_threshold):
    hitnet_predict = pd.read_csv(hitnet_predict_path)
    df = hitnet_predict.rename( columns = {'index':'frame', '0': 'no_hit_proba', '1': 'hit_proba'})

    df['hit'] = df.apply(lambda x: set_hit(x['hit_proba'], proba_threshold), axis=1)

    start_idx = -1
    end_idx = -1
    hit_idxs = []
    for i in range(len(df)):
        hit = df.loc[i]['hit']
        if hit == 1:
            if start_idx > 0:
                end_idx = i
            else:
                start_idx = i
                end_idx = i
        else:
            if start_idx > 0:
                hit_idxs.append(int(start_idx + (end_idx - start_idx)/2))
                start_idx = -1
                end_idx = -1

    df['hit'] = df.apply(lambda x: set_final_hit(x['frame'], hit_idxs), axis=1)

    result = df.drop(columns=['no_hit_proba', 'hit_proba'])

    return result


if __name__ == "__main__":
    result = find_final_predict_from_hitnet('./data/hitnet_predict_match9_1_07_11.csv',
                             params.FINAL_PREDICT_PROBA_THRESHOLD)
    print(result[result['hit'] == 1])
