import numpy as np
import pandas as pd


def entropy(p: list):
    tot = sum(p)
    p = np.array(p).astype(dtype='float64')
    p /= tot
    entropy = -np.sum(p * np.log2(p))
    return entropy


def information_gain(parent, child):
    parent_entropy = entropy(parent)
    l_parent = float(sum(parent))

    partition_entropy = []

    for ele in child:
        l_child = float(sum(ele))
        part_ent = entropy(ele)

        curr_ent = l_child / l_parent * part_ent
        partition_entropy.append(curr_ent)

    final_entropy = sum(partition_entropy)
    ig = parent_entropy - final_entropy

    return ig


def get_ig_idx(X, y, col_names):
    ig_list = list()
    parent_uniques, parent_cnts = np.unique(y, return_counts=True)

    for i in range(X.shape[1]):
        curr = X[:, i]
        uq = np.unique(curr)
        children = list()
        for ele in uq:
            ele_idx = (curr == ele)
            curr_y = y[ele_idx]
            uniq, cnts = np.unique(curr_y, return_counts=True)
            # child = [[6], [1, 3]]
            children.append(cnts)

        e = information_gain(parent=parent_cnts, child=children)
        ig_list.append(e)

    ig_list = np.array(ig_list)
    print("col: ", col_names)
    print("gr: ", ig_list)
    max_idx = np.argmax(ig_list)

    return max_idx


def decision_tree_continuous(X, y, col_names, thresholds):
    # 마지막 column만 coontinuous descriptive features인 경우의 decision tree 계산
    continuous = np.array(X[:, -1], dtype=float)
    categorized = None

    # get T/F according to threshold
    for th in thresholds:
        curr_tf = (continuous <= th)
        if categorized is None:
            categorized = curr_tf.reshape(-1, 1)
            continue

        categorized = np.append(categorized, curr_tf.reshape(-1, 1), axis=1)

    X_tot = np.append(X[:, :-1], categorized, axis=1)
    # columns_to_change = [0, 2, 3, 4, 5]
    # X_tot[:, columns_to_change] = X_tot[:, columns_to_change].astype(bool)

    # X_tot[:, [0, 2, 3, 4, 5]].astype('bool')

    col_names_tot = col_names[:-1] + [col_names[-1] + str(th) for th in thresholds]

    max_idx = get_ig_idx(X=X_tot, y=y, col_names=col_names_tot)
    print(f"h1 node: idx {max_idx} {col_names_tot[max_idx]}")
    # h1 node: idx 3 ELEVATION2250

    # h2-1 node separation
    # data filtration by ELEVATION2250 - True
    to_remain = (X_tot[:, max_idx])
    X1 = X_tot[to_remain]
    X1 = np.delete(X1, max_idx, axis=1)
    y1 = y[to_remain]
    col_names.pop(max_idx)




def main_routine():
    df = pd.read_csv('../data/vegetation_new.csv')
    my_np = df.to_numpy()
    data = my_np.tolist()
    print(data)

    col_names = ['ID', 'STREAM', 'SLOPE', 'ELEVATION', 'VEGETATION']
    data = [[2, True, 'moderate', 300, 'riparian'],
            [4, False, 'steep', 1200, 'chapparal'],
            [3, True, 'steep', 1500, 'riparian'],
            [7, True, 'steep', 3000, 'chapparal'],
            [1, False, 'steep', 3900, 'chapparal'],
            [5, False, 'flat', 4450, 'conifer'],
            [6, True, 'steep', 5000, 'conifer']]


    data = np.array(data)
    X = data[:, 1:-1]
    y = data[:, -1]

    threshold = [750, 2250, 3450, 4725]

    decision_tree_continuous(X, y, col_names[1:-1], threshold)



if __name__ == '__main__':
    main_routine()