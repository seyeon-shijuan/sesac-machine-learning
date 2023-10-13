import numpy as np
import pandas as pd

def get_dataset():
    # x - stream, slope, elevation / y - vegetation
    stream = np.array([False, True, True, False, False, True, True])
    slope = np.array(['steep', 'moderate', 'steep', 'steep', 'flat', 'steep', 'steep'])
    elevation = np.array(['high', 'low', 'medium', 'medium', 'high', 'highest', 'high'])
    y = np.array(['chapparal', 'riparian', 'riparian', 'chaparal', 'conifer', 'conifer', 'chapparal'])

    X = np.column_stack((stream, slope, elevation))

    return X, y


def get_entropy(x, y):
    uniques, cnts = np.unique(x, return_counts=True)
    n_x = len(uniques)


def entropy_test():
    # the 3rd entropy
    p31 = 1/2
    p32 = 1/2

    h3 = -(p31 * np.log2(p31) + p32 * np.log2(p32))
    print(h3) # 1.0

    # the 4th entropy
    p41 = 1/2
    p42 = 1/4
    p43 = 1/4

    h4 = -(p41 * np.log2(p41) + p42 * np.log2(p42) + p43 * np.log2(p43))
    print(h4) # 1.5

    # the 5th entropy
    p51 = 1/3
    p52 = 1/3
    p53 = 1/3

    h5 = -(p51 * np.log2(p51) + p52 * np.log2(p52) + p53 * np.log2(p53))
    print(h5) # 1.584962500721156

    # the 6th entropy
    p6 = np.array([1 / 12 for _ in range(12)])
    p6_entropy_list = p6 * np.log2(p6)
    h6 = -np.sum(p6_entropy_list)
    print(h6) # 3.584962500721156


def entropy_test2():
    p1 = np.array([2/3, 1/3])
    p2 = np.array([1/3, 2/3])

    p1_entropy_list = p1 * np.log2(p1)
    p2_entropy_list = p2 * np.log2(p2)
    p1 = -np.sum(p1_entropy_list)
    p2 = -np.sum(p2_entropy_list)
    print(p1, p2)
    final = p1 * 3/6 + p2 * 3/6
    print(final)


def entropy_test3():
    p1 = np.array([1/2, 1/2])
    p2 = np.array([1/2, 1/2])

    p1_entropy_list = p1 * np.log2(p1)
    p2_entropy_list = p2 * np.log2(p2)
    p1 = -np.sum(p1_entropy_list)
    p2 = -np.sum(p2_entropy_list)
    print(p1, p2)
    final = p1 * 2/6 + p2 * 4/6
    print(final)


def entropy_test4():
    # 1
    # p1 = np.array([1/4, 2/4, 1/4])
    # p1_entropy_list = p1 * np.log2(p1)
    # p1 = -np.sum(p1_entropy_list)
    #
    # p2 = np.array([2/3, 1/3])
    # p2_entropy_list = p1 * np.log2(p2)
    # p2 = -np.sum(p2_entropy_list)
    #
    # final = p1 * 4/7 + p2 * 3/7
    # print(final)

    # 2
    # p1 = np.array([3/5, 1/5, 1/5])
    # p1_entropy_list = p1 * np.log2(p1)
    # p1 = -np.sum(p1_entropy_list)
    # final = p1 * 5/7
    # print(final)

    # 3
    p1 = np.array([2/3, 1/3])
    p1_entropy_list = p1 * np.log2(p1)
    p1 = -np.sum(p1_entropy_list)
    final = p1 * 3/7 + 2/7
    print(final)


def multi_class_entropy_h1():
    # root node
    probs = [3/7, 2/7, 2/7]
    probs = np.array(probs)
    h0 = -np.sum(probs * np.log2(probs))
    print('root node H: ', h0)
    # root node H:  1.5566567074628228

    # stream
    # true
    probs = [1/4, 1/4, 2/4]
    probs = np.array(probs)
    h1_str_t = -np.sum(probs * np.log2(probs))
    print('h1층의 피처 stream에서 true의 entropy: ', h1_str_t)
    # h1층의 피처 stream의 entropy:  1.5

    # false
    probs = [2/3, 1/3]
    probs = np.array(probs)
    h1_str_f = -np.sum(probs * np.log2(probs))
    print('h1층의 피처 stream에서 false의 entropy: ', h1_str_f)
    # h1층의 피처 stream에서 false의 entropy:  0.9182958340544896

    # final h1_str
    h1_str = 4/7 * h1_str_t + 3/7 * h1_str_f
    print('h1층의 피처 stream의 최종 entropy: ', h1_str)
    # h1층의 피처 stream의 최종 entropy:  1.2506982145947811

    # slope
    # flat - 0
    # moderate - 0
    # steep
    probs = [3/5, 1/5, 1/5]
    probs = np.array(probs)
    h1_slo_s = -np.sum(probs * np.log2(probs))
    print('h1층의 피처 slope에서 steep의 entropy: ', h1_slo_s)
    # h1층의 피처 slope에서 steep의 entropy:  1.3709505944546687

    # final h1_slo
    h1_slo = 1/7 * 0 + 1/7 * 0 + 5/7 * h1_slo_s
    print('h1층의 피처 slope의 최종 entropy: ', h1_slo)
    # h1층의 피처 slope의 최종 entropy:  0.9792504246104776


    # elevation
    # high
    probs = [2/3, 1/3]
    probs = np.array(probs)
    h1_ele_high = -np.sum(probs * np.log2(probs))
    print('h1층의 피처 elevation에서 high의 entropy: ', h1_ele_high)
    # h1층의 피처 elevation에서 high의 entropy:  0.9182958340544896

    # highest - 0
    # low - 0
    # medium
    probs = [1/2, 1/2]
    probs = np.array(probs)
    h1_ele_medium = -np.sum(probs * np.log2(probs))
    print('h1층의 피처 elevation에서 medium의 entropy: ', h1_ele_medium)
    # h1층의 피처 elevation에서 medium의 entropy:  1.0

    # final h1_ele
    h1_ele = 3/7 * h1_ele_high + 1/7 * 0 + 1/7 * 0 + 2/7 * h1_ele_medium
    print('h1층의 피처 elevation의 최종 entropy: ', h1_ele)
    # h1층의 피처 elevation의 최종 entropy:  0.6792696431662097


def multi_class_entropy_h2():
    '''high node'''
    # stream
    # false
    probs = [1/2, 1/2]
    probs = np.array(probs)
    h2_str_f = -np.sum(probs * np.log2(probs))
    print('h2층의 피처 stream에서 false의 entropy: ', h2_str_f)
    # h2층의 피처 stream에서 false의 entropy:  1.0

    # final h1_str
    h2_str = 1/3 * 0 + 2/3 * h2_str_f
    print('h2층의 피처 stream의 최종 entropy: ', h2_str)
    # h2층의 피처 stream의 최종 entropy:  0.6666666666666666

    '''medium node'''
    # slope
    # steep
    probs = [1 / 2, 1 / 2]
    probs = np.array(probs)
    h2_slo_s = -np.sum(probs * np.log2(probs))
    print('h2층의 피처 slope에서 steep의 entropy: ', h2_slo_s)
    # h2층의 피처 slope에서 steep의 entropy:  1.0


'''IGR'''

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


def get_igr(parent, child):
    ig = information_gain(parent=parent, child=child)
    child_all = [sum(x) for x in child]
    iv = entropy(child_all)
    igr = ig / iv
    return igr


def test_routine():
    parent = [7, 3]
    child = [[6], [1, 3]]
    igr = get_igr(parent=parent, child=child)
    print(igr)

    # parent = [7, 3]
    # child = [[1], [1], [1], [1], [1], [1], [1], [1], [1, 1]]
    # ig = information_gain(parent=parent, child=child)
    # print("ig: ", ig)


def get_igr_idx(X, y, col_names):
    ig_list = list()
    h_list = list()
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

    # get iv by features
    for i in range(X.shape[1]):
        curr = X[:, i]
        uniques, cnts = np.unique(curr, return_counts=True)

        curr_entropy = entropy(cnts)
        h_list.append(curr_entropy)

    gr_list = np.array(ig_list) / np.array(h_list)
    print("col: ", col_names)
    print("gr: ", gr_list)
    max_idx = np.argmax(gr_list)

    return max_idx


def decision_tree_by_igr(data, col_names):
    X = data[:, 1:-1]
    y = data[:, -1]
    col_names = col_names[1:-1]

    # root node separation
    max_idx = get_igr_idx(X, y, col_names)
    print(f"h1 node: idx {max_idx} {col_names[max_idx]}")
    # flat-fin, moderate-fin, steep-3types

    print("=" * 30)

    # data filtration by slope - steep
    to_remain = (X[:, max_idx] == 'steep')
    X1 = X[to_remain]
    X1 = np.delete(X1, max_idx, axis=1)
    y1 = y[to_remain]
    col_names.pop(max_idx)

    # h1 separation
    max_idx = get_igr_idx(X1, y1, col_names)
    print(f"h2 node: idx {max_idx} {col_names[max_idx]}")
    # low-filled with chapparal, medium-2types, high-fin, highest-fin

    print("=" * 30)

    # data filtration by elevation - medium
    to_remain = (X1[:, max_idx] == 'medium')
    X2 = X1[to_remain]
    X2 = np.delete(X2, max_idx, axis=1)
    y2 = y1[to_remain]
    col_names.pop(max_idx)

    # h2 separation
    max_idx = get_igr_idx(X2, y2, col_names)
    print(f"h3 node: idx {max_idx} {col_names[max_idx]}")


def main_routine():
    df = pd.read_csv('../data/vegetation.csv')
    col_names = df.columns.to_list()
    data = df.to_numpy()
    decision_tree_by_igr(data, col_names)



if __name__ == '__main__':
    # entropy_test()
    # entropy_test2()
    # entropy_test3()
    # get_entropy(X[:, 0], y)
    # entropy_test4()
    # multi_class_entropy_h1()
    # multi_class_entropy_h2()
    # X, y = get_dataset()
    # get_entropy(iv = [1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 2/10])
    # get_entropy(iv=[1/4, 3/4])
    # test_routine()
    # decision_tree_by_igr()
    main_routine()








