
def e189(vectors=None):
    # hadamard product
    if vectors is None:
        vectors = [[1, 11, 21],
                   [2, 12, 22],
                   [3, 13, 23],
                   [4, 14, 24]]

    hp_list = list()
    for vector in vectors:
        hp = 1
        for e in vector:
            hp *= e

        hp_list.append(hp)

    print("hp_list: ", hp_list)
    return hp_list


def e190(vectors=None):
    # vector norm
    if vectors is None:
        vectors = [[1, 11, 21],
                   [2, 12, 22],
                   [3, 13, 23],
                   [4, 14, 24]]

    squared = [[x**2 for x in vector] for vector in vectors]
    vec_norms = [0 for x in range(len(vectors[0]))]

    for vector in squared:
        for i, e in enumerate(vector):
            vec_norms[i] += e

    vec_norms = [x**0.5 for x in vec_norms]
    print("vec_norms: ", vec_norms)
    return vec_norms


def e191(vectors=None):
    # unit vectors
    if vectors is None:
        vectors = [[1, 11, 21],
                   [2, 12, 22],
                   [3, 13, 23],
                   [4, 14, 24]]

    norms = e190(vectors)

    unit_vectors = [[e / norms[i] for i, e in enumerate(vector)] for vector in vectors]
    # print(unit_vectors)
    final_norms = e190(unit_vectors)
    print(final_norms)
    return final_norms


def e192():
    vectors = [[1, 11],
               [2, 12],
               [3, 13],
               [4, 14]]

    dot_product = 0
    for vector in vectors:
        multiplied = 1
        for e in vector:
            multiplied *= e

        dot_product += multiplied

    print(dot_product)


def e193():
    vectors = [[1, 11],
               [2, 12],
               [3, 13],
               [4, 14]]

    element = 0

    for vector in vectors:
        subtracted = vector[0] - vector[1]
        if subtracted < 0:
            subtracted = -subtracted

        element += subtracted**2

    ed = element**0.5
    print(ed)


def e194():
    # 최고점(국,영,수)
    scores = [[10, 40, 20],
              [50, 20, 60],
              [70, 40, 30],
              [30, 80, 40]]

    max_scores = [0 for x in range(len(scores[0]))]
    max_score_indices = max_scores.copy()

    for num, row in enumerate(scores):
        for i, score in enumerate(row):
            if score > max_scores[i]:
                max_scores[i] = score
                max_score_indices[i] = num

    print("max_scores :", max_scores)
    print("max_score indices: ", max_score_indices)


def e195(labels=None):
    # one hot encoding

    if labels is None:
        labels = [0, 1, 2, 1, 0, 3]

    n_label = len(labels)
    n_class = 0

    for label in labels:
        if label > n_class:
            n_class = label

    n_class += 1

    one_hot_mat = list()
    for label in labels:
        one_hot_vec = list()
        for _ in range(n_class):
            one_hot_vec.append(0)

        one_hot_vec[label] = 1

        one_hot_mat.append(one_hot_vec)

    print(one_hot_mat)
    return one_hot_mat


def e196():
    # accuracy
    predictions = [[1, 0, 0, 0], [0, 0, 1, 0],
                   [0, 0, 1, 0], [1, 0, 0, 0],
                   [1, 0, 0, 0], [0, 0, 0, 1]]

    labels = [0, 1, 2, 1, 0, 3]

    one_hot_mat = e195(labels)

    n_correct = 0
    for y_hat, y in zip(predictions, one_hot_mat):
        if y_hat == y:
            n_correct += 1

    accuracy = n_correct / len(predictions)
    print("accuracy: ", accuracy)


def e197():
    # matrix addition
    mat1 = [[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]]

    mat2 = [[11, 12, 13],
            [14, 15, 16],
            [17, 18, 19]]

    added_matrix = list()

    for row1, row2 in zip(mat1, mat2):
        multiplied = list()
        for rec1, rec2 in zip(row1, row2):
            multiplied.append(rec1+rec2)

        added_matrix.append(multiplied)

    print(added_matrix)


def e198():
    # matrix-vector multiplication
    mat = [[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]]

    vec = [10, 20, 30]

    multiplied_vec = list()

    for row in mat:
        col_sum = 0
        for i, e in enumerate(row):
            col_sum += e * vec[i]

        multiplied_vec.append(col_sum)

    print(multiplied_vec)


def e199():
    # matrix-matrix multiplication
    mat1 = [[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]]

    mat2 = [[11, 12, 13],
            [14, 15, 16],
            [17, 18, 19]]

    result = [[0 for y in range(len(mat2[0]))] for x in range(len(mat1))]

    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]

    print(result)


def e1100():
    mat1 = [[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]]

    result = [[0 for y in range(len(mat1[0]))] for x in range(len(mat1))]

    for i in range(len(mat1)):
        for j in range(len(mat1[0])):
            result[j][i] = mat1[i][j]

    print('result: ', result)


if __name__ == '__main__':
    # e189()
    # e190()
    # e191()
    # e192()
    # e193()
    # e194()
    # e195()
    # e196()
    # e197()
    # e198()
    # e199()
    e1100()


