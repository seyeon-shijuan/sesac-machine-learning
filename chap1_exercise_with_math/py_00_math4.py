

def e177():
    # accuracy
    predictions = [0, 1, 0, 2, 1, 2, 0]
    labels = [1, 1, 0, 0, 1, 2, 1]
    n_correct = 0

    for pred_idx in range(len(predictions)):
        if predictions[pred_idx] == labels[pred_idx]:
            n_correct += 1


    accuracy = n_correct / len(predictions)
    print("accuracy[%]: ", accuracy*100, '%')


def e178():
    # confusion vector
    predictions = [0, 1, 0, 2, 1, 2, 0]
    labels = [1, 1, 0, 0, 1, 2, 1]

    n_classes = None
    for label in labels:
        if n_classes == None or label > n_classes:
            n_classes = label

    n_classes += 1

    class_cnts, correct_cnts, confusion_vec = list(), list(), list()
    for _ in range(n_classes):
        class_cnts.append(0)
        correct_cnts.append(0)
        confusion_vec.append(None)

    for pred_idx in range(len(predictions)):
        pred = predictions[pred_idx]
        label = labels[pred_idx]

        class_cnts[label] += 1
        if pred == label:
            correct_cnts[label] += 1

    for class_idx in range(n_classes):
        confusion_vec[class_idx] = correct_cnts[class_idx]/class_cnts[class_idx]

    print("confusion vector: ", confusion_vec)


def e179():
    # histogram
    scores = [50, 20, 30, 40, 10, 50, 70, 80, 90, 20, 30]
    cutoffs = [0, 20, 40, 60, 80]
    histogram = [0, 0, 0, 0, 0]

    for score in scores:
        if score > cutoffs[4]:
            histogram[4] += 1
        elif score > cutoffs[3]:
            histogram[3] += 1
        elif score > cutoffs[2]:
            histogram[2] += 1

        elif score > cutoffs[1]:
            histogram[1] += 1
        elif score > cutoffs[0]:
            histogram[0] += 1
        else:
            pass

    print("histogram of the scores: ", histogram)


def e180():
    # abs
    numbers = [-2, 2, -1, 3, -4, 9]
    abs_numbers = list()

    for num in numbers:
        if num < 0:
            abs_numbers.append(-num)
        else:
            abs_numbers.append(num)

    print(abs_numbers)


def e181():
    # manhattan distance
    v1 = [1, 3, 5, 2, 1, 5, 2]
    v2 = [2, 3, 1, 5, 2, 1, 3]

    m_distance = 0
    for dim_idx in range(len(v1)):
        sub = v1[dim_idx] - v2[dim_idx]
        if sub < 0:
            m_distance += -sub

        else:
            m_distance += sub
    print("manhattan distance: ", m_distance)


def e182():
    # nested list
    scores = [[10, 20, 30], [50, 60, 70]]
    print(scores)
    print(scores[0])
    print(scores[1])
    print(scores[0][0], scores[0][1], scores[0][2])
    print(scores[1][0], scores[1][1], scores[1][2])


def e183():
    # nested list 원소 접근
    scores = [[10, 20, 30], [50, 60, 70]]

    for student_scores in scores:
        print(student_scores)
        for score in student_scores:
            print(score)


def e184():
    # 학생별 평균점수 구하기
    scores = [[10, 15, 20], [20, 25, 30], [30, 35, 40], [40, 45, 50]]

    n_class = len(scores[0])
    student_score_means = list()

    for student in scores:
        score_sum = 0
        for score in student:
            score_sum += score

        student_score_means.append(score_sum / n_class)

    print("mean of students' scores: ", student_score_means)


def e185(scores=None):
    # 과목별 평균
    if scores is None:
        scores = [[10, 15, 20], [20, 25, 30], [30, 35, 40], [40, 45, 50]]

    col_sums = [0 for x in range(len(scores[0]))]

    n_cols = len(scores[0])
    rows = len(scores)

    for i in range(n_cols):
        for j in range(rows):
            col_sums[i] += scores[j][i]

    col_means = [x/rows for x in col_sums]

    print("sum of classes' scores: ", col_sums)
    print("mean of classes' scores: ", col_means)

    return col_means


def e186():
    def e185_1(scores=None):
        # 과목별 평균
        if scores is None:
            scores = [[10, 15, 20], [20, 25, 30], [30, 35, 40], [40, 45, 50]]

        col_sums = [0 for x in range(len(scores[0]))]

        n_cols = len(scores[0])
        rows = len(scores)

        for i in range(n_cols):
            for j in range(rows):
                col_sums[i] += scores[j][i]

        col_means = [x / rows for x in col_sums]

        print("sum of classes' scores: ", col_sums)
        print("mean of classes' scores: ", col_means)

        return n_cols, rows, col_sums, col_means

    # 과목별 평균 mean subtraction
    scores = [[10, 15, 20], [20, 25, 30], [30, 35, 40], [40, 45, 50]]

    n_cols, rows, col_sums, col_means = e185_1(scores)

    scores_ms = list()
    for idx, student in enumerate(scores):
        m_subtracted = [x-col_means[i] for i, x in enumerate(student)]
        scores_ms.append(m_subtracted)

    print("="*20)
    print("mean subtracted scores: ", scores_ms)

    _, _, col_sums_ms, col_means_ms = e185_1(scores_ms)

    '''    
    sum of classes' scores:  [100, 120, 140]
    mean of classes' scores:  [25.0, 30.0, 35.0]
    ====================
    mean subtracted scores:  [[-15.0, -15.0, -15.0], [-5.0, -5.0, -5.0], [5.0, 5.0, 5.0], [15.0, 15.0, 15.0]]
    sum of classes' scores:  [0.0, 0.0, 0.0]
    mean of classes' scores:  [0.0, 0.0, 0.0]
    '''


def e187():
    def e185_2(scores=None):
        # 과목별 평균
        if scores is None:
            scores = [[10, 15, 20], [20, 25, 30], [30, 35, 40], [40, 45, 50]]

        col_sums = [0 for x in range(len(scores[0]))]

        n_cols = len(scores[0])
        rows = len(scores)

        for i in range(n_cols):
            for j in range(rows):
                col_sums[i] += scores[j][i]

        col_means = [x / rows for x in col_sums]

        # print("sum of classes' scores: ", col_sums)
        # print("mean of classes' scores: ", col_means)

        return n_cols, rows, col_sums, col_means

    #################
    #  분산 표준편차  #
    #################
    scores = [[10, 15, 20], [20, 25, 30], [30, 35, 40], [40, 45, 50]]

    # MOS
    squared = [[score**2 for score in student] for student in scores]
    _, _, _, moss = e185_2(squared)

    # SOM
    _, _, _, col_means = e185_2(scores)
    soms = [x**2 for x in col_means]

    variances = [m - s for m, s in zip(moss, soms)]
    stds = [v**0.5 for v in variances]

    print("variance values: ", variances)
    print("std values: ", stds)

    # variance values:  [125.0, 125.0, 125.0]
    # std values:  [11.180339887498949, 11.180339887498949, 11.180339887498949]


class MathBase:
    def __init__(self, scores=None):
        if scores is None:
            self.scores = [[10, 15, 20], [20, 25, 30], [30, 35, 40], [40, 45, 50]]
        else:
            self.scores = scores

        self.standardized = self.standardize(self.scores)

    def get_sum_mean(self, scores=None):
        if scores is None:
            scores = self.scores

        col_sums = [0 for x in range(len(scores[0]))]

        n_cols = len(scores[0])
        rows = len(scores)

        for i in range(n_cols):
            for j in range(rows):
                col_sums[i] += scores[j][i]

        col_means = [x / rows for x in col_sums]

        return n_cols, rows, col_sums, col_means

    def get_var_std(self, scores=None):
        if scores is None:
            scores = self.scores
        # MOS
        squared = [[score ** 2 for score in student] for student in scores]
        _, _, _, moss = self.get_sum_mean(squared)

        # SOM
        _, _, _, col_means = self.get_sum_mean(scores)
        soms = [x ** 2 for x in col_means]

        variances = [m - s for m, s in zip(moss, soms)]
        stds = [v ** 0.5 for v in variances]

        return variances, stds

    def standardize(self, scores=None):
        if scores is None:
            scores = self.scores

        _, _, _, col_means = self.get_sum_mean(scores)
        _, stds = self.get_var_std(scores)

        standardized = [[(score - col_means[idx]) / stds[idx] for idx, score in enumerate(student)]
                        for student in scores]

        return standardized


def e188():
    # standardization
    scores = [[10, 15, 20], [20, 25, 30], [30, 35, 40], [40, 45, 50]]
    base = MathBase(scores)
    standardized = base.standardize()
    print("standardized scores: ", standardized)

    var, std = base.get_var_std(standardized)
    print("vars and stds after standardization: ", var, std)

    '''
    standardized scores:  [[-1.3416407864998738, -1.3416407864998738, -1.3416407864998738], 
                            [-0.4472135954999579, -0.4472135954999579, -0.4472135954999579], 
                            [0.4472135954999579, 0.4472135954999579, 0.4472135954999579], 
                            [1.3416407864998738, 1.3416407864998738, 1.3416407864998738]]
    vars and stds after standardization:  [1.0, 1.0, 1.0] [1.0, 1.0, 1.0]
    '''


if __name__ == '__main__':
    e177()
    print("#############"*2)
    e178()
    print("#############" * 2)
    e179()
    print("#############" * 2)
    e180()
    print("#############" * 2)
    e181()
    print("#############" * 2)
    e182()
    print("#############" * 2)
    e183()
    print("#############" * 2)
    e184()
    print("#############" * 2)
    e185()
    print("#############" * 2)
    e186()
    print("#############" * 2)
    e187()
    print("#############" * 2)
    e188()


