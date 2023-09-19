
def e114():
    # vector-vector operations
    x1, y1, z1 = 1, 2, 3
    x2, y2, z2 = 3, 4, 5

    x3, y3, z3 = x1 + x2, y1 + y2, z1 + z2
    x4, y4, z4 = x1 - x2, y1 - y2, z1 - z2
    x5, y5, z5 = x1 * x2, y1 * y2, z1 * z2

    print(x3, y3, z3)
    print(x4, y4, z4)
    print(x5, y5, z5)


def e115():
    # scalar-vector operations
    a = 10
    x1, y1, z1 = 1, 2, 3
    x2, y2, z2 = a*1, a*y1, a*z1
    x3, y3, z3 = a+x1, a+y1, a+z1
    x4, y4, z4 = a-x1, a-y1, a-z1


def e116():
    # vector norm: 벡터의 크기
    x, y, z = 1, 2, 3
    norm = (x**2 + y**2 + z**2)**0.5
    print(norm)


def e117():
    # unit vectors
    x, y, z = 1, 2, 3
    norm = (x**2 + y**2 + z**2)**0.5
    print(norm)

    x, y, z = x/norm, y/norm, z/norm
    norm = (x**2 + y**2 + z**2)**0.5
    print(norm)


def e118():
    # dot product (내적)
    x1, y1, z1 = 1, 2, 3
    x2, y2, z2 = 3, 4, 5
    dot_prod = x1*x2 + y1*y2 + z1*z2
    print(dot_prod)


def e119():
    # euclidean distance
    x1, y1, z1 = 1, 2, 3
    x2, y2, z2 = 3, 4, 5
    e_distance = (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2
    e_distance **= 0.5

    print(e_distance)


def e121():
    # mean squared error
    pred1, pred2, pred3 = 10, 20, 30
    y1, y2, y3 = 10, 25, 40
    n_data = 3

    s_error1 = (pred1 - y1)**2
    s_error2 = (pred2 - y2)**2
    s_error3 = (pred3 - y3)**2

    mse = (s_error1 + s_error2 + s_error3) / n_data
    print(mse)


def e122():
    # list로 mean ... 등 수식 구현
    scores = [10, 20, 30]
    print(scores[0])
    print(scores[1])
    print(scores[2])

    # e123
    scores[0] = 100
    scores[1] = 200
    print(scores)

    # e124
    scores = [10, 20, 30]
    n_student = len(scores)
    mean = (scores[0] + scores[1] + scores[2]) / n_student
    print("score mean: ", mean)

    # e125 mean subtraction 2
    scores[0] -= mean
    scores[1] -= mean
    scores[2] -= mean

    mean = (scores[0] + scores[1] + scores[2]) / n_student
    print("score mean: ", mean)


def e126():
    # 리스트로 분산과 표준편차 구하기
    scores = [10, 20, 30]
    n_student = len(scores)
    mean = (scores[0] + scores[1] + scores[2]) / n_student
    square_of_mean = mean**2 # 평균의 제곱
    mean_of_square = (scores[0]**2 + scores[1]**2 + scores[2]**2) / n_student # (각 항목의) 제곱의 평균

    variance = mean_of_square - square_of_mean # MOS - SOM
    std = variance**0.5 # square root of the variance

    print("score mean: ", mean)
    print("score standard deviation: ", std)

    # 127 standardization 2
    print("standardization")
    scores[0] = (scores[0] - mean)/std
    scores[1] = (scores[1] - mean)/std
    scores[2] = (scores[2] - mean)/std

    mean = (scores[0] + scores[1] + scores[2]) / n_student
    print("score mean: ", mean)
    mean_of_square = (scores[0]**2 + scores[1]**2 + scores[2]**2) / n_student
    square_of_mean = mean**2
    variance = mean_of_square - square_of_mean # MOS-SOM
    std = variance**0.5
    print("score standard deviation: ", std)


def e127():
    # hadamard product 2
    # method 1
    v1, v2 = [1, 2, 3], [3, 4, 5]
    v3 = [v1[0] * v2[0], v1[1] * v2[1], v1[2] * v2[2]]
    print(v3)

    # method 2
    v1, v2 = [1, 2, 3], [3, 4, 5]
    v3 = [0, 0, 0]
    v3[0] = v1[0] * v2[0]
    v3[1] = v1[1] * v2[1]
    v3[2] = v1[2] * v2[2]
    print(v3)


def e129():
    v1 = list()
    print(v1)
    v1.append(1)
    print(v1)
    v1.append(2)
    print(v1)


def e130():
    v1, v2 = [1, 2, 3], [3, 4, 5]

    v3 = list()
    v3.append(v1[0] * v2[0])
    v3.append(v1[1] * v2[1])
    v3.append(v1[2] * v2[2])

    print(v3)


def e131():
    # vector norm 2
    v1 = [1, 2, 3]
    # method 1
    norm = (v1[0]**2 + v1[1]**2 + v1[2]**2)**0.5
    print(norm)

    # method 2
    norm = 0
    norm += v1[0]**2
    norm += v1[1]**2
    norm += v1[2]**2
    norm **= 0.5
    print(norm)


def e132():
    # unit vector 2
    v1 = [1, 2, 3]
    norm = (v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2) ** 0.5
    print(norm)

    v1 = [v1[0]/norm, v1[1]/norm, v1[2]/norm]
    norm = (v1[0]**2 + v1[1]**2 + v1[2]**2) ** 0.5
    print(norm)


def e133():
    # dot product 2
    v1, v2 = [1, 2, 3], [3, 4, 5]

    # method 1
    dot_prod = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    print(dot_prod)

    # method 2
    dot_prod = 0
    dot_prod += v1[0]*v2[0]
    dot_prod += v1[1]*v2[1]
    dot_prod += v1[2]*v2[2]
    print(dot_prod)


def e134():
    # euclidean distance
    v1, v2 = [1, 2, 3], [3, 4, 5]

    e_distance = 0
    e_distance += (v1[0] - v2[0])**2
    e_distance += (v1[1] - v2[1])**2
    e_distance += (v1[2] - v2[2])**2
    e_distance **= 0.5
    print(e_distance)


def e135():
    predictions = [10, 20, 30]
    labels = [10, 25, 40]
    n_data = len(predictions)

    mse = 0
    # 제곱하고 빼면안돼고, 빼고난 뒤에 제곱해야됨
    mse += (predictions[0] - labels[0])**2
    mse += (predictions[1] - labels[1])**2
    mse += (predictions[2] - labels[2])**2

    mse /= n_data
    print(mse)


def e136():
    # 반복문
    scores = [10, 20, 30]

    score_sum = 0

    for score in scores:
        score_sum += score
        print(score_sum)


def e138():
    numbers = [1, 4, 5, 6, 4, 2, 1]
    iter_cnt = 0

    for _ in numbers:
        iter_cnt += 1

    print(iter_cnt)

    # 139 sum of 1 to 100 using for loop
    num_sum = 0
    for i in range(101):
        num_sum += i

    print(num_sum)

    # 140 1 부터 100까지 list
    numbers = list()
    for i in range(1, 101):
        numbers.append(i)

    print(numbers)

    # 141 100개의 0을 가진 list
    numbers = list()
    for _ in range(100):
        numbers.append(0)
    print(numbers)

    # 142 for loop 으로 list 원소 접근 2
    scores = [10, 20, 30]

    # method 1
    for score in scores:
        print(score)

    # method 2
    for score_idx in range(len(scores)):
        print(scores[score_idx])

    # 143 for loop으로 list 원소 수정
    scores = [10, 20, 30, 40, 50]

    for score_idx in range(len(scores)):
        scores[score_idx] += 10

    print(scores)

    # 144 두개의 list 접근
    list1 = [10, 20, 30]
    list2 = [100, 200, 300]

    for idx in range(len(list1)):
        print(list1[idx], list2[idx])


def e145():
    # 평균
    scores = [10, 20, 30]

    # method 1
    score_sum = 0
    n_student = 0
    for score in scores:
        score_sum += score
        n_student += 1

    score_mean = score_sum / n_student
    print("score mean: ", score_mean)

    # method 2
    score_sum = 0
    for score_idx in range(len(scores)):
        score_sum += scores[score_idx]

    score_mean = score_sum / len(scores)
    print("score mean: ", score_mean)

    # mean subtraction
    # method 1
    scores_ms = list()
    for score in scores:
        scores_ms.append(score - score_mean)
    print(scores_ms)

    # method 2
    scores_ms = list()
    for score_idx in range(len(scores)):
        scores[score_idx] -= score_mean

    print(scores)


def e147():
    # 각 과목의 평균 mean
    math_scores = [40, 60, 80]
    english_scores = [30, 40, 50]

    n_class = 2
    n_student = len(math_scores)

    score_sums = list()
    score_means = list()

    for _ in range(n_class):
        score_sums.append(0)

    for student_idx in range(n_student):
        score_sums[0] += math_scores[student_idx]
        score_sums[1] += english_scores[student_idx]

    print("sums of scores: ", score_sums)

    for class_idx in range(n_class):
        class_mean = score_sums[class_idx] / n_student
        score_means.append(class_mean)

    print("means of scores: ", score_means)




if __name__ == '__main__':
    # e114()
    # e115()
    # e116()
    # e117()
    # e118()
    # e119()
    # e121()
    # e122()
    # e126()
    # e127()
    # e129()
    # e130()
    # e131()
    # e132()
    # e133()
    # e134()
    # e135()
    # e136()
    # e138()
    # e145()
    e147()


