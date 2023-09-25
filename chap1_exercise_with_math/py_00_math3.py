def e147():
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

    for student_idx in range(n_student):
        math_scores[student_idx] -= score_means[0]
        english_scores[student_idx] -= score_means[1]

    print("Math scores after mean subtraction: ", math_scores)
    print("English scores after mean subtraction: ", english_scores)

    tmp = 0
    for i in math_scores:
        tmp += i

    print(tmp / n_student)


def e149():
    scores = [10, 20, 30]
    n_student = len(scores)
    score_sum, score_square_sum = 0, 0

    for score in scores:
        score_sum += score
        score_square_sum += score**2

    mean = score_sum / n_student
    mean_of_square = score_square_sum / n_student
    square_of_mean = mean**2

    # mos - som
    variance = mean_of_square - square_of_mean
    std = variance**0.5

    print("variancee: ", variance)
    print("standard deviation: ", std)


def e150():
    scores = [10, 20, 30]
    n_student = len(scores)
    score_sum, score_square_sum = 0, 0

    for score in scores:
        score_sum += score
        score_square_sum += score ** 2

    mean = score_sum / n_student
    mean_of_square = score_square_sum / n_student
    square_of_mean = mean ** 2

    # mos - som
    variance = mean_of_square - square_of_mean
    std = variance ** 0.5

    mean = score_sum / n_student

    for student_idx in range(n_student):
        scores[student_idx] = (scores[student_idx] - mean) / std

    print(scores)

    mean = (scores[0] + scores[1] + scores[2]) / n_student

    score_sum = 0
    score_square_sum = 0

    for score in scores:
        score_sum += score
        score_square_sum += score **2

    mean_of_square = score_square_sum / n_student
    square_of_mean = mean ** 2

    variance = mean_of_square - square_of_mean
    std = variance ** 0.5

    print('mean: ', mean)
    print('standard deviation: ', std)


def e151():
    # 분산과 표준편차
    math_scores, english_scores = [50, 60, 70], [30, 40, 50]
    n_student = len(math_scores)

    math_sum, english_sum = 0, 0
    math_square_sum, english_square_sum = 0, 0

    for student_idx in range(n_student):
        math_sum += math_scores[student_idx]
        math_square_sum += math_scores[student_idx]**2

        english_sum += english_scores[student_idx]
        english_square_sum += english_scores[student_idx]**2

    math_mean = math_sum / n_student
    english_mean = english_sum / n_student

    # mean of square - square of mean
    math_variance = math_square_sum / n_student - math_mean**2
    english_variance = english_square_sum / n_student - english_mean**2

    math_std = math_variance**0.5
    english_std = english_variance**0.5

    print("mean/std of Math: ", math_mean, math_std)
    print("mean/std of English: ", english_mean, english_std)


def get_sum_square_mean(val):
    my_sum = 0
    my_sum_square = 0
    for i in val:
        my_sum += i
        my_sum_square += i**2

    return my_sum, my_sum_square, (my_sum/len(val))


def standardize(val):
    my_sum, my_sum_square, my_mean = get_sum_square_mean(val)

    mean_of_square = my_sum_square / len(val)
    square_of_mean = my_mean**2
    variance = mean_of_square - square_of_mean
    std = variance**0.5

    new = [(i - my_mean)/std for i in val]

    my_sum, my_sum_square, my_mean = get_sum_square_mean(new)
    mean_of_square = my_sum_square / len(val)
    square_of_mean = my_mean ** 2
    variance = mean_of_square - square_of_mean
    std = variance ** 0.5

    return new, my_mean, std


def e152():
    math_scores, english_scores = [50, 60, 70], [30, 40, 50]
    n_student = len(math_scores)
    math_sum, math_square, math_mean = get_sum_square_mean(math_scores)
    englsih_sum, english_square, english_mean = get_sum_square_mean(english_scores)

    for student_idx in range(n_student):
        math_scores[student_idx] = (math_scores[student_idx] - math_mean)
        english_scores[student_idx] = (english_scores[student_idx] - english_mean)

    # standardization
    math_scores, math_mean, math_std = standardize(math_scores)
    english_scores, english_mean, english_std = standardize(english_scores)

    print("Math scores after standardization: ", math_scores)
    print("English scores after standardization: ", english_scores)

    print("mean/std of Math: ", math_mean, math_std)
    print("mean/std of English: ", english_mean, english_std)


def e153():
    # hadamard product 원소별 곱셈
    v1 = [1, 2, 3, 4, 5]
    v2 = [10, 20, 30, 40, 50]

    # method 1
    v3 = list()
    for dim_idx in range(len(v1)):
        v3.append(v1[dim_idx] * v2[dim_idx])

    print(v3)

    # method 2
    v3 = list()
    for _ in range(len(v1)):
        v3.append(0)

    for dim_idx in range(len(v1)):
        v3[dim_idx] = v1[dim_idx] * v2[dim_idx]

    print(v3)


def e154(v1=None):
    # get norm
    if v1 is None:
        v1 = [1, 2, 3]

    square_sum = 0
    for dim_val in v1:
        square_sum += dim_val**2
    norm = square_sum**0.5
    print("in method norm of v1: ", norm)
    return norm


def e155():
    v1 = [1, 2, 3]
    norm = e154(v1)
    print("initial norm: ", norm)

    for dim_idx in range(len(v1)):
        v1[dim_idx] /= norm

    norm = e154(v1)
    print("new norm of v1: ", norm)


def e157():
    # euclidean distance
    v1, v2 = [1, 2, 3], [3, 4, 5]

    diff_square_sum = 0
    for dim_idx in range(len(v1)):
        diff_square_sum += (v1[dim_idx] - v2[dim_idx])**2

    e_distance = diff_square_sum**0.5
    print("euclidean distance between v1 and v2: ", e_distance)


def e158():
    predictions = [10, 20, 30]
    labels = [10, 25, 40]

    n_data = len(predictions)
    diff_square_sum = 0

    for data_idx in range(n_data):
        diff_square_sum += (predictions[data_idx] - labels[data_idx])**2

    mse = diff_square_sum / n_data
    print("MSE: ", mse)


def e159():
    numbers = [0, 2, 4, 2, 1, 4, 3, 1, 2, 3, 4, 1, 2, 3, 4]
    number_cnt = [0, 0, 0, 0, 0]

    for num in numbers:
        number_cnt[num] += 1

    print(number_cnt)


def e160():
    score = 60
    if score > 50:
        print("pass")


def e161():
    score = 40
    cutoff = 50

    if score > cutoff:
        print("pass")
    else:
        print("try again")


def e162():
    seconds = 200

    if seconds >= 60:
        minutes = seconds // 60
        seconds -= minutes*60

    else:
        minutes = 0

    print(minutes, "min", seconds, "sec")


def e163(seconds):
    if seconds is None:
        seconds = 5000

    ori_sec = seconds

    if seconds >= 60*60:
        hours = seconds // (60*60)
        seconds -= hours * (60*60)
        minutes = seconds // 60
        seconds -= minutes * 60
        print(f"{ori_sec} seconds is ", hours, "hours", minutes, "min", seconds, "sec")

    elif seconds >= 60:
        minutes = seconds // 60
        seconds -= minutes * 60
        print(f"{ori_sec} seconds is ", minutes, "min", seconds, "sec")

    else:
        print(f"{ori_sec} seconds is ", seconds, "seconds")


def e164():
    number = 10
    if number % 2 == 0:
        print('even')

    else:
        print("odd")


def e165():
    num1, num2 = 10, 10

    if num1 > num2:
        print("first number")

    elif num1 == num2:
        print("equal")

    else:
        print("second number")


def e169():
    scores = [20, 50, 10, 60, 90]
    cutoff = 50

    p_score_sum, n_p = 0, 0
    np_score_sum, n_np = 0, 0

    for score in scores:
        if score > cutoff:
            p_score_sum += score
            n_p += 1
        else:
            np_score_sum += score
            n_np += 1

    p_score_mean = p_score_sum / n_p
    np_score_mean = np_score_sum / n_np

    print("mean of passed scores: ", p_score_mean)
    print("mean of no passed scores: ", np_score_mean)


def e170():
    numbers = list()

    for num in range(10):
        numbers.append(num)

    numbers.append(3.14)
    print(numbers)

    for num in numbers:
        if num % 2 == 0:
            print('even number')
        elif num % 2 == 1:
            print("odd number")
        else:
            print('not an integer')


def e171():
    multiple_of = 3

    numbers = list()
    for num in range(100):
        numbers.append(num)


    sum_multiple_of_n = 0
    for num in numbers:
        if num % multiple_of == 0:
            sum_multiple_of_n += num

    print(sum_multiple_of_n)


def e172():
    scores = [60, 40, 70, 20, 30]
    M, m = 0, 100

    for score in scores:
        if score > M:
            M = score

        if score < m:
            m = score

    print("Max value: ", M)
    print("Min value: ", m)


def e173():
    scores = [-20, 60, 40, 70, 120]

    # method 1
    M, m = scores[0], scores[0]
    for score in scores:
        if score > M:
            M = score
        if score < m:
            m = score

    print("Max value: ", M)
    print("min value: ", m)

    # method 2
    M, m = None, None

    for score in scores:
        if M == None or score > M:
            M = score
        if m == None or score < m:
            m = score

    print("Max value: ", M)
    print("min value: ", m)


def e174():
    # min max normalization
    scores = [-20, 60, 40, 70, 120]
    # 최댓값, 최솟값
    # method 1
    M, m = scores[0], scores[0]
    for score in scores:
        if score > M:
            M = score
        if score < m:
            m = score

    print("Max value: ", M)
    print("Min value: ", m)

    for score_idx in range(len(scores)):
        scores[score_idx] = (scores[score_idx] - m) / (M-m)

    print("scores after normalization: \n", scores)

    # method 1
    M, m = scores[0], scores[0]
    for score in scores:
        if score > M:
            M = score
        if score < m:
            m = score

    print("Max value: ", M)
    print("Min value: ", m)


def e175():
    scores = [60, -20, 40, 120, 70]
    M, m = None, None
    M_idx, m_idx = 0, 0

    for score_idx in range(len(scores)):
        score = scores[score_idx]

        if M == None or score > M:
            M = score
            M_idx = score_idx

        if m == None or score < m:
            m = score
            m_idx = score_idx

    print("M/M_idx: ", M, M_idx)
    print("m/m_idx: ", m, m_idx)


def e176():
    scores = [40, 20, 30, 10, 50]
    sorted_scores = list()

    for _ in range(len(scores)):
        M, M_idx = scores[0], 0

        for score_idx in range(len(scores)):
            if scores[score_idx] > M:
                M = scores[score_idx]
                M_idx = score_idx

        tmp_scores = list()
        for score_idx in range(len(scores)):
            if score_idx == M_idx:
                sorted_scores.append(scores[score_idx])

            else:
                tmp_scores.append(scores[score_idx])

        scores = tmp_scores

        print("remaining scores: ", scores)
        print("sorted scores: ", sorted_scores, '\n')

if __name__ == '__main__':
    # e147()
    # e149()
    # e150()
    # e151()
    # e152()
    # e153()
    # e154()
    # e155()
    # e157()
    # e158()
    # e159()
    # e160()
    # e161()
    # e162()
    # e163(5000)
    # e163(200)
    # e163(45)
    # e164()
    # e165()
    # e169()
    # e170()
    # e171()
    # e172()
    # e173()
    # e174()
    # e175()
    e176()

    print("fin")




