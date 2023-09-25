def e109():
    score1 = 10
    score2 = 20
    score3 = 30
    n_student = 3
    mean = (score1 + score2 + score3) / n_student
    print(mean)

    score1 += 10
    score2 += 10
    score3 += 10
    print(score1, score2, score3)
    mean = (score1 + score2 + score3) / n_student
    print(mean)


def e110():
    # Mean Subtraction
    # 평균 값을 구하고, 그 평균을 각 값에서 뺀 다음에 평균을 구하면 0이됨
    score1 = 10
    score2 = 20
    score3 = 30
    n_student = 3

    score_mean = (score1 + score2 + score3) / n_student

    score1 -= score_mean
    score2 -= score_mean
    score3 -= score_mean

    score_mean = (score1 + score2 + score3) / n_student
    print(score_mean)


def e111():
    # 분산
    score1 = 10
    score2 = 20
    score3 = 30
    n_student = 3

    mean = (score1 + score2 + score3) / n_student
    square_of_mean = mean**2

    # 분산 = 편차 제곱의 평균
    mean_of_square = (score1**2 + score2**2 + score3**2) / n_student
    print("square of mean: ", square_of_mean)
    print("mean of square: ", mean_of_square)


def e112():
    score1 = 10
    score2 = 20
    score3 = 30
    n_student = 3

    score_mean = (score1 + score2 + score3) / n_student
    square_of_mean = score_mean**2
    # 제곱 평균
    mean_of_square = (score1**2 + score2**2 + score3**2) / n_student
    # 분산 = 제곱 평균 - 평균 제곱
    score_variance = mean_of_square - square_of_mean
    score_std = score_variance ** 0.5

    # ** 0.5 루트인듯?
    print("mean: ", score_mean)
    print("variance: ", score_variance)
    print("standard deviation: ", score_std)

def e113():
    score1 = 10
    score2 = 20
    score3 = 30
    n_student = 3

    # 평균
    score_mean = (score1 + score2 + score3) / n_student
    square_of_mean = score_mean**2 # 평균 제곱
    mean_of_square = (score1**2 + score2**2 + score3**2) / n_student # 제곱의 평균
    # 분산 = 제곱의 평균 - 평균의 제곱, 표준편차는 분산에 루트
    score_variance = mean_of_square - square_of_mean
    score_std = score_variance**0.5
    print("mean: ", score_mean)
    print("standard deviation: ", score_std)

    # score1 =


if __name__ == '__main__':
    # e109()
    # e110()
    # e111()
    # e112()
    e113()




