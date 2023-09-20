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





if __name__ == '__main__':
    e147()