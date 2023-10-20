import pandas as pd
import numpy as np

def bayesian_table_test():
    table = pd.DataFrame(index=['Spam', 'Ham'])
    table['prior'] = 0.5

    # link에 대한 (spam, ham)
    table['likelihood'] = 0.6, 0.2

    # joint probability
    table['unnorm'] = table['prior'] * table['likelihood']

    # normalization constant (분모)
    norm_const = table['unnorm'].sum()

    # posterior
    table['posterior'] = table['unnorm'] / norm_const

    print(table)


def bayesian_table(table: pd.DataFrame, prior: list, likelihood: list) -> pd.DataFrame:
    # table = pd.DataFrame(index=['Spam', 'Ham'])

    # table['prior'] = 0.5
    if 'posterior' in table.columns:
        table['prior'] = table['posterior']

    else:
        table['prior'] = prior

    # link에 대한 (spam, ham)
    # table['likelihood'] = 0.6, 0.2
    table['likelihood'] = likelihood

    # joint probability
    table['unnorm'] = table['prior'] * table['likelihood']

    # normalization constant (분모)
    norm_const = table['unnorm'].sum()

    # posterior
    table['posterior'] = table['unnorm'] / norm_const

    return table


def single_bayes_test():
    table = pd.DataFrame(index=['Spam', 'Ham'])

    print("link 기준 (단일) bayes table")
    df = bayesian_table(table=table, prior=0.5, likelihood=[0.6, 0.2])
    print(df)

    print("word 기준 (단일) bayes table")
    df2 = bayesian_table(table=table, prior=0.5, likelihood=[0.4, 0.05])
    print(df2)


def stacked_bayes_test():
    table = pd.DataFrame(index=['Spam', 'Ham'])

    print("link 기준 (단일) bayes table")
    df = bayesian_table(table=table, prior=0.5, likelihood=[0.6, 0.2])
    print(df)

    # set posterior prob to new prior prob
    prior_series = df['posterior']
    print("word 추가 반영한 bayes table")
    df2 = bayesian_table(table=df, prior=prior_series, likelihood=[0.4, 0.05])
    print(df2)


def balls_in_boxes_test():
    print("="*15+"Black"+"="*15)
    likelihood_black = [0.1, 0.8]
    table = pd.DataFrame(index=['X', 'Y'])
    df = bayesian_table(table=table, prior=0.5, likelihood=likelihood_black)
    print(df)

    print("=" * 15 + "White" + "=" * 15)
    likelihood_white = [0.9, 0.2]
    table = pd.DataFrame(index=['X', 'Y'])
    df2 = bayesian_table(table=table, prior=0.5, likelihood=likelihood_white)
    print(df2)

    print("=" * 15 + "B & B" + "=" * 15)
    likelihood_black = [0.1, 0.8]
    table = pd.DataFrame(index=['X', 'Y'])
    table = bayesian_table(table=table, prior=0.5, likelihood=likelihood_black)
    print(table)
    print("------1st update_black------")
    table = bayesian_table(table=table, prior=0.5, likelihood=likelihood_black)
    print(table)
    print("------2nd update_black------")

    print("=" * 15 + "B & W" + "=" * 15)
    likelihood_black = [0.1, 0.8]
    table = pd.DataFrame(index=['X', 'Y'])
    table = bayesian_table(table=table, prior=0.5, likelihood=likelihood_black)
    print(table)
    print("------1st update_black------")
    likelihood_white = [0.9, 0.2]
    table = bayesian_table(table=table, prior=0.5, likelihood=likelihood_white)
    print(table)


def get_likelihood(df, col):
    cls, cnts = np.unique(df.iloc[:, col], return_counts=True)
    prior = cnts.astype(dtype=float)
    prior /= np.sum(prior)
    return (cls, prior)


def play_tennis_test():
    # 0. Dataset
    # df = pd.read_csv('../data/PlayTennis.csv')
    # cols = df.columns.tolist()
    # data_list = df.to_numpy().tolist()
    data = [['Sunny', 'Hot', 'High', 'Weak', 'No'], ['Sunny', 'Hot', 'High', 'Strong', 'No'], ['Overcast', 'Hot', 'High', 'Weak', 'Yes'], ['Rain', 'Mild', 'High', 'Weak', 'Yes'], ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'], ['Rain', 'Cool', 'Normal', 'Strong', 'No'], ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'], ['Sunny', 'Mild', 'High', 'Weak', 'No'], ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'], ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'], ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'], ['Overcast', 'Mild', 'High', 'Strong', 'Yes'], ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'], ['Rain', 'Mild', 'High', 'Strong', 'No']]
    cols = ['Outlook', 'Temperature', 'Humidity', 'Wind', 'Play Tennis']
    df = pd.DataFrame(data, columns=cols)

    # 1. prior
    prior_probs = [get_likelihood(df, x) for x in range(df.shape[1])]
    prior = prior_probs[-1]
    X = prior_probs[:-1]



    df.columns


    # 2. likelihood
    cls, cnts = np.unique(df.iloc[:, -1], return_counts=True)

    print('here')






if __name__ == '__main__':
    # bayesian_table_test()
    # single_bayes_test()
    # stacked_bayes_test()
    balls_in_boxes_test()
    # play_tennis_test()



