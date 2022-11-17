import argparse
import json
import numpy as np


# Author: Maciej Leciejewski s21484


def build_arg_parser():
    """
    Enables to input user

    :return: argument parser
    """
    parser = argparse.ArgumentParser(description='Compute similarity score')
    parser.add_argument('--user', dest='user', required=True, help='User')

    return parser


def euclidean_score(dataset, user1, user2):
    """
    Compute the Euclidean distance score between user1 and user2

    :param: dataset: contains users, movies and ratings
    :param: user1: name of user1
    :param: user2: name of user2

    :return: the Euclidean distance score between user1 and user2
    """
    common_movies = {}
    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1

    if len(common_movies) == 0:
        return 0
    squared_diff = []
    for item in dataset[user1]:
        if item in dataset[user2]:
            squared_diff.append(np.square(dataset[user1][item] - dataset[user2][item]))

    return 1 / (1 + np.sqrt(np.sum(squared_diff)))


def pearson_score(dataset, user1, user2):
    """
    Compute the Pearson correlation score between user1 and user2

    :param: dataset: contains users, movies and ratings
    :param: user1: name of user1
    :param: user2: name of user2

    :return: the Pearson correlation score between user1 and user2
    """
    common_movies = {}
    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1

    num_ratings = len(common_movies)

    # If there are no common movies between user1 and user2, then the score is 0
    if num_ratings == 0:
        return 0

    # Calculate the sum of ratings of all the common movies
    user1_sum = np.sum([dataset[user1][item] for item in common_movies])
    user2_sum = np.sum([dataset[user2][item] for item in common_movies])

    # Calculate the sum of squares of ratings of all the common movies
    user1_squared_sum = np.sum([np.square(dataset[user1][item]) for item in common_movies])
    user2_squared_sum = np.sum([np.square(dataset[user2][item]) for item in common_movies])

    # Calculate the sum of products of the ratings of the common movies
    sum_of_products = np.sum([dataset[user1][item] * dataset[user2][item] for item in common_movies])

    # Calculate the Pearson correlation score
    xy = sum_of_products - (user1_sum * user2_sum / num_ratings)
    xx = user1_squared_sum - np.square(user1_sum) / num_ratings
    yy = user2_squared_sum - np.square(user2_sum) / num_ratings

    if xx * yy == 0:
        return 0

    return xy / np.sqrt(xx * yy)


def get_users_list(dataset):
    """
    Get users list

    :param: dataset: contains users, movies and ratings

    :return: users list
    """
    user_list = []
    for i in dataset:
        if i != user1:
            user_list.append(i)

    return user_list


def get_matching_results(data, user1, users_list):
    """
    Compute Euclidean and Pearson scores for every user

    :param: data: contains users, movies and ratings
    :param: user1: name of user1
    :param: users_list: names of all users

    :return: Euclidean and Pearson scores for every user
    """
    euclidean_score_list = {}
    pearson_score_list = {}
    for user in users_list:
        euclidean_score_list[user] = euclidean_score(data, user1, user)
        pearson_score_list[user] = pearson_score(data, user1, user)

    euclidean_score_list = sorted(euclidean_score_list.items(), key=lambda x: x[1], reverse=True)
    pearson_score_list = sorted(pearson_score_list.items(), key=lambda x: x[1], reverse=True)

    return euclidean_score_list, pearson_score_list


def print_movies(movies, user1_movies):
    """
    Print chosen movies
    """
    count = 0
    for movie in movies:
        if count < 5 and movie not in user1_movies:
            count += 1
            print(movie)


def get_recommended_movies(data, user1, matched_user):
    """
    Choose recommended movies

    :param: data: contains users, movies and ratings
    :param: user1: name of user1
    :param: matched_user: name of matched user
    """
    user1_movies = data[user1]
    matched_user_movies = sorted(data[matched_user].items(), key=lambda x: x[1], reverse=True)
    print("Recommended movies:")
    print_movies(matched_user_movies, user1_movies)


def get_not_recommended_movies(data, user1, scores_list):
    """
    Choose not recommended movies

    :param: data: contains users, movies and ratings
    :param: user1: name of user1
    :param: scores_list: computed movies scores
    """
    user1_movies = data[user1]
    not_recommended_movies = data[scores_list[2][0]]
    not_recommended_movies.update(data[scores_list[1][0]])
    not_recommended_movies.update(data[scores_list[0][0]])
    not_recommended_movies = sorted(not_recommended_movies.items(), key=lambda x: x[1])
    print("\nNot recommended movies:")
    print_movies(not_recommended_movies, user1_movies)


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    user1 = args.user
    ratings_file = 'ratings.json'

    with open(ratings_file, 'r', encoding='UTF8') as f:
        data = json.loads(f.read())

    users_list = get_users_list(data)
    euclideanScoreList, pearsonScoreList = get_matching_results(data, user1, users_list)
    print("Euclidean algorithm")
    get_recommended_movies(data, user1, euclideanScoreList[0][0])
    get_not_recommended_movies(data, user1, euclideanScoreList)
    print("\nPearson algorithm")
    get_recommended_movies(data, user1, pearsonScoreList[0][0])
    get_not_recommended_movies(data, user1, pearsonScoreList)
