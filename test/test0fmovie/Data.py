#!usr/bin/python
# coding:utf-8
# 获取movielens数据

import mysql.connector

config = {'host': 'localhost',
          'user': 'root',
          'password': '',
          'port': '3306',
          'database': 'movielens',
          'charset': 'utf8'}


def load_data():
    try:
        conn = mysql.connector.connect(**config)
    except mysql.connector.Error as e:
        print('connect failed!{}'.format(e))
    cursor = conn.cursor()
    try:
        # 加载用户数据
        query_users = 'select UserID, Gender, Age, Occupation, zipcode from users'
        cursor.execute(query_users)
        users = []
        for userId, gender, age, occupation, zipcode in cursor:
            user = {}
            user['userID'] = userId
            if ('F' == gender):
                user['gender'] = 0
            elif ('M' == gender):
                user['gender'] = 1
            user['age'] = age
            user['occupation'] = occupation
            user['zipCode'] = zipcode
            users.append(user)
        # 加载电影数据
        genresMap = {"Action": 0,
                     "Adventure": 1,
                     "Animation": 2,
                     "Children's": 3,
                     "Comedy": 4,
                     "Crime": 5,
                     "Documentary": 6,
                     "Drama": 7,
                     "Fantasy": 8,
                     "Film-Noir": 9,
                     "Horror": 10,
                     "Musical": 11,
                     "Mystery": 12,
                     "Romance": 13,
                     "Sci-Fi": 14,
                     "Thriller": 15,
                     "War": 16,
                     "Western": 17}
        query_movies = 'select MovieID, Title, Genres from newmovies'
        cursor.execute(query_movies)
        movies = []
        for movieId, title, genres in cursor:
            movie = {}
            movie['movieId'] = movieId
            if (title.endswith(')')):
                movie['title'] = title[0:len(title) - 7]
                movie['year'] = title[-5:-1]
            else:
                movie['title'] = title
            genreList = [0 for i in range(len(genresMap))]
            for i in genres.split("|"):
                if(genresMap.get(i) != None):
                    genreList[genresMap[i]] = 1
            movie['genres'] = genreList
            movies.append(movie)
        # 加载投票数据
        ratingAll = [list() for i in range(len(users))]
        ratingTrain = [list() for i in range(len(users))]
        ratingTest = [list() for i in range(len(users))]
        ratingLengths = [0 for i in range(len(users))]
        query_ratings = 'select UserID, MovieID, Rating, timestamp from ratings'
        cursor.execute(query_ratings)
        # ratings = []
        for userId, movieId, rating, timestamp in cursor:
            rating = {}
            ratingLengths[userId - 1] = ratingLengths[userId - 1] + 1
            rating['userId'] = userId
            rating['movieId'] = movieId
            rating['rating'] = rating
            rating['time'] = timestamp
            ratingAll[userId-1].append(rating)
        for i in range(len(ratingAll)):
           ratingAll[i].sort(key= lambda rating:rating['time'])
           ratingTrain[i] = ratingAll[i][0:int(len(ratingAll[i])*0.8)]
           ratingTest[i] = ratingAll[i][int(len(ratingAll[i])*0.8):len(ratingAll[i])]
        return users, movies, ratingAll, ratingLengths, ratingTrain, ratingTest
    except mysql.connector.Error as e:
        print('connect failed!{}'.format(e))
    finally:
        cursor.close()
        conn.close()



users, movies, ratings, ratingLengths, ratingTrain, ratingTest = load_data()
print(len(ratingTrain[0]))
print(len(ratings))
print(max(ratingLengths))
print(min(ratingLengths))
for i in range(len(ratingLengths)):
    if(ratingLengths[i] == 0 ):
        print(i)
