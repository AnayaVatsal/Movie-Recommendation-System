import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# loading the data from the csv file to a pandas data frame
movies_data = pd.read_csv('movies.csv')

# printing first 5 rows of the data frame
# print(movies_data.head())
# printing number of rows and columns in the data frame
# print(movies_data.shape)

# selecting relevant features for recommendation
selected_features = ['title','overview','tagline','genres','keywords']

# replacing the null values with null string
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# combining the 5 selected features
combined_features = movies_data['title']+' '+movies_data['overview']+' '+movies_data['tagline']+' '+movies_data['genres']+' '+movies_data['keywords']
# print(combined_features)

# converting the text data to feature vectors so that we can use cosine similarity effectively
# cosine similarity works more effectively on numerical values rather than text data
# basically converting them to numerical values

# creating an instance
vectorizer = TfidfVectorizer()
# storing all numerical values in feature_vectors
feature_vectors = vectorizer.fit_transform(combined_features)
# print(feature_vectors)

# getting the similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors)
# it compares one movie with all the other movies (i.e. the numerical values) in the data to check the similarity and will return the score
# print(similarity)
# print(similarity.shape)

# take input from the user
movie_name = input("Enter a movie - ")

# creating a list with all the movie names given in the dataset
list_of_all_titles = movies_data['title'].to_list()
# print(list_of_all_titles)

# finding the closest match for the movie name given by the user
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
# print(find_close_match)
# choosing the first index value from the closest matches
close_match = find_close_match[0]

# finding the index of the movie with title
movie_index = movies_data[movies_data.title == close_match]['index'].values[0]

# getting a list of similar movies
similarity_score = list(enumerate(similarity[movie_index]))

# sorting the movies based on their similarity score
# reverse is set to true so we get descending order meaning highest similarity score on the top
# x is the list of similarity score, x[1] will take the similarity score as x[0] gives the index
sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)

# print the name of similar movies based on the index
print("Recommended movies - ")

i = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index==index]['title'].values[0]
    if i<31:
        print(i,'.',title_from_index)
        i+=1