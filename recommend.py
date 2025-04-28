import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

movies = pd.read_csv('movies.csv')
print("Movies loaded:", movies.shape)
print(movies.head())

movies['genres'] = movies['genres'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(movies['genres'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def recommend(title, cosine_sim=cosine_sim):
   title = title.lower()

   all_titles = movies['title'].str.lower()
   matched_titles = all_titles[all_titles.str.contains(title)]

   if matched_titles.empty:
      return f"Movie '{title}'not found in dataset. Please check spelling."
   idx = indices[matched_titles.index[0]]
   
   sim_scores = list(enumerate(cosine_sim[idx]))
   sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
   sim_scores = sim_scores[1:6]
   movie_indices = [i[0] for i in sim_scores]
   return movies['title'].iloc[movie_indices]

   
if __name__ == "__main__":
   user_movie = input("Enter a movie name (exactly as in dataset): ")
   print(f"\nRecommendations for '{user_movie}':")
   print(recommend(user_movie))
