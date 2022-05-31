#Hybrid Recommender System

########################################################
#İŞ PROBLEMİ
########################################################
#ID'si verilen kullanıcı için item-based ve user-based recommender yöntemlerini kullanarak 10 film önerisi yapınız

############################
#VERİ SETİ HİKAYESİ
############################
#Veri seti, bir film tavsiye hizmeti olan MovieLens tarafından sağlanmıştır.
#İçerisinde filmler ile birlikte bu filmlere yapılan derecelendirme puanlarını barındırmaktadır.
#27.278 filmde 2.000.0263 derecelendirme içermektedir.
#Bu veri seti ise 17 Ekim 2016 tarihinde oluşturulmuştur.
#138.493 kullanıcı ve 09 Ocak 1995 ile 31 Mart 2015 tarihleri arasında verileri içermektedir.
# Kullanıcılar rastgele seçilmiştir.
#Seçilen tüm kullanıcıların en az 20 filme oy verdiği bilgisi mevcuttur

############################
#movie.csv

#movieId: Eşsiz film numarası.
#title: Film adı
#genres: Tür

############################
#rating.csv

#userid: Eşsiz kullanıcı numarası. (UniqueID)
#movieId: Eşsiz film numarası. (UniqueID)
#rating: Kullanıcı tarafından filme verilen puan
#timestamp: Değerlendirme tarihİ


## User Based Recommendation

########################################################
#VERİYİ HAZIRLAMA
########################################################
import pandas as pd
pd.pandas.set_option("display.max_columns", None)
pd.pandas.set_option("display.width", 100)

#movieId, film adı ve filmin tür bilgisini içeren veri seti
movie = pd.read_csv("tavsiye sistemleri/ödev 2/movie.csv")
movie.head()
movie.shape

#userID, film adı, filme verilen oy ve zaman bilgisini içeren veri seti
rating = pd.read_csv("tavsiye sistemleri/ödev 2/rating.csv")
rating.head()
rating.shape
rating["userId"].nunique()

#rating veri setine Id’lere ait film isimlerini ve türünü movie veri setinden ekledim.
df = movie.merge(rating, how="left", on="movieId")

#Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini listede tuttum ve veri setinden çıkarttım.
comment_counts = pd.DataFrame(df["title"].value_counts())
rare_movies = comment_counts[comment_counts["title"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]

#index'te userID'lerin sutunlarda film isimlerinin ve değer olarak ratinglerin bulunduğu dataframe için pivot table oluşturdum.
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
user_movie_df.head()


#Yapılan tüm işlemleri fonksiyonlaştıdım.
def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv("pythonProject/tavsiye sistemleri/ödev 2/movie.csv")
    rating = pd.read_csv("pythonProject/tavsiye sistemleri/ödev 2/rating.csv")
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return  user_movie_df

user_movie_df = create_user_movie_df()

########################################################
#Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
########################################################

random_user = 108170

random_user_df = user_movie_df[user_movie_df.index == random_user]


random_user_df.isnull().sum()
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()


########################################################
#Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişilmesi
#######################################################
user_movie_df = user_movie_df[movies_watched]
movie_watched_df.head(2)

user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]

perc = len(movies_watched) * 60 / 100

users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]


########################################################
#Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
########################################################
final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]
final_df.head()
final_df.shape

corr_df =final_df.T.corr().unstack().sort_values()
corr_df = pd.DataFrame(corr_df, columns=[corr])
corr_df.index.names = ["userId_1", "userId_2"]
corr_df = corr_df.reset_index()

corr_df[corr_df["userId_1"] == random_user]

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)
top_users_rating = top_users.merge(rating[["userId", "movieId", "rating"]], how="inner")


########################################################
#Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulmas
########################################################
top_users_rating["userId"].nunique()
top_users_rating["weighted_rating"] =top_users_rating["corr"] * top_users_rating["rating"]

recommendation_df[recommendation_df["weighted_rating"] > 3.5]
movies_to_be_reccomend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)
movies_to_be_reccomend.merge(movie[["movieId","title"]])["title"][0:5]


## Item Based Recommendation (Kullanıcının izlediği en son ve en yüksek puan verdiği filme göre item-based öneri)

import pandas as pd
pd.pandas.set_option("display.max_columns", None)
pd.pandas.set_option("display.width", 100)

#movieId, film adı ve filmin tür bilgisini içeren veri seti
movie = pd.read_csv("tavsiye sistemleri/ödev 2/movie.csv")


#userID, film adı, filme verilen oy ve zaman bilgisini içeren veri seti
rating = pd.read_csv("tavsiye sistemleri/ödev 2/rating.csv")


movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]

movie_df = user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]

user_movie_df.corrwith(movie_df).sort_values(ascending=False).head(10)
