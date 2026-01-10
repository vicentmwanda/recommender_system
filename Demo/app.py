import streamlit as st
from my_recommender_package.model import RecommenderModel
from my_recommender_package.data_structure import MyDataStruct
import requests
import pickle
import  io

# Load environment and API keys

#from dotenv import load_dotenv
#import os
#sys_path = os.getcwd()
#load_dotenv(os.path.join(sys_path, ".env"))
#OMDB_API_KEY = st.getenv("OMDB") 
#TMDB_API_KEY = os.getenv("TMDB") 

OMDB_API_KEY = st.secrets["OMDB"] 
TMDB_API_KEY = st.secrets["TMDB"] 


# Page config
st.set_page_config(
    page_title="Movie Recommender Demo",
    layout="wide"
)

st.title(f"🎬 Movie Recommendation System Demo \n Created By: https://github.com/vicentmwanda")


# Load model and data
def load_model_from_url(url, data_struct=None, load_temp=False):
    r = requests.get(url)
    model = pickle.load(io.BytesIO(r.content))
    model.data_struct = data_struct
    if getattr(model, "load_temp_data", False) and data_struct and load_temp:
        data_struct.idx_to_movie_id = model.tmp_idx_to_movie_id
        data_struct.movie_data = model.tmp_movie_data
    return model


MODEL_URL = "https://raw.githubusercontent.com/vicentmwanda/recommender_system/refs/heads/main/Demo/model.pkl"

database = MyDataStruct(path="", OMDB_api_key=OMDB_API_KEY, TMDB_api_key=TMDB_API_KEY)
model = load_model_from_url(MODEL_URL, database, load_temp=True)


# Poster caching with grey rectangle fallback
GREY_PLACEHOLDER_URL = "https://placehold.co/300x450"  # grey rectangle


def get_poster(movie_id):
    poster = database.generate_image(movie_id, display=False)
    print(poster)
    if not poster or poster == 0 or poster == None:
        return GREY_PLACEHOLDER_URL
    return poster

# Sidebar – User Preferences
st.sidebar.header("Your Preferences")
preference_option = st.sidebar.radio(
    "Select how you want to provide preferences:",
    ["🎬 Select by Movies", "🎭 Select by Genres"]
)
user_ratings = []

#  Rate movies
if preference_option == "🎬 Select by Movies":
    st.sidebar.markdown("### Choose movies you like")
    movie_titles = [
        (i, database.get_movie_data(i, use_idx=True)["title"])
        for i in range(len(database.idx_to_movie_id))
    ]
    movie_dict = dict(movie_titles)
    DEFAULT_MOVIES = []
    for movie_id, title in movie_dict.items():
        if "Finding Nemo" in title:
            DEFAULT_MOVIES.append(movie_id)
            break

    selected_movies = st.sidebar.multiselect(
        "Movies",
        options=list(movie_dict.keys()),
        format_func=lambda x: movie_dict[x],
        default=DEFAULT_MOVIES
    )
    st.markdown('Current Preference:')
    for mid in selected_movies:
        rating = st.sidebar.slider(
            movie_dict[mid],
            0.5, 5.0, 5.0, 0.5,
            key=f"rating_{mid}"
        )
        user_ratings.append((mid, rating))
    

# Select genres
if preference_option == "🎭 Select by Genres":
    st.sidebar.markdown("### Choose your favorite genres")
    selected_genres = st.sidebar.multiselect("Genres", model.genre_labels)
    for g in selected_genres:
        user_ratings.append((g, 5.0))

# Personalization Settings
st.sidebar.header("Personalization Settings")
personalized_n = st.sidebar.slider("Number of personalized results", 3, 20, 10)
genre_n = st.sidebar.slider("Number of movies per genre", 3, 20, 5)
epochs = st.sidebar.slider("User adaptation epochs", 10, 100, 50, 10, help ='Control number of epochs used in creating user trait and bias.')
use_updated_genre = st.sidebar.checkbox("Use trained genre embeddings", False)
item_factor = st.sidebar.slider("Item bias weight", 0.0, 1.0, 0.05, 0.05)
user_factor = st.sidebar.slider("User bias weight", 0.0, 1.0, 0.0, 0.05,
                                help="Has no effect because user bias is set to zero already!")
remove_polarized = st.sidebar.checkbox(
    "Remove polarized items",
    True,
    help="Filter out movies with highly mixed opinions"
)


# Personalized Recommendations
st.subheader("🎯 Recommended Movies")
if user_ratings:
    st.markdown("### Recommended for You")
    personalized = model.predict(
        user_ratings=user_ratings,
        number=personalized_n,
        epochs=epochs,
        genre_only=(preference_option == "🎭 Select by Genres"),
        use_updated_genre=use_updated_genre,
        item_factor=item_factor,
        user_factor=user_factor,
        remove_polarized=remove_polarized
    )
    cols = st.columns(5)
    for i, rec in enumerate(personalized):
        with cols[i % 5]:
            st.image(get_poster(rec["movie_id"]), use_container_width=True)
            st.markdown(f"**{rec['title']}**")
            st.markdown(
            f"""
            <a href="{database.get_movie_url(rec['movie_id'], database=2)}"
            target="_blank"
            style="
                display:inline-block;
                background-color:#ff7a00;
                color:white;
                padding:8px 14px;
                border-radius:6px;
                font-weight:600;
                text-decoration:none;
                text-align:center;
            ">
            View on IMDb
            </a>
            """,
            unsafe_allow_html=True
            )
            st.caption(rec["genre"])


# Genre Buttons
st.divider()
st.markdown("### 🎭 Browse Top Genres")
top_genres = ["Action", "Comedy", "Sci-Fi", "Drama", "Romance"]
if "active_genre" not in st.session_state:
    st.session_state.active_genre = top_genres[0]

cols = st.columns(len(top_genres))
for i, genre in enumerate(top_genres):
    if genre == st.session_state.active_genre:
        cols[i].markdown(
            f"""
            <div style="
                text-align:center;
                padding:10px;
                background-color:#1f77b4;
                color:white;
                border-radius:8px;
                font-weight:600;">
                {genre}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        if cols[i].button(genre):
            st.session_state.active_genre = genre


# Genre-based Recommendations
active_genre = st.session_state.active_genre
st.markdown(f"### Top {genre_n} {active_genre} Movies")
genre_recs = model.predict(
    user_ratings=[(active_genre, 5.0)],
    number=genre_n,
    epochs=epochs,
    genre_only=True,
    use_updated_genre=use_updated_genre,
    item_factor=item_factor,
    user_factor=user_factor,
    remove_polarized=remove_polarized
)
cols = st.columns(5)
for i, rec in enumerate(genre_recs):
    with cols[i % 5]:
        st.image(get_poster(rec["movie_id"]), use_container_width=True, )
        st.markdown(f"**{rec['title']}**")
        st.markdown(
            f"""
            <a href="{database.get_movie_url(rec['movie_id'], database=2)}"
            target="_blank"
            style="
                display:inline-block;
                background-color:#ff7a00;
                color:white;
                padding:8px 14px;
                border-radius:6px;
                font-weight:600;
                text-decoration:none;
                text-align:center;
            ">
            View on IMDb
            </a>
            """,
            unsafe_allow_html=True
        )

        st.caption(rec["genre"])
        st.caption(f'Rank: {int(rec["scaled_rating"])}/5')
