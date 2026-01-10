from collections import Counter
from PIL import Image
from io import BytesIO
import math
import requests
import matplotlib.pyplot as plt
from pathlib import Path


def print_epochs(epoch,epochs,data,prints=10):
           """
            This function  prints results during training process.
           """
           every=epochs//prints
           if(every==0):
               every=1

           if((epoch+1)%every == 0 or epoch == epochs ):
                print(data)
        
def create_storage_dirs(base_path="drive/MyDrive/ml_scale"):
        """
        This function creates directories for saving the result files.
        """
        main_dirs=['models','results']
        sub_dirs={
            'results':[
                'random_split',
                'random_split/data_exploration',
                'item_split',
                'item_split/data_exploration',
                'user_split',
                'user_split/data_exploration',
            ],
            'models':[
                'random_split',
                'item_split',
                'user_split',
            ]
        }

        base_path = Path(base_path)

        #create main directories
        for d in main_dirs:
            (base_path / d).mkdir(parents=True, exist_ok=True)

        #create sub-directories
        for parent, paths in sub_dirs.items():
            for p in paths:
                (base_path / parent / p).mkdir(parents=True, exist_ok=True)

def print_selection(selection, by_dummy_user = False):
    """
    This function prints the details of the movies from the prediction selection.
    """
    for i, movie in enumerate(selection, 1):
        print(f"=== {i}. {movie['title']}  ===")
        print(f"Movie ID: {movie['movie_id']}")
        if(by_dummy_user == False):
               print(f"Model Score: {movie['model_rating']:.4f}") 
               print(f"Rank: {movie['scaled_rating']:.2f}")
        else:
         
               print(f"User Rank: {movie['rating']:.4f}")
        print(f"Genres: {movie['genre']}")
        print(f"Url: {movie['url']}")
        print()
        
def print_selection_summary(selection):
    """
    This function prints a summary of the movies from the prediction selection
    """
    print(f"SUMMARY OF RECOMMENDATIONS")
    print(f"RANKING: TITLE  - RATING")
    for i, movie in enumerate(selection, 1):
        print(f"{i}: {movie['title']} - {movie['model_rating']:.2f}")

def plot_genres(genres,top, search = '', data_title='',split_strategy='',title="pred_plot"):
    "This function plots histogram of genres in predictions"
    genre_strings = genres

    DELIM = "|"

    # split and flatten
    all_genres = []
    for s in genre_strings:
        if not s:
            continue
        parts = [g.strip() for g in s.split(DELIM) if g.strip()]
        all_genres.extend(parts)

    # count occurrences
    counts = Counter(all_genres)

    # prepare for plotting
    labels = list(counts.keys())
    values = [counts[l] for l in labels]

    # bar chart
    plt.figure(figsize=(8,4.5))
    plt.bar(labels, values)
    plt.xlabel("Genre" , fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.title(f"Top {top} Genre counts {search}", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f'drive/MyDrive/ml_scale/results/{split_strategy}_split/{title}_{split_strategy}_split_{data_title}_dataset.pdf',format='pdf',dpi=300)
    plt.show()

def generate_movie_grid(data_struct,movie_dict, cols=4,search_title='',data_title='',split_strategy='',title=''):
    """
    Display movie posters from a dictionary of moves/movie IDs in a grid.

    """
    images = []

    # Load images
    for data in movie_dict:
        movie_id=data['movie_id']
        poster_url = data_struct.get_movie_poster(movie_id, use_idx=False)
      
        if poster_url and poster_url != 'N/A':
            try:
                response = requests.get(poster_url)
                img = Image.open(BytesIO(response.content))
           
                images.append(img)
            except Exception as e:
                print(f"Error loading poster for movie {movie_id}: {e}")
                images.append(None)
        else:
            images.append(None)

    # Calculate grid size
    n_images = len(images)
    rows = math.ceil(n_images / cols)

    # Plot grid
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 4))
    fig.suptitle(search_title, fontsize=12)
    axes = axes.flatten()  

    for i in range(len(axes)):
        axes[i].axis('off')  # turn off axes
        if i < n_images and images[i] is not None:
            axes[i].imshow(images[i])
        else:
            axes[i].imshow(Image.new('RGB', (200, 300), color='gray'))  # placeholder

    plt.tight_layout()
    plt.savefig(f'drive/MyDrive/ml_scale/results/{split_strategy}_split/predict_layout_{title}_{split_strategy}_split_{data_title}_dataset.pdf',format='pdf',dpi=300)
    plt.show()


def print_genre_choices(data):
     "This displays the predictions for genres."
     print("GENRE: RATING")
     for (genre,rating) in data:
           print(f'{genre}: {rating}')

def print_selection_summary_latex(selection):
    """
    This functions print the movie prediction as latex table.
    """
    print(r"\begin{table}[h]")
    print(r"\label{sample-table}")
    print(r"\vskip 0.15in")
    print(r"\begin{center}")
    print(r"\begin{small}")
    print(r"\begin{sc}")
    print(r"\begin{tabular}{llll}")
    print(r"\toprule")
    print(r"Ranking & Title & Model Score & Scaled Rank   \\")
    print(r"\midrule")
    for i, movie in enumerate(selection, 1):
       
        title = movie['title'].replace('_', r'\_')
        print(f"{i} & {title} & {movie['model_rating']:.2f} & {movie['scaled_rating']:.2f}  \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{sc}")
    print(r"\end{small}")
    print(r"\end{center}")
    print(r"\vskip -0.1in")
    print(r"\end{table}")


def print_selection_summary_latex_genre(selection):
    """
    This functions print the movie prediction based on genre as latex table.
    """
    #start table
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Summary of Recommended Movies}")
    print(r"\label{tab:recommendations}")
    print(r"\vskip 0.15in")

    #use small font and scaled text
    print(r"\begin{small}")
    print(r"\begin{sc}")

    #define tabular with wrapping for Title and Genre
    print(r"\begin{tabular}{l>{\raggedright\arraybackslash}p{6cm}>{\raggedright\arraybackslash}p{4cm}cc}")
    print(r"\toprule")
    print(r"Ranking & Title & Genre & Model Score & Scaled Rank \\")
    print(r"\midrule")

    #populate table
    for i, movie in enumerate(selection, 1):
      
        title = movie['title'].replace('_', r'\_')
        genre_raw = movie.get('genre', '')
        #replacing non-breaking characters 
        genre = genre_raw.replace('_', r'\_').replace('|', r'\slash ')
        rating = f"{movie['model_rating']:.2f}"
        print(f"{i} & {title} & {genre} & {rating} & {movie['scaled_rating']:.2f} \\\\")

    #end table
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{sc}")
    print(r"\end{small}")
    print(r"\vskip -0.1in")
    print(r"\end{table}")

