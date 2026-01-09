
from my_recommender_package.ALS_operations import SEED
from datetime import datetime
import os
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import gc

#setting numpy seed
np.random.seed(SEED)
#from google.colab import drive, userdata
def get_file_data(path,name):
    '''
    This function opens a file for reading data.
    Parameters:
         name(str): file name
    Ouputs:
          file object: opened file.
    '''


    file = open(os.path.join(path, name), 'r')
    return file

def read_ratings(path,file_name):
    '''
    This function reads user ratings from csv file.
    Parameters:
         name(str): file name
    Yields:
        tuple: (user_id (int), movie_id (int), rating (float))
    '''
    print('started reading data!')
    with get_file_data(path,file_name) as f:
        next(f)  # skip header
        for line in f:
            user_id, movie_id, rating, timestamp = line.strip().split(',')
            yield int(user_id), int(movie_id), float(rating)
    print('reading data completed!')


def read_genres(path,file_name):
    """
    This function reads movie genres and titles from a csv file.

    Parameters:
        file_name (str): Name of the movies/genres file.

    Yields:
        tuple: (movie_id (int), title (str), genres (str))
    """
    print('started reading data!')
    with get_file_data(path,file_name) as f:
        next(f)  # skip header
        for line in f:

            data= line.strip().split(',')

            movie_id = int(data[0])
            genres = str(data[-1])

            # to include titles with commas
            title = ','.join(data[1:-1])
            yield int(movie_id), str(title), str(genres)
    print('reading data completed!')
    
def read_links(path,file_name):
    """
    This function reads movie link information (IMDb and TMDB IDs)
    from a csv file.

    Parameters:
        file_name (str): Name of the links file.

    Yields:
        tuple: (movie_id (int), imdb_id (str), tmdb_id (str))
    """
    print('started reading data!')
    with get_file_data(path,file_name) as f:
        next(f)  # skip header
        for line in f:
            movie_id, imdbId, tmbdId = line.strip().split(',')
            yield int(movie_id), imdbId, tmbdId
    print('reading data completed!')





class MyDataStruct():
     """
     This class stores and manages movie ratings data, including training and test splits,
     mapping between user/movie IDs and internal indices, and optimized flattened structures.
     """
     def __init__(self,path,train_split=0.2, split_strategy='random' ,OMDB_api_key='', TMDB_api_key = ''):
        """
        Initializing data structure.

        Parameters:
        -----------
        path : str for data file directory

        train_split : float for train test split
  
        split_strategy : str with values; 'random', 'user' or 'item'

        api_key: str for the API Key for loading movie images from IMDB
        """
        #setting train_split strategy 
        #split_stategy = 'data' for considering all data at once        
        #split_stategy = 'user' for considering data at user level
        #split_stategy = 'item' for considering data at item level

        self.split_strategy=split_strategy
        #movie data api key for getting images
        self.OMDB_api_key = OMDB_api_key
        self.TMDB_api_key = TMDB_api_key
        #data file
        self.path=path
        # train test split  ratio of data to use for testing
        self.train_split_ratio = train_split

        # lists to map indices back to IDs
        self.idx_to_movie_id =[]
        self.idx_to_user_id = []



        # dictionaries to map IDs to indices
        self.movie_id_to_idx = {}
        self.user_id_to_idx = {}

        self.feature_to_idx={} #idxs for the self.genre_labels array



        # dictionaries to store start indices of flattened test/train data
        self.movie_id_to_data_idx_test = {}
        self.user_id_to_data_idx_test = {}
        self.movie_id_to_data_idx_train = {}
        self.user_id_to_data_idx_train = {}

        self.movie_idx_to_feature_data_idx = {}
        self.feature_idx_to_movie_data_idx = {}

        # lists of lists to store test data as its read from file
        self.data_by_movie_test = []
        self.data_by_user_test = []

        # flattened list to store test data for quick access
        self.rating_data_by_user_test = [] # ratings data by user
        self.rating_data_by_movie_test = [] #rating data by movie
        self.id_data_by_user_test = [] # idx data by user
        self.id_data_by_movie_test = [] # idx data by movie


        # lists to temporary store  data as its read from file
        self.data_by_movie_train = []
        self.data_by_user_train = []

        # flattened list to store train data for quick access
        self.rating_data_by_user_train = [] # ratings data by user
        self.rating_data_by_movie_train = [] # ratings data by movie
        self.id_data_by_user_train = [] # idx data by user
        self.id_data_by_movie_train = [] # idx data by movie

        #flattened list to store feature data for quick access
        self.movie_idx_data_by_feature=[]  # movie idxs  data  by feature
        self.feature_idx_data_by_movie=[]  # feature idxs data by movie

        #variables to store statistics about the dataset
        self.counts = 0  # total entries in the rating dataset
        self.train_count = 0 # total entries in train rating data
        self.test_count = 0  # total entries in test rating data

        #rating bins for plotting rating statistics
        self.rating_bins = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        self.rating_counts = {r: 0 for r in self.rating_bins}
        #dictionary to store movie meta data (genre, title and IMDB IDs)
        self.movie_data={}
        #list of all genres in the data
        self.genre_labels = []

     def get_or_create_user_idx(self,user_id):
          """
           This function returns the internal index for an existing  user_id or  creates a new user entry if necessary.
           Also initializes data structures for training and test data.
           Input:
                user_id(int): ID for user
           Output:
                 int: internal index for accessing the rating data by user
           """
          idx = -1
          if user_id in self.user_id_to_idx:
              # get user idx if user exists
              idx = self.user_id_to_idx[user_id]
          else:
            # creating new index for user data
            idx=len(self.idx_to_user_id)
            self.idx_to_user_id.append(user_id)
            self.user_id_to_idx[user_id] = idx


            #data structures for train test split
            self.data_by_user_train.append([])
            self.data_by_user_test.append([])
          return idx
     def get_or_create_movie_idx(self,movie_id):
          """
           This function returns the internal index for an existing  movie_id or  creates a new movie entry if necessary.
           Also initializes data structures for training and test data.
           Input:
                movie_id(int): ID for movie
           Output:
                 int: internal index for accessing rating data by movie
           """
          idx = -1
          if movie_id in self.movie_id_to_idx:
              #get movie  idx if movie exists
              idx=self.movie_id_to_idx[movie_id]
          else:
            # creating new index for movie data
            idx = len(self.idx_to_movie_id)
            self.idx_to_movie_id.append(movie_id)
            self.movie_id_to_idx[movie_id] = idx


            # data structures for train test split
            self.data_by_movie_test.append([])
            self.data_by_movie_train.append([])
          return idx
     def clean_train_test_split(self):
            """
            This function ensures  explain movie and user in the train data has a rating
            after split.
            """
            print('Cleaning train test split')
            #starting with user data
            for user_idx in range(len(self.data_by_user_train)):
                if len(self.data_by_user_train[user_idx]) == 0:

                    if len(self.data_by_user_test[user_idx]) == 0:
                        continue  #skip if test also has no rating

                    movie_idx, rating = self.data_by_user_test[user_idx].pop()

                    self.data_by_user_train[user_idx].append((movie_idx, rating))

                    #removing rating data from movie test data also
                    for index, (user_idx_test, r) in enumerate(self.data_by_movie_test[movie_idx]):
                        if user_idx_test == user_idx and r == rating:
                            self.data_by_movie_test[movie_idx].pop(index)
                            break
                    self.data_by_movie_train[movie_idx].append((user_idx, rating))

            #also checking movie data
            for movie_idx in range(len(self.data_by_movie_train)):
                if len(self.data_by_movie_train[movie_idx ]) == 0:
                    if len(self.data_by_movie_test[movie_idx]) == 0:
                        continue #skip if test also has no rating

                    user_idx, rating = self.data_by_movie_test[movie_idx].pop()

                    self.data_by_movie_train[movie_idx].append((user_idx, rating))

                    #removing rating from user test data
                    for index, (movie_idx_test, r) in enumerate(self.data_by_user_test[user_idx]):
                        if movie_idx_test == movie_idx and r == rating:
                            self.data_by_user_test[user_idx].pop(index)
                            break

                    self.data_by_user_train[user_idx].append((movie_idx, rating))
     def second_train_split(self):
            '''
            This function does train test split at user or item level
            (optimized, minimal changes)
            '''
            train_set = self.data_by_user_train
            target_set = self.data_by_user_test
            target_set2 = self.data_by_movie_test

            second_set=self.data_by_movie_train
            

            if self.split_strategy == 'item':
                train_set = self.data_by_movie_train
                target_set = self.data_by_movie_test
                target_set2 = self.data_by_user_test

                second_set=self.data_by_user_train
                

            ratio = self.train_split_ratio

            # implementing train split
            for idx, entries in enumerate(train_set):
                num = len(entries)
                if num <= 1:
                    continue  # ensuring train set has at least one value

                test_num = int(ratio * num)
                if test_num == 0:
                    continue

                
                perm = np.random.permutation(num)
                test_idx = perm[:test_num]
                train_idx = perm[test_num:]

                # split entries
                test_entries = [entries[i] for i in test_idx]
                train_entries = [entries[i] for i in train_idx]

                train_set[idx] = train_entries
                target_set[idx] = test_entries

                #updating the complementary data sets
                for other_idx, rating in test_entries:
                    target_set2[other_idx].append((idx, rating))
                
                for other_idx, rating in entries:
                    second_set[other_idx].append((idx, rating))

                # updating train/test split counts
                self.train_count -= test_num
                self.test_count += test_num
                
                
     def load_data(self,file_name):
         """
         This functions loads ratings data from a file, and  splits it into training and test sets,
         and call the optimization function to create optimized data structures.
         Input:
             file_name(str): file name
         Output:
              None
         """
         print('started loading data')

         total = self.counts
         ratio = self.train_split_ratio
         train_ratio = 1 - ratio

         time_stamp=datetime.now().timestamp()
         for user_id, movie_id, rating in read_ratings(self.path,file_name):

              total = self.counts
              test_count = self.test_count
              train_count = self.train_count
               # Get internal indices for user and movie
              user_idx=self.get_or_create_user_idx(user_id)
              movie_idx=self.get_or_create_movie_idx(movie_id)
              
              if(self.split_strategy == 'random'):
                  # Randomly decide if this rating goes to test set
                  choice=choice = np.random.rand() < ratio

                  desired_test_count = ratio * total
                  desired_train_count = train_ratio * total

                  if (test_count >= desired_test_count) and (choice == True):
                        # if too many test values ,add to train then
                        choice = False

                  if (train_count >= desired_train_count) and (choice == False):
                          # if too many train values
                          choice = True
              else:
                   #if splitting at item or user level, then save all data as train
                   choice = False
              # Add rating to appropriate split
              self.add_train_split_data(user_idx,movie_idx,rating,choice)
              # Update the statistics  counters
              self.rating_counts[rating] += 1
              self.counts += 1
         if(self.split_strategy != 'random'):
             self.second_train_split()
         self.clean_train_test_split()

         time_stamp=datetime.now().timestamp()-time_stamp

         print('data loading completed!')

         #optimization
         self.optimize_data_struct()

         #train split results
         final_split=self.get_final_train_split()
         strategy={
             'random': 'Random',
             'user': 'By user',
             'item': 'By movie',
         }

         print(f"""
######### TRAIN SPLIT RESULTS (TARGET RATIO: {(self.train_split_ratio*100):.02f}%) ############
Strategy: {strategy[self.split_strategy]}

Train count:{self.train_count}
Test count:{self.test_count}

Total count:{self.train_count+self.test_count}

Final Train/Test split: {final_split:.02f} %

Data loading & Train Split Duration: {time_stamp:.3f} seconds
###############################################################################################
         """
         )
     def get_final_train_split(self):
          if(self.train_count == 0 or self.test_count == 0 ):
                return 0
          return self.test_count*100/(self.train_count+self.test_count)
     def optimize_data_struct(self):
         """
          This function creates a flatten training and test data structure for efficient access.
          Converts lists of lists into continuous arrays and records start indices for each user/movie.
          At the start index in the optimized data structure, size of the entries for the user/movie is entered.
         """
         print('###data structure optimization started')

         print('optimizing training data')

         #Flatten  training data by movie
         data_list = self.data_by_movie_train
         rating_list = self.rating_data_by_movie_train
         id_list = self.id_data_by_movie_train
         for n in range(len(data_list)):
                  movie_idx = n
                  movie_id = self.idx_to_movie_id[movie_idx]
                  data = data_list[n]
                  num = len(data)
                  data_idx = len(rating_list)

                  # First element stores number of ratings
                  rating_list.append(num)
                  id_list.append(num)
                  # Append all user indices and ratings
                  for (user_idx,rating) in data:
                       rating_list.append(rating)
                       id_list.append(user_idx)
                  # Store start index for this movie
                  self.movie_id_to_data_idx_train[movie_id] = data_idx

         # Flatten training data by user
         data_list = self.data_by_user_train
         rating_list = self.rating_data_by_user_train
         id_list=self.id_data_by_user_train
         for m in range(len(data_list)):
                  user_idx = m
                  user_id = self.idx_to_user_id[user_idx]
                  data = data_list[m]
                  num = len(data)
                  data_idx = len(rating_list)
                  # First element stores number of ratings
                  rating_list.append(num)
                  id_list.append(num)
                  # Append all user indices and ratings
                  for (movie_idx,rating) in data:
                       rating_list.append(rating)
                       id_list.append(movie_idx)
                  # Store start index for this movie
                  self.user_id_to_data_idx_train[user_id] = data_idx
         print('optimizing test data')
        # Flatten test data by movie
         data_list = self.data_by_movie_test
         rating_list = self.rating_data_by_movie_test
         id_list = self.id_data_by_movie_test
         for n in range(len(data_list)):
                  movie_idx = n
                  movie_id = self.idx_to_movie_id[movie_idx]
                  data = data_list[n]
                  num = len(data)
                  data_idx = len(rating_list)
                  # First element stores number of ratings
                  rating_list.append(num)
                  id_list.append(num)
                  # Append all user indices and ratings
                  for (user_idx,rating) in data:
                       rating_list.append(rating)
                       id_list.append(user_idx)
                  # Store start index for this movie
                  self.movie_id_to_data_idx_test[movie_id ]= data_idx

         # Flatten test data by user
         data_list = self.data_by_user_test
         rating_list = self.rating_data_by_user_test
         id_list=self.id_data_by_user_test
         for m in range(len(data_list)):
                  user_idx = m
                  user_id = self.idx_to_user_id[user_idx]
                  data = data_list[m]
                  num = len(data)
                  data_idx = len(rating_list)
                  # First element stores number of ratings
                  rating_list.append(num)
                  id_list.append(num)
                  # Append all user indices and ratings
                  for (movie_idx,rating) in data:
                       rating_list.append(rating)
                       id_list.append(movie_idx)
                  # Store start index for this movie
                  self.user_id_to_data_idx_test[user_id] = data_idx
         print('###data structure optimization started!')
         print('###train split completed!')
     def optimize_feature_data(self):
              feature_data=[[] for i in range(len(self.genre_labels))]

              #updating flatten array for feature data by movie
              movie_feature_list=self.feature_idx_data_by_movie
              for  movie_id in self.movie_data:
                  #checking if the id is part the general movie data and also the training data
                  if movie_id in self.movie_id_to_idx and movie_id in  self.movie_id_to_data_idx_train:
                      movie_idx=self.movie_id_to_idx[movie_id]
                      data=self.movie_data[movie_id]

                      genre_idxs=data['genre_idxs']
                      #number of features for this movie
                      num = len(genre_idxs)

                      data_idx = len(movie_feature_list)
                      # First element stores number of ratings
                      movie_feature_list.append(num)
                      # Append all user indices and ratings
                      for idx in genre_idxs:
                          movie_feature_list.append(idx)
                          #add the movie idx to feature temporary array
                          feature_data[idx].append(movie_idx)

                      # Store start data index for this movie's feature data
                      self.movie_idx_to_feature_data_idx[movie_idx]= data_idx

              #updating flatten array for movie data by feature
              feature_movie_list=self.movie_idx_data_by_feature
              for  feature_idx, data in enumerate(feature_data):
                      #number of movies with that feature
                      num = len(data)

                      data_idx = len(feature_movie_list)
                      # First element stores number of ratings
                      feature_movie_list.append(num)
                      # Append all user indices and ratings
                      for idx in data:
                          feature_movie_list.append(idx)

                      # Store start data index for this movie's feature data
                      self.feature_idx_to_movie_data_idx[feature_idx]= data_idx

              del feature_data
              #freeing the  unoptimized structures
              self.data_by_movie_train = None
              self.data_by_user_train = None
              self.data_by_movie_test = None
              self.data_by_user_test = None

              gc.collect()

     def load_item_data(self,genre_file,link_file,get_links=True,get_genre=True):
         """
          This function loads movie metadata (genres, titles, IMDB/TMDB links) and store it in self.movie_data.

          Inputs:
          genre_file(str): Path to the genres file.
          link_file (str): Path to the links file with IMDB IDs.
          get_links (bool): Boolean to check whether to load link data.
          get_genre (bool): Boolean to check whether to load genre/title data.

          Output:
             None
          """

         print('started loading genre/feature data')
         if(get_genre == True):
            # Load genres and titles from file
            for movie_id, title, genres in read_genres(self.path,genre_file):
                          # Store title + genre for this movie
                          self.movie_data[movie_id]={
                              'title':title,
                              'genre':genres,
                              'genre_idxs':[]
                          }
                          # Extract unique genre labels
                          genre_labels=genres.split('|')
                          for genre in genre_labels:
                               if genre not in self.genre_labels:
                                   idx=len(self.genre_labels)
                                   self.genre_labels.append(genre)
                                   self.feature_to_idx[genre]=idx
                                   self.movie_data[movie_id]['genre_idxs'].append(idx)
                               else:
                                   idx=self.feature_to_idx[genre]
                                   self.movie_data[movie_id]['genre_idxs'].append(idx)


         print('data loading completed!')

         print('started loading link data')
         if(get_links==True):
             # Load IMDB/TMDB IDs for creating poster urls/links
            for movie_id, imdbId,tmbdId in read_links(self.path,link_file):
                      if movie_id in self.movie_data:
                          self.movie_data[movie_id]['imdbId']=imdbId
                          self.movie_data[movie_id]['tmbdId']=tmbdId
                      else:
                          self.movie_data[movie_id]={
                              'imdbId':imdbId,
                              'tmbdId':tmbdId
                          }
         print('data loading completed!')

         print('optimizing features data')
         self.optimize_feature_data()
         print('optimizing completed')


     

     def add_train_split_data(self,user_idx,movie_idx,rating,to_test=False):
         """
          This function stores a user and movie rating either in the training set
          or the test set depending on the 'to_test' boolean check.

          Inputs:
              user_idx (int): Internal user index.
              movie_idx (int): Internal movie index.
              rating (float or int): Rating value.
              to_test (bool): Boolean to check whether the data should be added to the test split.
        """

         if(to_test == True):
            # Test data
            self.data_by_movie_test[movie_idx].append((user_idx,rating))
            self.data_by_user_test[user_idx].append((movie_idx,rating))

            self.test_count+=1

         else:
            # Training data
            if(self.split_strategy == 'random' or self.split_strategy == 'item'):
                self.data_by_movie_train[movie_idx].append((user_idx,rating))
            if(self.split_strategy == 'random' or self.split_strategy == 'user'):
                self.data_by_user_train[user_idx].append((movie_idx,rating))

            self.train_count+=1

     def get_data_by_movies(self):
            """
            This function returns a list of combined train+test rating data for each movie.

            Outputs:
                list: A list where each element contains all (user_idx, rating)
                      pairs for the corresponding movie.
            """
            data=[]
            for idx in range(len(self.idx_to_movie_id)):
                  data_train=self.get_ratings_vector_by_movie(idx,True)
                  data_test=self.get_ratings_vector_by_movie(idx,True,is_train=False)
                  data.append(data_train+data_test)
            return data
     def get_data_by_users(self):
            """
              This function returns a list of combined train+test rating data for each user.

              Outputs:
                  list: A list where each element contains all (movie_idx, rating)
                        pairs for the corresponding user.
            """
            data=[]
            for idx in range(len(self.idx_to_user_id)):

                  data_train=self.get_ratings_vector_by_user(idx,True)
                  data_test=self.get_ratings_vector_by_user(idx,True,is_train=False)
                  data.append(data_train+data_test)
            return data

     def get_data_by_movie_id(self,movie_id,use_idx=False):
             """
              This function returns all rating data (train + test) for a given movie id.

              Inputs:
                  movie_id (int): External movie ID unless use_idx=True.
                  use_idx (bool): If True, 'movie_id' is treated as an internal movie index.

              Outputs:
                  list: A combined list of ratings  for the movie.
             """

             if movie_id not in self.movie_id_to_idx and use_idx!=True:
                  print('Movie ID not found!')
                  return None
             if(use_idx==False):
                   movie_idx=self.movie_id_to_idx[movie_id]
             else:
                   movie_idx=movie_id
             data_train=self.get_ratings_vector_by_movie(movie_idx,True)
             data_test=self.get_ratings_vector_by_movie(movie_idx,True,is_train=False)
             return (data_train+data_test)

     def get_data_by_user_id(self,user_id,use_idx=False):
             """
              This function returns all rating data (train + test) for a given user id.

              Inputs:
                  user_id (int): External user ID unless use_idx=True.
                  use_idx (bool): If True, 'user_id' is treated as an internal user index.

              Outputs:
                  list: A combined list of ratings for the user.
             """
             if user_id not in self.user_id_to_idx and use_idx!=True:
                  print('User ID not found!')
                  return None
             if(use_idx==False):
                 user_idx=self.user_id_to_idx[user_id]
               
             else:
                  user_idx=user_id


             data_train=self.get_ratings_vector_by_user(user_idx,True)
             data_test=self.get_ratings_vector_by_user(user_idx,True,is_train=False)
             return (data_train+data_test)
     def get_feature_data_by_movie_idx(self,movie_idx):
             """
              This function returns all feature idxs for a given movie idx.

              Inputs:
                  movie_idx (int): idx for movie


              Outputs:
                  list: A list of feature data for the movie.
             """
             data_idx=self.movie_idx_to_feature_data_idx[movie_idx]
             data_list=self.feature_idx_data_by_movie

             num=data_list[data_idx]

             #start and end of the movie's features in the flat array
             start=int(data_idx+1)
             end=int(data_idx+num+1)

             return data_list[start:end]

     def get_movie_data_by_feature_idx(self,feature_idx):
             """
              This function returns all movie idxs for a given feature idx.

              Inputs:
                  movie_idx (int): idx for feature


              Outputs:
                  list: A list of movie idxs for the feature.
             """
             data_idx=self.feature_idx_to_movie_data_idx[feature_idx]
             data_list=self.movie_idx_data_by_feature

             num=data_list[data_idx]

             #start and end of the feature's movies in the flat array
             start=int(data_idx+1)
             end=int(data_idx+num+1)

             return data_list[start:end]
     def get_data_by_movie_train(self,movie_id,use_idx=False):
             """
              This function returns all training ratings for a given movie.

              Inputs:
                  movie_id (int): External movie ID unless use_idx=True.
                  use_idx (bool): If True, 'movie_id' is treated as an internal movie index.

              Outputs:
                  list: A list of ratings from the training set.
             """
             if movie_id not in self.movie_id_to_idx and use_idx!=True:
                  print('Movie ID not found!')
                  return None
           
             return self.get_ratings_vector_by_movie(movie_id,use_idx)
     def get_data_by_movie_test(self,movie_id,use_idx=False):
             """
              This function returns all test ratings for a given movie.

              Inputs:
                  movie_id (int): External movie ID unless use_idx=True.
                  use_idx (bool): If True, 'movie_id' is treated as an internal movie index.

              Outputs:
                  list: A list of ratings  pairs from the test set.
              """

             if movie_id not in self.movie_id_to_idx and use_idx!=True:
                  print('Movie ID not found!')
                  return None
             return self.get_ratings_vector_by_movie(movie_id,use_idx,is_train=False)
     def get_data_by_user_train(self,user_id,use_idx=False):
             """
              This function returns all training rating data for a given user.

              Inputs:
                  user_id (int): External user ID unless use_idx=True.
                  use_idx (bool): If True, 'user_id' is treated as an internal user index.

              Outputs:
                  list: A list of (movie_idx, rating) pairs from the training set.
              """

             if user_id not in self.user_id_to_idx and use_idx!=True:
                  print('User ID not found!')
                  return None
             return self.get_ratings_vector_by_user(user_id,use_idx)
     def get_data_by_user_test(self,user_id,use_idx=False):
             """
            This function returns all test rating data for a given user.

            Inputs:
                user_id (int): External user ID unless use_idx=True.
                use_idx (bool): If True, 'user_id' is treated as an internal user index.

            Outputs:
                list: A list of (movie_idx, rating) pairs from the test set.
            """

             if user_id not in self.user_id_to_idx and use_idx!=True:
                  print('Movie ID not found!')
                  return None
             
             return self.get_ratings_vector_by_user(user_id,use_idx,is_train=True)

     def get_ratings_vector_by_movie(self,movie_id,use_idx=False,is_train=True):
                 """
                This function returns a vector of rating values for a given movie.

                Inputs:
                    movie_id (int): External movie ID unless use_idx=True.

                    use_idx (bool): If True, 'movie_id' is treated as
                    an internal movie index.

                    is_train (bool): If True, return training ratings;
                    otherwise return test ratings.

                Outputs:
                    numpy array: Array of rating values for the movie.
                 """
                 if(use_idx==True):
                      movie_id=self.idx_to_movie_id[movie_id]

                 data_idx=self.movie_id_to_data_idx_train[movie_id]
                 data_list=self.rating_data_by_movie_train

                 if(is_train==False):
                     data_idx=self.movie_id_to_data_idx_test[movie_id]
                     data_list=self.rating_data_by_movie_test

                 num=data_list[data_idx]
                 start=int(data_idx+1)
                 end=int(data_idx+num+1)
                 return data_list[start:end]
     def get_ratings_vector_by_user(self,user_id,use_idx=False,is_train=True):
                 """
              This function returns a vector/array of rating values for a given
              user.

              Inputs:
                  user_id (int): External user ID unless use_idx=True.

                  use_idx (bool): If True, user_id is treated as an internal
                   user index.

                  is_train (bool): If True, return training ratings; otherwise
                  return test ratings.

              Outputs:
                  numpy array: Array of rating values for the user.
              """
                 if(use_idx==True):
                      user_id=self.idx_to_user_id[user_id]

                 data_idx=self.user_id_to_data_idx_train[user_id]

                 data_list=self.rating_data_by_user_train


                 if(is_train==False):
                     data_idx=self.user_id_to_data_idx_test[user_id]
                     data_list=self.rating_data_by_user_test

                 num=data_list[data_idx]
                 start=int(data_idx+1)
                 end=int(data_idx+num+1)

                 return data_list[start:end]
     def get_ids_vector_by_movie(self,movie_id,use_idx=False,is_train=True):
                 """
                This function returns a vector/array of user indices for a given
                movie.

                Inputs:
                    movie_id (int): External movie ID unless use_idx=True.

                    use_idx (bool): If True, movie_id is treated as an
                    internal movie index.

                    is_train (bool): If True, return training user IDs;
                    otherwise test user IDs.

                Outputs:
                    numpy array: Array of user indices who rated the movie.
                """
                 if(use_idx==True):
                      movie_id=self.idx_to_movie_id[movie_id]

                 data_idx=self.movie_id_to_data_idx_train[movie_id]
                 data_list=self.id_data_by_movie_train

                 if(is_train==False):
                     data_idx=self.movie_id_to_data_idx_test[movie_id]
                     data_list=self.id_data_by_movie_test

                 num=data_list[data_idx]
                 start=int(data_idx+1)
                 end=int(data_idx+num+1)
                 return data_list[start:end]
     def get_ids_vector_by_user(self,user_id,use_idx=False,is_train=True):
                 """
            This function returns a vector of movie indices rated by a given
            user.

            Inputs:
                user_id (int): External user ID unless use_idx=True.

                 use_idx (bool): If True, 'user_id' is treated as an internal
                user index.

                is_train (bool): If True, return training movie IDs; otherwise
                test movie IDs.

            Outputs:
                numpy array: Array of movie indices rated by the user.
            """
                 if(use_idx==True):
                      user_id=self.idx_to_user_id[user_id]

                 data_idx=self.user_id_to_data_idx_train[user_id]
                 data_list=self.id_data_by_user_train

                 if(is_train==False):
                     data_idx=self.user_id_to_data_idx_test[user_id]
                     data_list=self.id_data_by_user_test

                 num=data_list[data_idx]
                 start=int(data_idx+1)
                 end=int(data_idx+num+1)
                 return data_list[start:end]

     def get_movie_url(self,movie_id,use_idx=False,database=2):
             """
              This function returns the URL of a movie from MovieLens, IMDB, or
              TMDB.

              Inputs:
                  movie_id (int): External movie ID unless use_idx=True.

                  use_idx (bool): If True, treat 'movie_id' as an internal
                  index.

                  database (int): Select which URL source to generate:
                                  1 =>  MovieLens
                                  2 => IMDB
                                  3 => TMDB

              Outputs:
                  str: A fully generated URL pointing to the chosen movie page.
              """
             if(use_idx==True):
                 movie_id=self.idx_to_movie_id[movie_id]
             source={
                 1:'https://movielens.org/movies',
                 2: 'http://www.imdb.com/title',
                 3: 'https://www.themoviedb.org/movie'
             }
             id={
                 1:movie_id,
                 2:'tt'+str(self.movie_data[movie_id]['imdbId']),
                 3:self.movie_data[movie_id]['tmbdId']
             }
             url_id=id[database]
             src=source[database]
             return src+f'/{str(url_id)}/'
     def get_movie_poster(self, movie_id, use_idx=False):
            """
            This function accesses the OMDB API to retrieve the poster URL for a
            movie.

            Inputs:
                movie_id (int): External movie ID unless use_idx=True.

                use_idx (bool): If True, treat 'movie_id' as an internal index.

            Outputs:
                str or None: The poster image URL if available; otherwise None.
            """
            if use_idx:
                movie_id = self.idx_to_movie_id[movie_id]
            data=self.movie_data[movie_id]
            imdb_id = 'tt'+data.get('imdbId')
            tmbd_id = data.get('tmbdId')
           
            OMDB_KEY = self.OMDB_api_key
            TMDB_KEY = self.TMDB_api_key
            
            if(OMDB_KEY != ''):
                if imdb_id:
                 
                        url = f'http://www.omdbapi.com/?i={imdb_id}&apikey={OMDB_KEY}'
                        try:
                            response = requests.get(url, timeout=10)
                            
                            data = response.json()
                        except ValueError:

                            print(f"Error fetching poster:ID={movie_id}")
                            data = None
                        if(data != None):
                            
                                poster_url = data.get('Poster')
                                if(poster_url):
                                      return poster_url
          
            if(TMDB_KEY != ''):
                  if  tmbd_id:
  
                        url = f"https://api.themoviedb.org/3/movie/{tmbd_id}?api_key={TMDB_KEY}"
                        try:
                            
                            response = requests.get(url, timeout=10)
                            data = response.json()
                            poster_path = data.get('poster_path')
                           
                            if poster_path:
                                # Build full image URL
                                return f"https://image.tmdb.org/t/p/w500{poster_path}"
                            return None
                        except ValueError:
                            print(f"Error fetching poster:ID={movie_id}")
                            return None

                        poster_url = data.get('Poster')
                        if(poster_url):
                                    return poster_url
            return None
     def generate_image(self,movie_id,use_idx=False,display=True):
        """
        This function downloads and returns a movie poster as a PIL Image object.

        Inputs:
            movie_id (int): External movie ID unless use_idx=True.
            use_idx (bool): If True, treat 'movie_id' as an internal index.
            display (bool): If True, automatically display the image.

        Outputs:
            Pillow Image Object or None: The poster image object, or None
            if unavailable.
        """
        poster_url=self.get_movie_poster(movie_id,use_idx)
        if poster_url and poster_url != 'N/A':
           try:
                    response = requests.get(poster_url)
                    img = Image.open(BytesIO(response.content))
                    if(display==True):
                        img.show()
                    return img
           except:
              return None
        else:
            print("No poster available")
            return None
     def get_movie_data(self,movie_id,use_idx=False):
             """
          This function retrieves all metadata associated with a specific movie.

          Inputs:
              movie_id (int): External movie ID unless use_idx=True.

              use_idx (bool): If True, treat 'movie_id' as an internal index.

          Outputs:
              dict: Dictionary containing stored metadata for the movie.
           """
             if(use_idx==True):
                 movie_id=self.idx_to_movie_id[movie_id]
             return self.movie_data[movie_id]
     