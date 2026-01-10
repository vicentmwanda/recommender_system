from my_recommender_package.ALS_operations import (update_feature_vectors_parallel,update_item_biases_parallel,update_item_embeddings_parallel,
              update_user_biases_parallel,update_user_embeddings_parallel,prepare_data_for_numba,compute_rmse_parallel)
from datetime import datetime
import numpy as np
import pickle
import gc
#setting numpy seed
SEED=42
np.random.seed(SEED)

def get_movie_data(database,top_n, min_rating=0, max_rating=0,min_target=0.5, max_target=5.0, add_data_rating = False):
     "This is function prepares and returns the details of the predicted movies from the system."
     selection=[]
     for (movie_idx,r) in top_n:
          #copy movie meta data dictionary
          data=database.get_movie_data(movie_idx,use_idx=True).copy()
          #adding information to dictionary
          data['movie_id']=database.idx_to_movie_id[movie_idx]
          data['url']=database.get_movie_url(movie_idx,True,2)
          #adding average rank from dataset
          if add_data_rating == True:
                    data['rating']=np.mean(database.get_data_by_movie_id(movie_idx,use_idx=True))
          #adding predicted score
          data['model_rating']=r
          #rescale rating
          data['scaled_rating']=rescale_rating(r,min_rating, max_rating, min_target, max_target)

          selection.append(data)
     return selection
     
def rescale_rating(rating, min_rating, max_rating, min_target=0.5, max_target=5.0):
    """
    This function rescales a rating from the models rating scale  to  target scale.

    Parameters:
        rating (float): 
        min_rating (float): minimum of the model ratings.
        max_rating (float): maximum of the model ratings.
        min_target (float): minimum of the target scale (default 0.5).
        max_target (float): maximum of the target scale (default 5.0).

    Output:
        float: rescaled rating.
    """
   
    if max_rating == min_rating :
        return (min_target + max_target) / 2
    return min_target + (rating - min_rating) / (max_rating - min_rating) * (max_target - min_target)
class RecommenderModel():
     def __init__(self,data_structure=None,K=20, tau=1, gamma=0.1, lambda_param=1, epochs=15,
                  include_features=True, load_avg_genres=True, load_temp_data=True):
          """
            This function trains the matrix factorization model using parallelized
            and vectorized operations via Numba. It updates user/item biases and
            embeddings iteratively to minimize the regularized squared error.

            Training progress is monitored by computing RMSE on both training and
            test datasets at each epoch.

            Updates:
                self.U: user latent factor matrix

                self.V: item latent factor matrix

                self.user_biases: user bias vector

                self.item_biases: item bias vector

                self.train_nloglikelihood: list of training negative loss

                self.train_rmse: list of training RMSE values
                self.test_rmse: list of test RMSE values

                self.load_avg_genres (Boolean): indicates whether genre embeddings are initialized from averaged item embeddings

                self.load_temp_data (Boolean): indicates whether temporary data structures are saved and loaded with the model to pickle.
            """

          self.data_struct = data_structure
          if(data_structure != None):
                self.genre_labels=data_structure.genre_labels
          else:
                self.genre_labels=[]

          #boolean variable to check whether to add features to training
          self.include_features=include_features

          #number of users and movies

          self.M = len(self.data_struct.idx_to_user_id) if self.data_struct != None else 0
          self.N = len(self.data_struct.idx_to_movie_id) if self.data_struct != None else 0


          #number of features
          self.Fn=len(self.genre_labels)

          #initialization of biases
          self.user_biases = np.zeros(self.M, dtype=np.float64)
          self.item_biases = np.zeros(self.N, dtype=np.float64)

          # latent dimension
          self.K = K
          #user embeddings
          self.U = np.random.normal(loc=0, scale=1/np.sqrt(self.K), size=(self.M, self.K)).astype(np.float64)
          #item embeddings
          self.V = np.random.normal(loc=0, scale=1/np.sqrt(self.K), size=(self.N, self.K)).astype(np.float64)

          #feature embeddings
          self.F = np.random.normal(loc=0, scale=1/np.sqrt(self.K), size=(self.Fn, self.K)).astype(np.float64)

          # Hyperparameters
          self.lambda_param = lambda_param
          self.gamma = gamma
          self.tau = tau

          # Training settings
          self.epochs = epochs

          #genre embeddings generated as average of movie embeddings
          self.avg_genre_embeddings={}

          self.avg_genre_biases={}

          #boolean to track if genres initialized after 
          self.load_avg_genres = load_avg_genres

          #arrays to store metrics
          self.train_nloglikelihood = []

          self.train_rmse = []
          self.test_rmse = []
          
          #minimum and maximum scale used by users
          self.min_target = 0.5
          self.max_target = 5
          
          #list to store IDs for polarizing items 
          self.polarized=[]
          
          #arrays to temporary info from data structure when saving
          self.tmp_idx_to_movie_id=[]
          self.tmp_movie_data={}

          self.load_temp_data = load_temp_data
          
          
          
     def set_parameters(self,K=20,tau=1,gamma=0.1,lambda_param=1,epochs=15):
          """
          This function sets the parameters for the model.
          """
          #number of users and movies

          self.M = len(self.data_struct.idx_to_user_id) if self.data_struct!=None else 0
          self.N = len(self.data_struct.idx_to_movie_id) if self.data_struct!=None else 0

          #initialization of biases
          self.user_biases = np.zeros(self.M, dtype=np.float64)
          self.item_biases = np.zeros(self.N, dtype=np.float64)

          #user and item embeddings
          self.K = K
          self.U = np.random.normal(loc=0, scale=1/np.sqrt(self.K), size=(self.M, self.K)).astype(np.float64)
          self.V = np.random.normal(loc=0, scale=1/np.sqrt(self.K), size=(self.N, self.K)).astype(np.float64)

          #feature embeddings
          self.F= np.random.normal(loc=0, scale=1/np.sqrt(self.K), size=(self.Fn, self.K)).astype(np.float64)

          #hyperparameters
          self.lambda_param = lambda_param
          self.gamma = gamma
          self.tau = tau

          #resetting the training settings
          self.epochs = epochs

          self.train_nloglikelihood = []
         
          self.train_rmse = []
          self.test_rmse = []
    
     def save_to_picke_file(self,path=''):
         """
        This function saves the current model to a pickle file.
        

        Parameters:
            path (str, optional): Path where the pickle file will be saved. 
            If left empty, the model is saved in the working directory.

        """
         #copying the model's basic settings
         model_copy=RecommenderModel(self.data_struct,
                                      self.K, self.tau,
                                      self.gamma,self.lambda_param,self.epochs,
                                      include_features=self.include_features,
                                      load_temp_data=self.load_temp_data
                                      )

         #copying the trait vectors
         model_copy.U = self.U
         model_copy.V = self.V
         model_copy.F = self.F
         model_copy.user_biases = self.user_biases
         model_copy.item_biases = self.item_biases
         
         model_copy.avg_genre_embeddings =self.avg_genre_embeddings
         model_copy.avg_genre_biases=self.avg_genre_biases

         model_copy.polarized = self.polarized
         
         #saving temporary data from data structure
         if(model_copy.load_temp_data == True):
               
               model_copy.tmp_idx_to_movie_id = model_copy.data_struct.idx_to_movie_id
               model_copy.tmp_movie_data = model_copy.data_struct.movie_data
              
         
         #removing data structure to reduce size of saved model
         model_copy.data_struct=None

         with open(path, "wb") as f:
                 pickle.dump(model_copy, f)
                 f.close()
         #cleaning the memory
         del model_copy
         gc.collect()
     def load_from_pickle_file(self,path,data_struct=None, load_temp = False):
          """
            This function loads a previously saved model from a pickle file. 

            Parameters:
                path (str): Path to the pickle file containing the saved model.

                data_struct (object, optional): A data structure to attach to the
                    loaded model. 

                load_temp (bool, optional): If True and temporary data was saved,
                    restores movie index mappings and movie metadata.
          """
          with open(path, "rb") as f:
              model = pickle.load(f)
              model.data_struct =  data_struct
              #loading necessary temporary data from saved model
              if(model.load_temp_data == True and model.data_struct != None and load_temp == True):
                   model.data_struct.idx_to_movie_id = model.tmp_idx_to_movie_id
                   model.data_struct.movie_data = model.tmp_movie_data
          return model
     def load_data(self,data_structure):
         self.data_struct=data_structure
     def print_epochs(self,epoch,epochs,data,prints=10):
           every=epochs//prints
           if(every==0):
               every=1

           if((epoch+1)%every == 0 or epoch == epochs ):
                print(data)
     def train(self):
          """
            This function trains the matrix factorization model using parallelized
            and vectorized operations via Numba. It updates user/item biases and
            embeddings iteratively to minimize the regularized squared error.

            Training progress is monitored by computing RMSE on both training and
            test datasets at each epoch.

            Updates:
                self.U: user latent factor matrix

                self.V: item latent factor matrix

                self.user_biases: user bias vector

                self.item_biases: item bias vector

                self.train_nloglikelihood: list of training negative loss

                self.train_rmse: list of training RMSE values
                self.test_rmse: list of test RMSE values
            """

          print("Preparing data structures for Numba...")
          time_stamp=datetime.now().timestamp()
          (user_data, item_data, item_data_test,
          feature_movie_data, movie_feature_data) = prepare_data_for_numba(self.data_struct,self.M, self.N,self.Fn)
          
          print("Starting training with Numba parallelization + Vectorized Operations")
          
          user_biases=self.user_biases
          item_biases=self.item_biases

          K = self.K #dimension of the user and item vector embeddings

          U = self.U  # user vector embeddings
          V = self.V  # item vector embeddings

          F = self.F # feature vectors

          include_features = self.include_features

          #array to store the compute sum of feature terms for each movie
          feature_sums = np.zeros((self.N, self.K), dtype=np.float64)


          lambda_param=self.lambda_param
          gamma=self.gamma

          tau=self.tau

          #arrays to store the rmse and loss values
          self.train_nloglikelihood = []

          self.train_rmse = []
          self.test_rmse = []

          epochs=self.epochs

          for epoch in range(epochs):
              #updating user biases with parallelization and vectorization
              update_user_biases_parallel(user_biases, item_biases, U, V, user_data,
                                        lambda_param, gamma)

              #updating item biases with parallelization and vectorization
              update_item_biases_parallel(user_biases, item_biases, U, V, item_data,
                                        lambda_param, gamma)

              #updating user vectors with parallelization and vectorization
              update_user_embeddings_parallel(user_biases, item_biases, U, V, user_data,
                                            lambda_param, tau, K)

              #updating item vectors with parallelization and vectorization
              update_item_embeddings_parallel(user_biases, item_biases, U, V, item_data,
                                          F, movie_feature_data,lambda_param, tau, K,feature_sums,include_features)

              #updating feature vectors with parallelization
              if(include_features == True):

                   update_feature_vectors_parallel(feature_movie_data,movie_feature_data,V,F,K)

              #computing RMSE  with parallelization
              sse_train, count_train, sse_test, count_test = compute_rmse_parallel(
                  user_biases, item_biases, U, V, item_data, item_data_test)

              #computing regularization
              reg_term = 0.5 * gamma * np.sum(user_biases**2) + 0.5 * gamma * np.sum(item_biases**2)




              #adding for regularization based on features
              if (include_features == True):
                    item_term = V  - feature_sums

                    reg_term += 0.5 * tau * np.sum(U**2) + 0.5 * tau * np.sum(item_term**2)

                    reg_term +=  0.5 * tau * np.sum(F*F)
              else:
                    reg_term += 0.5 * tau * np.sum(U**2) + 0.5 * tau * np.sum(V**2)

              #computing loss function (negative loss function)
              Likelihood_train = (0.5 * lambda_param) * sse_train + reg_term

              #computing RMSE
              RMSE_train = np.sqrt(sse_train / max(count_train, 1))
              RMSE_test = np.sqrt(sse_test / max(count_test, 1))

              #printing progress
              self.print_epochs(epoch, epochs,
                          f'#### Epoch: {epoch+1}, Train RMSE: {RMSE_train:.4f}, Test RMSE: {RMSE_test:.4f}',
                          10)

              #storing metrics
              self.train_nloglikelihood.append(Likelihood_train)
    
              self.train_rmse.append(RMSE_train)
              self.test_rmse.append(RMSE_test)
          #cleaning memory after trianing
          del user_data
          del item_data
          del item_data_test
          del movie_feature_data
          del feature_movie_data
          del feature_sums
          gc.collect()

          time_stamp=datetime.now().timestamp()-time_stamp
          print("Training complete!")
          print(f'''
#########FINAL RMSE AND NEGATIVE LOSS #####
Train RMSE: {self.train_rmse[-1]:.04f}

Test RMSE: {self.test_rmse[-1]:.04f}

Train Negative Loss: {self.train_nloglikelihood[-1]:.04f}

Duration: {time_stamp:.3f} seconds
###########################################
''')
          if(self.load_avg_genres == True):
                print('Generating Average Genre Embeddings')
                self.generate_genre_parameters()

     def generate_user_parameters(self,user_ratings,epochs=50,genre_only=False,use_updated_genre=False):
          """
          This function computes the user embedding vector and user bias for a given
          set of user ratings using the model parameters.

          Parameters:
              user_ratings (list of tuples): List of (movie_index, rating) pairs
              for the user.

              epochs (int): Number of iterations for updating user parameters.

              genre_only (bool): If True, use only genre-based embeddings and biases for prediction.

          Outputs:
              user_vector (numpy array): The latent feature vector for the user.
              user_bias (float): The bias term for the user.
          """
          model=self
          tau=model.tau
          gamma=model.gamma
          lambda_param=model.lambda_param

          K = model.K

          if(genre_only==True):
              item_biases=model.avg_genre_biases
              if(use_updated_genre==True):
                    V=model.F
              else:
                    V=model.avg_genre_embeddings
              if len(V) == 0 or len(item_biases) == 0:
                   print('No genre parameters found!')
                   return np.zeros(K),0
          else:
              item_biases=model.item_biases
              V = model.V


          epochs=epochs

          user_bias=0
          user_vector = np.zeros(K)

          n_ratings = len(user_ratings)
          
          #to avoid underfitting for users with very few ratings
          effective_tau = tau * min(n_ratings / epochs, 1.0)
          effective_gamma = gamma * min(n_ratings / epochs, 1.0)

          for epoch in range(epochs):

                bias=0
                for (n,r) in user_ratings:
                  index = n
                  if(genre_only == True and use_updated_genre == True):
                             index = self.genre_labels.index(n)
                  embedding=np.dot(user_vector,V[index])
                  bias+=lambda_param*(r-(embedding+item_biases[n]))

                user_bias=bias/(lambda_param*n_ratings+effective_gamma)


                first_term= effective_tau * np.eye(K)
                second_term=np.zeros(K)
                for (n,r) in user_ratings:
                    index = n
                    if(genre_only == True and use_updated_genre == True):
                             index = self.genre_labels.index(n)
                    v_n= V[index]
                    first_term+=lambda_param*(np.outer(v_n,v_n))
                    residual=(r-user_bias-item_biases[n])
                    second_term+=lambda_param*v_n*residual

                user_vector= np.linalg.solve(first_term, second_term)
          return user_vector, user_bias
     def predict(self,user_ratings,number=10,show_stats=False,epochs=50,genre_only=False,use_updated_genre=False
                  ,item_factor=0.05,user_factor=0, filter=100, remove_polarized = True):
                """
                This function predicts top-N movie recommendations for a given user based
                on their provided ratings and the trained model parameters.

                Inputs:
                    user_ratings (list of tuples): List of (movie_index, rating)
                    pairs for the user.

                    number (int): Number of top movie recommendations to return.

                    show_stats (bool): If True, prints statistics of all
                    predicted ratings.

                    epochs (int): Number of epochs to generate user parameters.

                    genre_only (bool): If True, use only genre-based features
                    for prediction.

                    use_updated_genre (bool): If True, use genre embedddings from 
                    training update.

                    item_factor (float): Weight factor for item bias in the
                    predicted rating.

                    user_factor (float): Weight factor for user bias in the
                    predicted rating.

                    filter (int): Number of top movie filter to polarization

                Outputs:
                    list: A list of dictionaries with recommended movie details,
                    obtained    from the data structure using `get_movie_data`.
                """

                if(self.data_struct==None):
                         print('Model prediction failed. Data Structure Missing!')
                         return None
                #set genres in user choices
                database = self.data_struct
                genre_choices  = set()

                for id, _ in user_ratings:
              
                    if(genre_only == False):
                              genres = self.data_struct.get_movie_data(id,use_idx=True)['genre'].split('|')
                              genre_choices.update(genres)
                    else:
                          if(id in self.genre_labels):
                               genre_choices.add(id)

                N = number # number of top movies to recommend

                user_vector,user_bias=self.generate_user_parameters(user_ratings,epochs,genre_only=genre_only,use_updated_genre=use_updated_genre)

                
                movie_rating=[]

                item_biases=self.item_biases
                item_embedding=self.V
                

                for n in range(len(item_biases)):
                      rating = np.dot(user_vector,item_embedding[n])+item_factor*item_biases[n]+user_factor*user_bias
                      movie_rating.append((n,rating))

                all_ratings = [s for _, s in movie_rating]

                min_rating = min(all_ratings)
                max_rating = max(all_ratings)

                
                
                #removing items already rated
                rated_items = {n for n, r in user_ratings}
                filtered = [(i, s) for (i, s) in movie_rating if i not in rated_items]


                #sorting by predicted rating (descending)
                top = sorted(filtered, key=lambda x: x[1], reverse=True)
                
                 #selecting required number of movies, and
                #checking for polarizing movies
                #this ensures that the polarizing are not among a specific number of the top movies
                top_n = []
                polarization_count = filter
                if(len(genre_choices) > 0 and remove_polarized == True):
                        for idx, rating in top:
                                movie_id = database.idx_to_movie_id[idx] 
                                movie_data = self.data_struct.get_movie_data(movie_id)
                                genres = movie_data['genre'].split('|')
                                if not any(genre in genre_choices for genre in genres): 
                                             #skip the movie in the final prediction if it is polarizing
                                             if(movie_id in self.polarized and polarization_count > 0):
                                                      print(genre_choices,genres)
                                                      if(show_stats == True):
                                                            print(f"polarizing movie found! [ID:{movie_id}] {movie_data['title']} : {movie_data['genre']}")
                                                      polarization_count-=1
                                                      continue
                                if(len(top_n)>=N):
                                      break
                                top_n.append((idx,rating))
                                                         
                                          
                else:
                     top_n = top[:N]
                     
                
               
                

                #getting movie details  
                selection = get_movie_data(database, top_n, min_rating, max_rating, self.min_target, self.max_target)

                if(show_stats==True):
                   
                    print(f"\nPredicted ratings stats:")
                    print(f"  Min: {min_rating:.4f}")
                    print(f"  Max: {max_rating :.4f}")
                    print(f"  Mean: {np.mean(all_ratings):.4f}")
                    print(f"  Std: {np.std(all_ratings):.4f}")
                    print(f"  Movies considered: {len(movie_rating)}")
                    print("==================\n")
                return selection 
     def generate_genre_parameters(self):

                  """
                This function generates embeddings and bias values for each genre
                ased on bthe existing item (movie) embeddings and item biases.

                It computes the average latent embedding and average bias for each
                genre by aggregating over all movies that belong to that genre.

                Updates-:
                    self.genre_embeddings (dict): Mapping from genre name to
                    latent vector.

                    self.genre_biases (dict): Mapping from genre name to
                    bias value.
                """
                  genres=self.genre_labels
                  model=self
                  data_struct=self.data_struct
                  K=model.K
                  V=model.V

                  item_biases=model.item_biases

                  embeddings={i:np.zeros(K) for i in genres }

                  biases={i:0 for i in genres }

                  counts={i:0 for i in genres}

                  for n in range(len(V)):
                          v=V[n]
                          bias=item_biases[n]
                          id=data_struct.idx_to_movie_id[n]
                          movie_genre=data_struct.get_movie_data(id)['genre']
                          for genre in genres:
                              if genre in movie_genre:
                                    counts[genre]+=1
                                    embeddings[genre]+=v
                                    biases[genre]+=bias

                  for genre in genres:
                      if counts[genre] > 0:
                          embeddings[genre]=embeddings[genre]/counts[genre]
                          biases[genre]=biases[genre]/counts[genre]
                  self.avg_genre_embeddings=embeddings
                  self.avg_genre_biases=biases

