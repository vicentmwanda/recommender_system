from numba import njit, prange
from numba.typed import List
import numpy as np

#setting numpy seed
SEED=42
np.random.seed(SEED)

#The training functions for bias only have been vectorized and optimized with numba
#parallel processing using Numba
@njit(parallel=True)
def update_user_biases_only_parallel(user_biases, item_biases, user_data, lambda_param, gamma):
    """This function updates all user biases only in parallel with vectorization.
    Parameters:
              user_biases : array of current user bias values (size M)
              item_biases : array of current item bias values (size N)
              user_data : array with movie indices based on user
              lambda_param : float
              gamma : float
    """
    M = len(user_biases)

    for m in prange(M):
        n_indice = user_data[m][0]
        ratings = user_data[m][1]

        if len(n_indice) == 0:
            continue

        residual = ratings - item_biases[n_indice]

        numerator = lambda_param * np.sum(residual)
        denominator = lambda_param * len(n_indice) + gamma
        user_biases[m] = numerator / denominator


@njit(parallel=True)
def update_item_biases_only_parallel(user_biases, item_biases, item_data, lambda_param, gamma):
    """This function updates all item biases only in parallel with vectorization
     Parameters:
              user_biases : array of current user bias values (size M)
              item_biases : array of current item bias values (size N)
              item_data : array of user indices based on movie
              F: array of feature vectors (N x K)
              movie_feature_data: array of features indices based on movie
              lambda_param : float
              gamma :float
    """
    N = len(item_biases)

    for n in prange(N):

        m_indice = item_data[n][0]
        ratings = item_data[n][1]

        if len(m_indice) == 0:
            continue


        residual = ratings - user_biases[m_indice]

        numerator = lambda_param * np.sum(residual)
        denominator = lambda_param * len(m_indice) + gamma

        item_biases[n] = (numerator / denominator)
@njit(parallel=True)
def compute_rmse_bias_only_parallel(user_biases, item_biases, item_data_train, item_data_test):
    """This function computes train and test RMSE in parallel.

    Parameters:
              user_biases : array of current user bias values (size M)
              item_biases : array of current item bias values (size N)
              item_data_train : array of training user indices based on movie
              item_data_test : array of test user indices based on movie
    """
    N = len(item_biases)

    sse_train = 0.0
    count_train = 0
    sse_test = 0.0  #squared error
    count_test = 0

    for n in prange(N):
        # Train RMSE
        m_indice_train = item_data_train[n][0]
        ratings_train = item_data_train[n][1]

        if len(m_indice_train) > 0:
            pred =  user_biases[m_indice_train] + item_biases[n]
            err = (ratings_train - pred) ** 2
            sse_train += np.sum(err)
            count_train += len(m_indice_train)

        # Test RMSE
        m_indice_test = item_data_test[n][0]
        ratings_test = item_data_test[n][1]

        if len(m_indice_test) > 0:
            pred =  user_biases[m_indice_test] + item_biases[n]
            err = (ratings_test - pred) ** 2
            sse_test += np.sum(err)
            count_test += len(m_indice_test)

    return sse_train, count_train, sse_test, count_test

#The training functions have been vectorized and optimized with numba
@njit(parallel=True)
def update_user_biases_parallel(user_biases, item_biases, U, V, user_data, lambda_param, gamma):
    """This function updates all user biases in parallel with vectorization.
    Parameters:
              user_biases : array of current user bias values (size M)
              item_biases : array of current item bias values (size N)
              U : array of user  embedding vectors (M x K)
              V : array of item  embeddings vectors (N x K)
              user_data : array with movie indices based on user
              lambda_param : float
              gamma : float
    """
    M = len(user_biases)

    for m in prange(M):
        n_indice = user_data[m][0]
        ratings = user_data[m][1]

        if len(n_indice) == 0:
            continue

        V_n = V[n_indice]
        predictions_without_user_bias = V_n @ U[m] + item_biases[n_indice]
        residual = ratings - predictions_without_user_bias

        numerator = lambda_param * np.sum(residual)
        denominator = lambda_param * len(n_indice) + gamma
        user_biases[m] = numerator / denominator


@njit(parallel=True)
def update_item_biases_parallel(user_biases, item_biases, U, V, item_data, lambda_param, gamma):
    """This function updates all item biases in parallel with vectorization
     Parameters:
              user_biases : array of current user bias values (size M)
              item_biases : array of current item bias values (size N)
              U : array of user  embedding vectors (M x K)
              V : array of item  embeddings vectors (N x K)
              item_data : array of user indices based on movie
              F: array of feature vectors (N x K)
              movie_feature_data: array of features indices based on movie
              lambda_param : float
              gamma :float
    """
    N = len(item_biases)

    for n in prange(N):

        m_indice = item_data[n][0]
        ratings = item_data[n][1]

        if len(m_indice) == 0:
            continue

        U_m = U[m_indice]



        predictions_without_item_bias = U_m @ V[n] + user_biases[m_indice]
        residual = ratings - predictions_without_item_bias

        numerator = lambda_param * np.sum(residual)
        denominator = lambda_param * len(m_indice) + gamma


        item_biases[n] = (numerator / denominator)

@njit(parallel=True)
def update_feature_vectors_parallel(feature_movie_data,movie_feature_data,V,F,K):
    """This function updates all feature vectors in parallel.
    Parameters:
              feature_movie_data: array of movie indices based on feature/genre.
              movie_feature_data: array of features indices based on movie
              V : array of item  embeddings vectors (N x K)
              F: array of feature vectors (N x K)
              K: latent dimension
    """

    F_total = len(F)  # total number of features

    for a in prange(F_total):
        m_indice = feature_movie_data[a]  # indices for movies containing feature a

        vector_sum = np.zeros(K)
        feature_sum = np.zeros(K)
        denom_sum = 0.0

        for n in m_indice:
            v_n = V[n]
            
            #indices for genres for given movie
            f_indice = movie_feature_data[n]

            # number of features in movie n
            Fn = len(f_indice)

            # sum all feature vectors for this movie
            F_l = F[f_indice].sum(axis=0)

            # subtract feature vector for a in order to exclude index a
            feature_sum += (F_l - F[a]) / Fn

            vector_sum += v_n / np.sqrt(Fn)
            denom_sum += 1.0 / Fn

        F[a] = (vector_sum - feature_sum) / (1.0 + denom_sum)



@njit(parallel=True)
def update_user_embeddings_parallel(user_biases, item_biases, U, V, user_data, lambda_param, tau, K):
    """This function updates all user embeddings in parallel with vectorization
    Parameters:
              user_biases : array of current user bias values (size M)
              item_biases : array of current item bias values (size N)
              U : array of user  embedding vectors (M x K)
              V : array of item  embeddings vectors (N x K)
              user_data : array with movie indices based on user
              lambda_param : float
              tau : float
              K : integer
    """
    M = len(U)

    for m in prange(M):
        n_indice = user_data[m][0]
        ratings = user_data[m][1]

        if len(n_indice) == 0:
            continue

        V_n = V[n_indice]
        residual = ratings - user_biases[m] - item_biases[n_indice]

        # vectorized operations
        A = lambda_param * (V_n.T @ V_n) + tau * np.eye(K)
        b = lambda_param * (V_n.T @ residual)
        U[m] = np.linalg.solve(A, b)

@njit(parallel=True)
def update_item_embeddings_parallel(user_biases, item_biases, U, V, item_data, F, movie_feature_data, lambda_param, tau,
                                    K,feature_sums, include_features):
    """This function updates all item embeddings in parallel with vectorization.

    Parameters:
              user_biases : array of current user bias values (size M)
              item_biases : array of current item bias values (size N)
              U : array of user  embedding vectors (M x K)
              V : array of item  embeddings vectors (N x K)
              item_data : array of user indices based on movie
              lambda_param : float
              tau : float
              K : integer
    """
    N = len(V)

    for n in prange(N):
        m_indice = item_data[n][0]
        ratings = item_data[n][1]

        if len(m_indice) == 0:
            continue

        U_m = U[m_indice]


        residual = ratings - user_biases[m_indice] - item_biases[n]

        # vectorized operations
        A = lambda_param * (U_m.T @ U_m) + tau * np.eye(K)

        if(include_features == True):
            #feature indices for movie n
            f_indice = movie_feature_data[n]
            #feature vectors
            F_l = F[f_indice]
            #total number of vectors
            Fn = len(F_l)

            feature_sum = F_l.sum(axis=0)*tau/np.sqrt(Fn)

            b = lambda_param * (U_m.T @ residual) + feature_sum
            V[n] = np.linalg.solve(A, b)
            feature_sums[n]=feature_sum
        else:
            #computing without features
            b = lambda_param * (U_m.T @ residual)
            V[n] = np.linalg.solve(A, b)



@njit(parallel=True)
def compute_rmse_parallel(user_biases, item_biases, U, V, item_data_train, item_data_test):
    """This function computes train and test RMSE in parallel.

    Parameters:

              user_biases : array of current user bias values (size M)
              item_biases : array of current item bias values (size N)
              U : array of user  embedding vectors (M x K)
              V : array of item  embeddings vectors (N x K)
              item_data_train : array of training user indices based on movie
              item_data_test : array of test user indices based on movie
    """
    N = len(item_biases)

    sse_train = 0.0
    count_train = 0
    sse_test = 0.0  #squared error
    count_test = 0

    for n in prange(N):
        # Train RMSE
        m_indice_train = item_data_train[n][0]
        ratings_train = item_data_train[n][1]

        if len(m_indice_train) > 0:
            U_m = U[m_indice_train]
            embedding = U_m @ V[n]
            pred = embedding + user_biases[m_indice_train] + item_biases[n]
            err = (ratings_train - pred) ** 2
            sse_train += np.sum(err)
            count_train += len(m_indice_train)

        # Test RMSE
        m_indice_test = item_data_test[n][0]
        ratings_test = item_data_test[n][1]

        if len(m_indice_test) > 0:
            U_m = U[m_indice_test]
            embedding = U_m @ V[n]
            pred = embedding + user_biases[m_indice_test] + item_biases[n]
            err = (ratings_test - pred) ** 2
            sse_test += np.sum(err)
            count_test += len(m_indice_test)

    return sse_train, count_train, sse_test, count_test


#preparing data for numba
def prepare_data_for_numba(data, M, N, Fn):
    """This function prepares data in Numba compatible format.

    Parameters:
              data : data structure object
              M : number of users
              N : number of items
    """


    #Preparing user data
    user_data = List()
    for m in range(M):
        n_indice = np.array(data.get_ids_vector_by_user(m, True), dtype=np.int64)
        ratings = np.array(data.get_ratings_vector_by_user(m, True), dtype=np.float64)
        user_data.append((n_indice, ratings))

    #Preparing item data (all ratings) for training
    item_data = List()
    for n in range(N):
        m_indice = np.array(data.get_ids_vector_by_movie(n, True), dtype=np.int64)
        ratings = np.array(data.get_ratings_vector_by_movie(n, True), dtype=np.float64)
        item_data.append((m_indice, ratings))


    #Preparing item data (for test loss only)
    item_data_test = List()
    for n in range(N):
        m_indice = np.array(data.get_ids_vector_by_movie(n, True, is_train=False), dtype=np.int64)
        ratings = np.array(data.get_ratings_vector_by_movie(n, True, is_train=False), dtype=np.float64)
        item_data_test.append((m_indice, ratings))

    #Preparing feature data (training only)

    #feature data based on feature
    feature_movie_data = List()
    for f in range(Fn):
        m_indice = np.array(data.get_movie_data_by_feature_idx(f), dtype=np.int64)
        feature_movie_data.append(m_indice)

    #feature data based on movie
    movie_feature_data = List()
    for n in range(N):
        f_indice = np.array(data.get_feature_data_by_movie_idx(n), dtype=np.int64)
        movie_feature_data.append(f_indice)

    return user_data, item_data, item_data_test, feature_movie_data, movie_feature_data
