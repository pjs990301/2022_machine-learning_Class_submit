
#function definition comment#

def autoML(data, encodings, scalers, features, models, parameter_s)
Summary of the autoML function:
    The ‘autoML’ function is making the clustering result about input 
    parameters.
     Function get data and list of encoding, scaler function and 
    number of using features name of clustering models and 
    parameters corresponding to each models.
     Then, automatically make and run the models, and compare the clustered 
    result with the answer label feature ‘median_house_value’.

Parameters:
    data : Dataframe, Refined original data, 
           In this project the data is The California 
           Housing Prices Dataset(California Housing Prices | Kaggle)
    encodings : List, various encoding functions list.
    scalers : List, various scaling functions list.
    features : List, that sets how many features to use in   
               clustering.
    models : List, name of various clustering model list. 
    parameter_s : List, various parameters for each clustering model.

Return:
    None


def kmeans_f(data_feature, encoding, scaler, feature, n_clusters_s)
Summary of the kmeans_f function:
    The ‘kmeans_f’ function created and learned a kmeans clustering model. 
     The results are then extracted using several performance
    metrics(purity_score, silhouette_score).
     Finally, the results learned and performance indicators are visualized and
    shown.

Parameters:
   data_feature : data for data feature selected by correlation
   encoding : the name of the data encoding method used
   scaler : data scaler method name used
   feature : data feature by correlation
   n_clusters_s : kmeans model parameter_cluster number

Return:
   None


def dbscan_f(data_feature, encoding, scaler, feature, eps_s)
Summary of the dbscan_f function:
    The ‘dbscan_f’ function created and learned a dbscan clustering model. 
     The results are then extracted using several performance 
    metrics(purity_score, silhouette_score).
     Finally, the results learned and performance indicators are visualized and 
    shown.

Parameters:
   data_feature : data for data feature selected by correlation
   encoding : the name of the data encoding method used
   scaler : data scaler method name used
   feature : data feature by correlation
   eps_s: dbscan model parameter_Circle radius(distance)

Return:
   None

def em_f(data_feature, encoding, scaler, feature, n_components_s)
Summary of the em_f function:
    The em_f function created and learned a GaussianMixture clustering model. 
     The results are then extracted using several performance
    metrics(purity_score, silhouette_score).
     Finally, the results learned and performance indicators are visualized and
    shown.

Parameters:
   data_feature : data for data feature selected by correlation
   encoding : the name of the data encoding method used
   scaler : data scaler method name used
   feature : data feature by correlation
   n_components : kmeans_f model parameter_cluster number

Return:
   None


def clarans_f(data_feature, encoding, scaler, feature, number_clusters_s)
Summary of the clarans_f function:
    The ‘clarans_f’ function created and learned a clarans clustering model. 
     The results are then extracted using several performance 
    metrics(purity_score, silhouette_score).
     Finally, the results learned and performance indicators are visualized and 
    shown.

Parameters:
   data_feature : data for data feature selected by correlation
   encoding : the name of the data encoding method used
   scaler : data scaler method name used
   feature : data feature by correlation
   number_clusters_s : clarans model parameter_cluster number

Return:
   None

def purity_score(y_true, y_pred)
Summary of purity_score function:
    Calculate the purity score about input parameter ‘y_true’ and ‘y_pred’

Parameters:
   y_true : Clustering label of original data
   y_pred : Clustering label that we predicted

Return: score : float
   Clustering Quality Assessment Score
