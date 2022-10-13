import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN

from sklearn.mixture import GaussianMixture
from pyclustering.cluster.clarans import clarans

from sklearn import metrics
from sklearn.metrics import silhouette_score

import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

pd.options.display.float_format = '{:.5f}'.format
warnings.filterwarnings(action='ignore')


def autoML(data, encodings, scalers, features, models, parameter_s):
    for encoding in encodings:
        for scaler in scalers:
            for feature in features:
                # 원본 데이터 유지를 위해서 복사
                data_copy = data.copy()

                # 원본 데이터의 2% 사용
                ratio = int(len(data_copy) / 100 * 2)

                # 데이터 샘플링
                data_copy = data_copy.sample(ratio, random_state=42)

                # Encoding
                encoder = encoding.fit(data_copy[["ocean_proximity"]])
                data_copy["ocean_proximity"] = encoder.transform(data_copy[["ocean_proximity"]])

                # Scaling
                scaled = scaler.fit_transform(data_copy)
                data_copy = pd.DataFrame(scaled, columns=data_copy.columns, index=list(data_copy.index.values))

                # Correlation
                corr_matrix = data_copy.corr()
                corr_matrix = corr_matrix['median_house_value'].sort_values(ascending=False).nlargest(feature + 1)

                # median_house_value로 sort했기 때문에 항상 첫번째는 median_house_value로 나옴
                feature = corr_matrix.index[0:feature + 1]

                # 선택된 Feature만 가진 데이터 프레임 생성
                data_feature = data_copy[feature.values]

                for model in models:
                    if model == "KMeans":
                        print()
                        kmeans_f(data_feature, encoding, scaler, len(feature), parameter_s[0])
                    elif model == "DBSCAN":
                        print()
                        dbscan_f(data_feature, encoding, scaler, len(feature), parameter_s[1])
                    elif model == "EM":
                        print()
                        em_f(data_feature, encoding, scaler, len(feature), parameter_s[2])
                    elif model == "CLARANS":
                        print()
                        clarans_f(data_feature, encoding, scaler, len(feature), parameter_s[3])
                    else:
                        print("Input model Error")
                        exit(0)


def kmeans_f(data_feature, encoding, scaler, feature, n_clusters_s):
    for n_clusters in n_clusters_s:
        # 원본 데이터 유지를 위해서 복사
        original_data_feature = data_feature.copy()

        # 클러스터링을 위해서 median_house_value 값을 제거
        data_feature_copy = data_feature.drop(['median_house_value'], axis=1)

        # KMeans 알고리즘 실행
        kmeans_model = KMeans(n_clusters=n_clusters, init="k-means++",
                              random_state=42).fit(data_feature_copy)
        # cluster label 생성
        data_feature_copy["cluster"] = kmeans_model.labels_

        # median_house_value 데이터 다시 붙이기
        data_feature_copy['median_house_value'] = original_data_feature['median_house_value'].values

        # 원본 데이터를 가지고 median_house_value기준으로 정렬
        data_feature_copy = data_feature_copy.sort_values(ascending=False, by='median_house_value')

        # 원본 데이터 값을 등분 (등분 수 = cluster 수)
        data_feature_copy['target'] = pd.cut(data_feature_copy['median_house_value'], n_clusters,
                                             labels=range(0, n_clusters))

        compare_score = len(
            data_feature_copy.loc[data_feature_copy['target'] == data_feature_copy['cluster']]) / len(
            data_feature_copy) * 100

        title1 = "Encoding : " + str(encoding) + " Scaler : " + str(
            scaler) + " Clustering : K-means" + " n_clusters : " + str(n_clusters) + " Feature : " + str(feature - 1)

        print(title1)

        # compare_score
        print("Compare to quantiles clustering result : %0.2f%%" % compare_score)

        # purity_score
        print("Compare the clustering with purity_score : %0.2f%%" % purity_score(data_feature_copy['target'],
                                                                                  data_feature_copy['cluster']))

        # silhouette_score
        print("Compare the clustering with silhouette_score : %0.2f%%" % silhouette_score(data_feature_copy,
                                                                                          data_feature_copy["cluster"],
                                                                                          metric='euclidean'))

        score1 = "\nCompare to quantiles clustering result : %0.2f%%" % compare_score
        score2 = "\nCompare the clustering with purity_score : %0.2f%%" % purity_score(data_feature_copy['target'],
                                                                                       data_feature_copy['cluster'])
        score3 = "\nCompare the clustering with silhouette_score : %0.2f%%" % silhouette_score(data_feature_copy,
                                                                                               data_feature_copy[
                                                                                                   "cluster"],
                                                                                               metric='euclidean')
        title2 = title1 + score1 + score2 + score3

        data_feature_copy = data_feature_copy.drop(['median_house_value'], axis=1)

        # plot
        plot = sns.pairplot(data_feature_copy, hue='cluster', palette="bright", corner=True)
        plot.fig.suptitle(title2, y=1.05)
        plot.fig.set_size_inches(10, 10)
        title1 = title1.replace(":", "-")
        save = "./fig/kmeans/" + title1 + ".png"
        plot.savefig(save)


def dbscan_f(data_feature, encoding, scaler, feature, eps_s):
    for eps in eps_s:
        # 원본 데이터 유지를 위해서 복사
        original_data_feature = data_feature.copy()

        # 클러스터링을 위해서 median_house_value 값을 제거
        data_feature_copy = data_feature.drop(['median_house_value'], axis=1)

        # DBScan 알고리즘 실행
        dbscan_model = DBSCAN(eps=eps).fit(data_feature_copy)

        # cluster label 생성
        data_feature_copy["cluster"] = dbscan_model.labels_

        # median_house_value 데이터 다시 붙이기
        data_feature_copy['median_house_value'] = original_data_feature['median_house_value'].values

        # 원본 데이터를 가지고 median_house_value기준으로 정렬
        data_feature_copy = data_feature_copy.sort_values(ascending=False, by='median_house_value')
        bin = data_feature_copy["cluster"].max() + 2

        # 원본 데이터 값을 등분 (등분 수 = cluster 수)
        data_feature_copy['target'] = pd.cut(data_feature_copy['median_house_value'], bin,
                                             labels=range(0, bin))

        compare_score = len(
            data_feature_copy.loc[data_feature_copy['target'] == data_feature_copy['cluster']]) / len(
            data_feature_copy) * 100

        title1 = "Encoding : " + str(encoding) + " Scaler : " + str(
            scaler) + " Clustering : DBScan " + "eps : " + str(eps) + " Feature : " + str(feature - 1)

        print(title1)

        # compare_score
        print("Compare to quantiles clustering result : %0.2f%%" % compare_score)

        # purity_score
        print("Compare the clustering with purity_score : %0.2f%%" % purity_score(data_feature_copy['target'],
                                                                                  data_feature_copy['cluster']))

        # silhouette_score
        try:
            print("Compare the clustering with silhouette_score : %0.2f%%" % silhouette_score(data_feature_copy,
                                                                                              data_feature_copy[
                                                                                                  "cluster"],
                                                                                              metric='euclidean'))
        except:
            print("Compare the clustering with silhouette_score : %0.2f%%" % 0)

        score1 = "\nCompare to quantiles clustering result : %0.2f%%" % compare_score
        score2 = "\nCompare the clustering with purity_score : %0.2f%%" % purity_score(data_feature_copy['target'],
                                                                                       data_feature_copy['cluster'])
        try:
            score3 = "\nCompare the clustering with silhouette_score : %0.2f%%" % silhouette_score(data_feature_copy,
                                                                                                   data_feature_copy[
                                                                                                       "cluster"],
                                                                                                   metric='euclidean')
        except:
            score3 = "\nCompare the clustering with silhouette_score : %0.2f%%" % 0

        title2 = title1 + score1 + score2 + score3

        data_feature_copy = data_feature_copy.drop(['median_house_value'], axis=1)

        # plot
        plot = sns.pairplot(data_feature_copy, hue='cluster', palette="bright", corner=True)
        plot.fig.suptitle(title2, y=1.05)
        plot.fig.set_size_inches(10, 10)
        title1 = title1.replace(":", "-")
        save = "./fig/dbscan/" + title1 + ".png"
        plot.savefig(save)


def em_f(data_feature, encoding, scaler, feature, n_components_s):
    for n_components in n_components_s:
        # 원본 데이터 유지를 위해서 복사
        original_data_feature = data_feature.copy()

        # 클러스터링을 위해서 median_house_value 값을 제거
        data_feature_copy = data_feature.drop(['median_house_value'], axis=1)

        # GaussianMixture 알고리즘 실행
        gmm_model = GaussianMixture(n_components=n_components).fit(data_feature_copy)

        # cluster label 생성
        data_feature_copy["cluster"] = gmm_model.predict(data_feature_copy)

        # plot
        title1 = "Encoding : " + str(encoding) + " Scaler : " + str(
            scaler) + " Clustering : EM(GMM) " + "n_components : " + str(n_components) + " Feature : " + str(
            feature - 1)

        # median_house_value 데이터 다시 붙이기
        data_feature_copy['median_house_value'] = original_data_feature['median_house_value'].values

        # 원본 데이터를 가지고 median_house_value기준으로 정렬
        data_feature_copy = data_feature_copy.sort_values(ascending=False, by='median_house_value')

        # 원본 데이터 값을 등분 (등분 수 = cluster 수)
        data_feature_copy['target'] = pd.cut(data_feature_copy['median_house_value'], n_components,
                                             labels=range(0, n_components))

        compare_score = len(
            data_feature_copy.loc[data_feature_copy['target'] == data_feature_copy['cluster']]) / len(
            data_feature_copy) * 100

        print(title1)

        # compare_score
        print("Compare to quantiles clustering result : %0.2f%%" % compare_score)

        # purity_score
        print("Compare the clustering with purity_score : %0.2f%%" % purity_score(data_feature_copy['target'],
                                                                                  data_feature_copy['cluster']))

        # silhouette_score
        print("Compare the clustering with silhouette_score : %0.2f%%" % silhouette_score(data_feature_copy,
                                                                                          data_feature_copy["cluster"],
                                                                                          metric='euclidean'))

        score1 = "\nCompare to quantiles clustering result : %0.2f%%" % compare_score
        score2 = "\nCompare the clustering with purity_score : %0.2f%%" % purity_score(data_feature_copy['target'],
                                                                                       data_feature_copy['cluster'])
        score3 = "\nCompare the clustering with silhouette_score : %0.2f%%" % silhouette_score(data_feature_copy,
                                                                                               data_feature_copy[
                                                                                                   "cluster"],
                                                                                               metric='euclidean')
        title2 = title1 + score1 + score2 + score3

        data_feature_copy = data_feature_copy.drop(['median_house_value'], axis=1)

        # plot
        plot = sns.pairplot(data_feature_copy, hue='cluster', palette="bright", corner=True)
        plot.fig.suptitle(title2, y=1.05)
        plot.fig.set_size_inches(10, 10)
        title1 = title1.replace(":", "-")
        save = "./fig/em/" + title1 + ".png"
        plot.savefig(save)


def clarans_f(data_feature, encoding, scaler, feature, number_clusters_s):
    for number_clusters in number_clusters_s:
        # 원본 데이터 유지를 위해서 복사
        original_data_feature = data_feature.copy()

        # 클러스터링을 위해서 median_house_value 값을 제거
        data_feature_copy = data_feature.drop(['median_house_value'], axis=1)

        # Clarans 알고리즘 실행
        clarans_model = clarans(data_feature_copy.values.tolist(), number_clusters, 3, 5).process()

        # cluster label 생성
        idx_list = [-1 for i in range(0, len(data_feature_copy))]
        idx = 0
        for k in clarans_model.get_clusters():
            for i in k:
                idx_list[i] = idx
            idx = idx + 1
        data_feature_copy["cluster"] = idx_list

        # plot
        title1 = "Encoding : " + str(encoding) + " Scaler : " + str(
            scaler) + " Clustering : CLARANS " + "number_clusters : " + str(number_clusters) + " Feature : " + str(
            feature - 1)

        # median_house_value 데이터 다시 붙이기
        data_feature_copy['median_house_value'] = original_data_feature['median_house_value'].values

        # 원본 데이터를 가지고 median_house_value기준으로 정렬
        data_feature_copy = data_feature_copy.sort_values(ascending=False, by='median_house_value')

        # 원본 데이터 값을 등분 (등분 수 = cluster 수)
        data_feature_copy['target'] = pd.cut(data_feature_copy['median_house_value'], number_clusters,
                                             labels=range(0, number_clusters))

        compare_score = len(
            data_feature_copy.loc[data_feature_copy['target'] == data_feature_copy['cluster']]) / len(
            data_feature_copy) * 100

        print(title1)

        # compare_score
        print("Compare to quantiles clustering result : %0.2f%%" % compare_score)

        # purity_score
        print("Compare the clustering with purity_score : %0.2f%%" % purity_score(data_feature_copy['target'],
                                                                                  data_feature_copy['cluster']))

        # silhouette_score
        print("Compare the clustering with silhouette_score : %0.2f%%" % silhouette_score(data_feature_copy,
                                                                                          data_feature_copy["cluster"],
                                                                                          metric='euclidean'))

        score1 = "\nCompare to quantiles clustering result : %0.2f%%" % compare_score
        score2 = "\nCompare the clustering with purity_score : %0.2f%%" % purity_score(data_feature_copy['target'],
                                                                                       data_feature_copy['cluster'])
        score3 = "\nCompare the clustering with silhouette_score : %0.2f%%" % silhouette_score(data_feature_copy,
                                                                                               data_feature_copy[
                                                                                                   "cluster"],
                                                                                               metric='euclidean')
        title2 = title1 + score1 + score2 + score3

        data_feature_copy = data_feature_copy.drop(['median_house_value'], axis=1)

        # plot
        plot = sns.pairplot(data_feature_copy, hue='cluster', palette="bright", corner=True)
        plot.fig.suptitle(title2, y=1.05)
        plot.fig.set_size_inches(10, 10)
        title1 = title1.replace(":", "-")
        save = "./fig/clarans/" + title1 + ".png"
        plot.savefig(save)


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)

    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


if __name__ == '__main__':
    df = pd.read_csv("data/housing.csv")

    print("Colum info")
    print(df.info())
    print()

    print("Checking Nan value")
    print(df.isnull().sum())
    print()

    df.dropna(inplace=True)

    print("After drop Nan value")
    print(df.isnull().sum())
    print()

    print("Data shape : " + str(df.shape))

    encoding_s = [#OrdinalEncoder(),
        LabelEncoder()]
    scaler_s = [StandardScaler(), MinMaxScaler()]
    feature_s = [3, 4, 5, 6]
    model_s = ["KMeans", "DBSCAN", "EM", "CLARANS"]

    # KMeans
    n_clusters_s = [2, 4, 6, 8, 10, 12]

    # DBSCAN
    eps_s = [0.05, 0.1, 0.5, 1, 3]

    # EM
    n_components_s = [2, 4, 6, 8, 10, 12]

    # CLARANS
    number_clusters_s = [2, 4, 6, 8, 10, 12]

    parameter = [n_clusters_s, eps_s, n_components_s, number_clusters_s]
    autoML(df, encoding_s, scaler_s, feature_s, model_s, parameter)
