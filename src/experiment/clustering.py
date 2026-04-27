from sklearn.cluster import SpectralClustering
import pandas as pd



def cluster_on_data(pairwise_sim_path:str):
    adjacency_matrix = to_adjacency_matrix(pairwise_sim_path)
    categories = SpectralClustering(n_clusters = 8,affinity='precomputed').fit(adjacency_matrix)


def to_adjacency_matrix(pairwise_sim_path:str) -> pd.DataFrame:
    df = pd.read_json(pairwise_sim_path)
    df_cropped = df.pivot(index="nameX",columns=["nameY"],values=["D_stitch","D_ppm"])
    
    print(df_cropped)
    df_cropped.to_csv("temp.csv")

def test():
    to_adjacency_matrix("data/result/(ctx=False)/pairwise_similarity.json")

if __name__ == "__main__":
    test()