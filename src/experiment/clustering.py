from collections import defaultdict

from sklearn.cluster import spectral_clustering
from sklearn.manifold import spectral_embedding
import pandas as pd

from model.utils import save_json


def cluster_on_data(pairwise_sim_path: str, distance_field_name:str):
    adjacency_matrix = to_adjacency_matrix(pairwise_sim_path, distance_field_name)
    npMatrix = adjacency_matrix.to_numpy()

    embedding = spectral_embedding(adjacency=npMatrix, n_components=2)
    print(embedding)
    coordinates = [
        {
            "piece_name": piece_name,
            "x": float(embedding[i][0]),
            "y": float(embedding[i][1]),
        }
        for i, piece_name in enumerate(adjacency_matrix.index.to_list())
    ]
    # print(coordinates)

    # print(npMatrix)
    categories = spectral_clustering(n_clusters=4, affinity=npMatrix)

    # print(categories)
    labeled = pd.Series(categories, index=adjacency_matrix.index)
    grouped = {k: v.to_list() for k, v in labeled.groupby(labeled).groups.items()}
    return (labeled.to_dict(),grouped, coordinates)


def to_adjacency_matrix(
    pairwise_sim_path: str, distance_field_name: str
) -> pd.DataFrame:
    df = pd.read_json(pairwise_sim_path)
    df = df.pivot(index="nameX", columns="nameY", values=distance_field_name)
    adjacency = df.fillna(df.T)
    adjacency = adjacency.fillna(
        0
    )  # since we didn't calculate d(x,x), fill it with zero for d to be a distance
    return adjacency


def cluster_test(result_folder):
    # to_adjacency_matrix("data/result/(ctx=False)/pairwise_similarity.json")
    clustering_result = defaultdict(lambda: defaultdict(dict))
    for model_name in ["ppm", "stitch"]:
        for with_context in [True, False]:
            categories,partition, embedding = cluster_on_data(
          f"{result_folder}/(ctx={with_context})/pairwise_similarity.json",
          distance_field_name=f"D_{model_name}"
      )
            ctxName = f"ctx_{with_context}"
            for entry in embedding:
                entry["category"]=categories[entry["piece_name"]]
            clustering_result[model_name][ctxName]= {
        "partition" : partition,
        "embedding" : embedding,
        }
    save_json(f"{result_folder}/clustering.json", clustering_result)

def cluster_edit_distances(result_folder):
    clustering_result = defaultdict(lambda: defaultdict(dict))
    for model_name in ["SED", "TED(zss)"]:
        categories, partition, embedding = cluster_on_data(
            f"{result_folder}/edit_distances(normalized).json",
            distance_field_name=model_name,
        )
        for entry in embedding:
            entry["category"] = categories[entry["piece_name"]]
        clustering_result[model_name] = {
            "partition": partition,
            "embedding": embedding,
        }
    save_json(f"{result_folder}/clustering_edit_distance(normalized).json", clustering_result)
