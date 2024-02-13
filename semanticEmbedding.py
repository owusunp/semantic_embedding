import pandas as pd
import tensorflow_hub as hub
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import numpy as np
from math import ceil

class SemanticSearch:
    def __init__(self):
        self.use = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
        self.fitted = False

    def fit(self, data, batch=500, n_neighbors=5, entries_per_cluster=4):
        self.data = data
        self.embeddings, self.clusters = self.get_text_embedding(data, batch=batch, entries_per_cluster=entries_per_cluster)
        n_neighbors = min(n_neighbors, len(self.embeddings))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(self.embeddings)
        self.fitted = True

    def __call__(self, text, return_data=True):
        inp_emb = self.use([text])
        neighbors = self.nn.kneighbors(inp_emb, return_distance=False)[0]
        if return_data:
            return [self.data[i] for i in neighbors]
        else:
            return neighbors

    def get_text_embedding(self, texts, batch=512, entries_per_cluster=4):
        embeddings = []
        for i in range(0, len(texts), batch):
            text_batch = texts[i:i+batch]
            emb_batch = self.use(text_batch)
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)

        n_clusters = ceil(len(texts) / entries_per_cluster)
        kmeans = KMeans(n_clusters=n_clusters, random_state=3)
        clusters = kmeans.fit_predict(embeddings)

        text_clusters = [[] for _ in range(n_clusters)]
        for text, cluster in zip(texts, clusters):
            text_clusters[cluster].append(text)

        uneven_clusters = [cluster for cluster in text_clusters if len(cluster) > entries_per_cluster]
        for cluster in uneven_clusters:
            num_excess_entries = len(cluster) - entries_per_cluster
            for _ in range(num_excess_entries):
                smallest_cluster = min(text_clusters, key=len)
                text = cluster.pop(0)
                smallest_cluster.append(text)

        return embeddings, text_clusters

# Initialize the SemanticSearch object
semantic_search = SemanticSearch()


# Load the human responses CSV file
responses_file_path = '/Users/owusunp/Desktop/embeddingFolder/embedding/output/desc_data.csv'

# Load the responses into a DataFrame
responses_df = pd.read_csv(responses_file_path, header=None, names=['ID', 'CLIP-IQA Attribute', 'Response'])

# Extract the responses for embedding
responses = responses_df['Response'].tolist()


# Load the dimensions CSV file 
dimensions_file_path = '/Users/owusunp/Desktop/embeddingFolder/embedding/dimensions.csv'


# Load the dimensions into a DataFrame
dimensions_df = pd.read_csv(dimensions_file_path, header=None, names=['Dimension', 'Words'], quotechar='"')

# Combine the 'Dimension' and 'Words' columns for embedding
dimensions = (dimensions_df['Dimension'] + "," + dimensions_df['Words']).tolist()

# Fit the semantic search with the dimensions
semantic_search.fit(dimensions)

# Now you can find the closest dimension for each participant response
for response in responses:
    closest_matches = semantic_search(response)
    print(f"Closest matches for '{response}':", closest_matches)

# Specify the output file path
output_file_path = '/Users/owusunp/Desktop/embeddingFolder/embedding/output/output_for_embedding.csv'

try:
    # Open the file for writing
    with open(output_file_path, 'w') as file:
        # Iterate over responses and write closest matches to the file
        for response in responses:
            closest_matches = semantic_search(response)
            file.write(f"Closest matches for '{response}': {closest_matches}\n")
except Exception as e:
    print(f"An error occurred: {e}")
#python /Users/owusunp/Desktop/embeddingFolder/embedding/semanticEmbedding.py