import json # For loading the jokes dataset
from top2vec import Top2Vec # For creating the Top2Vec model
import tensorflow_hub as hub # For loading the Universal Sentence Encoder

# Load the Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Load the jokes dataset
json_data = json.load(open('reddit_jokes.json'))

# Extract the jokes from the dataset to a string list for the Top2Vec model
documents = [joke['title'] + ' ' + joke['body'] for joke in json_data]

# Create the Top2Vec model, using the Universal Sentence Encoder and our jokes dataset
model = Top2Vec(documents, speed='learn', workers=32, embedding_model=embed)

# Save the model
model.save('top2vec_model')