import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs


class UserModel(tf.keras.Model):
    def __init__(self, n_unique_user_ids, embedding_size=32):
        super().__init__()

        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(n_unique_user_ids + 1, embedding_size)
        ])

    def call(self, inputs):
        return self.user_embedding(inputs['user_id'])


class MovieModel(tf.keras.Model):
    def __init__(self, unique_movie_titles, embedding_size=32):
        super().__init__()
        self.title_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_movie_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_size)
        ])

    def call(self, titles):
        return self.title_embedding(titles)


class QueryCandidateModel(tf.keras.Model):
    def __init__(self, layer_sizes, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model
        self.dense_layers = tf.keras.Sequential()
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation='relu'))
        self.dense_layers.add(tf.keras.layers.Dense(layer_sizes[-1]))

    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)


class MovieLensModel(tfrs.models.Model):
    def __init__(self, layer_sizes, movies, unique_movie_titles, n_unique_user_ids):
        super().__init__()
        self.query_model = QueryCandidateModel(layer_sizes, UserModel(n_unique_user_ids))
        self.candidate_model = QueryCandidateModel(layer_sizes, MovieModel(unique_movie_titles))
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=movies.batch(128).map(self.candidate_model),
            ),
        )

    def compute_loss(self, features, training=False):
        query_embeddings = self.query_model({
            'user_id': features['user_id']
        })
        movie_embeddings = self.candidate_model(features['movie_title'])
        return self.task(query_embeddings, movie_embeddings, compute_metrics=not training)


if __name__ == '__main__':
    movies = (tfds.load("movielens/100k-movies", split="train")
              .map(lambda x: x["movie_title"]))
    ratings = (tfds.load("movielens/100k-ratings", split="train")
               .shuffle(100_000, seed=42, reshuffle_each_iteration=False))
    unique_movie_titles = np.unique(np.concatenate(list(ratings.map(lambda x: x["movie_title"]).batch(1000))))
    unique_user_ids = np.unique(np.concatenate(list(ratings.map(lambda x: x["user_id"]).batch(1000))))
    trainset = (ratings
                .take(80_000)
                .shuffle(100_000)
                .apply(tf.data.experimental.dense_to_ragged_batch(2048))
                .cache())

    testset = (ratings
               .skip(80_000)
               .take(20_000)
               .apply(tf.data.experimental.dense_to_ragged_batch(4096))
               .cache())

    num_epochs = 300

    model = MovieLensModel([32], movies, unique_movie_titles, len(unique_user_ids))
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1), run_eagerly=True)

    one_layer_history = model.fit(
        trainset,
        validation_data=testset,
        validation_freq=5,
        epochs=num_epochs,
        verbose=1)

    accuracy = one_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"][-1]
    print(f"Top-100 accuracy: {accuracy:.2f}.")

