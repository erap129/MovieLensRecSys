import argparse

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs


class UserModel(tf.keras.Model):
    def __init__(self, n_unique_user_ids, embedding_size=32, additional_features=(), additional_feature_info=None):
        super().__init__()
        self.additional_embeddings = {}

        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(n_unique_user_ids + 1, embedding_size)
        ])

        if 'timestamp' in additional_features:
            self.additional_embeddings['timestamp'] = tf.keras.Sequential([
                tf.keras.layers.Discretization(additional_feature_info['timestamp_buckets'].tolist()),
                tf.keras.layers.Embedding(len(additional_feature_info['timestamp_buckets']) + 1, 32),
            ])

        if 'movie_genres' in additional_features:
            self.additional_embeddings['movie_genres'] = tf.keras.Sequential([
                tf.keras.layers.Embedding(len(additional_feature_info['unique_movie_genres']) + 1, 32),
                tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis=1))
            ])

    def call(self, inputs):
        return tf.concat([self.user_embedding(inputs['user_id'])] +
                         [self.additional_embeddings[k](inputs[k]) for k in self.additional_embeddings],
                         axis=1)


class MovieModel(tf.keras.Model):
    def __init__(self, unique_movie_titles, embedding_size=32):
        super().__init__()
        self.additional_embeddings = {}

        self.title_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_movie_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_size)
        ])

    def call(self, inputs):
        return self.title_embedding(inputs)


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
    def __init__(self, layer_sizes, movies, unique_movie_titles, n_unique_user_ids, additional_features,
                 additional_feature_info):
        super().__init__()
        self.additional_features = additional_features
        self.query_model = QueryCandidateModel(layer_sizes, UserModel(n_unique_user_ids,
                                                                      additional_features=self.additional_features,
                                                                      additional_feature_info=additional_feature_info))
        self.candidate_model = QueryCandidateModel(layer_sizes, MovieModel(unique_movie_titles))
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=(movies
                            .batch(128)
                            .map(self.candidate_model)),
            ),
        )

    def compute_loss(self, features, training=False):
        query_embeddings = self.query_model({
            'user_id': features['user_id'],
            **{k: features[k] for k in self.additional_features}
        })
        movie_embeddings = self.candidate_model(features['movie_title'])
        return self.task(query_embeddings, movie_embeddings, compute_metrics=not training)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--embedding_size', type=int, default=32)
    parser.add_argument('--layer_sizes', nargs='+', default=[32])
    parser.add_argument('--additional_features', nargs='+', default=[], help='options: timestamp,')
    args = parser.parse_args()

    movies = (tfds.load("movielens/100k-movies", split="train")
              .map(lambda x: x['movie_title']))
    ratings = (tfds.load("movielens/100k-ratings", split="train")
               .shuffle(100_000, seed=42, reshuffle_each_iteration=False))
    unique_movie_titles = np.unique(np.concatenate(list(ratings.map(lambda x: x["movie_title"]).batch(1000))))
    unique_movie_genres = np.unique(np.concatenate(list(ratings.map(lambda x: x["movie_genres"]).as_numpy_iterator())))
    unique_user_ids = np.unique(np.concatenate(list(ratings.map(lambda x: x["user_id"]).batch(1000))))

    timestamps = np.concatenate(list(ratings.map(lambda x: x["timestamp"]).batch(100)))
    max_timestamp = timestamps.max()
    min_timestamp = timestamps.min()
    additional_feature_info = {'timestamp_buckets': np.linspace(min_timestamp, max_timestamp, num=1000),
                               'unique_movie_genres': unique_movie_genres}

    trainset = (ratings
                .take(80_000)
                .shuffle(100_000)
                .apply(tf.data.experimental.dense_to_ragged_batch(2048))
                .cache())

    testset = (ratings
               .skip(80_000)
               .take(20_000)
               .apply(tf.data.experimental.dense_to_ragged_batch(2048))
               .cache())

    model = MovieLensModel(args.layer_sizes, movies, unique_movie_titles, len(unique_user_ids),
                           additional_features=args.additional_features,
                           additional_feature_info=additional_feature_info)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1), run_eagerly=True)

    model_history = model.fit(
        trainset,
        validation_data=testset,
        validation_freq=3,
        epochs=args.num_epochs,
        verbose=1)

    accuracy = model_history.history["val_factorized_top_k/top_100_categorical_accuracy"][-1]

    print(f'Run settings: {vars(args)}')
    print(f"Top-100 accuracy: {accuracy:.2f}")

