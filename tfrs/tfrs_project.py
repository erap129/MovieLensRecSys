import argparse
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import pickle
import plotly.graph_objects as go
import plotly.io as pio
import re
from cachier import cachier
from imdb import Cinemagoer
pio.renderers.default = "browser"


ia = Cinemagoer()
MOVIE_FEATURES = ['movie_title', 'movie_genres', 'movie_title_text']
USER_FEATURES = ['user_id', 'timestamp', 'bucketized_user_age']


class UserModel(tf.keras.Model):
    def __init__(self, unique_user_ids, embedding_size=32, additional_features=(), additional_feature_info=None):
        super().__init__()
        self.additional_embeddings = {}

        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_size)
        ])

        if 'timestamp' in additional_features:
            self.additional_embeddings['timestamp'] = tf.keras.Sequential([
                tf.keras.layers.Discretization(additional_feature_info['timestamp_buckets'].tolist()),
                tf.keras.layers.Embedding(len(additional_feature_info['timestamp_buckets']) + 1, embedding_size),
            ])

        if 'bucketized_user_age' in additional_features:
            self.user_age_normalizer = tf.keras.layers.Normalization(axis=None)
            self.user_age_normalizer.adapt(additional_feature_info['bucketized_user_age'])
            self.additional_embeddings['bucketized_user_age'] = tf.keras.Sequential([self.user_age_normalizer,
                                                                                     tf.keras.layers.Reshape([1])])

    def call(self, inputs):
        return tf.concat([self.user_embedding(inputs['user_id'])] +
                         [self.additional_embeddings[k](inputs[k]) for k in self.additional_embeddings],
                         axis=1)


class MovieModel(tf.keras.Model):
    def __init__(self, unique_movie_titles, additional_features, additional_feature_info, embedding_size=32):
        super().__init__()
        self.additional_embeddings = {}

        self.title_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_movie_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_size)
        ])

        if 'movie_genres' in additional_features:
            self.additional_embeddings['movie_genres'] = tf.keras.Sequential([
                tf.keras.layers.Embedding(max(additional_feature_info['unique_movie_genres']) + 1, embedding_size),
                tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis=1))
            ])

        if 'movie_title_text' in additional_features:
            max_tokens = 10_000
            self.title_vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_tokens)
            self.title_vectorizer.adapt(unique_movie_titles)
            self.additional_embeddings['movie_title_text'] = tf.keras.Sequential([
                self.title_vectorizer,
                tf.keras.layers.Embedding(max_tokens, embedding_size, mask_zero=True),
                tf.keras.layers.GlobalAveragePooling1D(),
            ])

    def call(self, inputs):
        return tf.concat([self.title_embedding(inputs['movie_title'])] +
                         [self.additional_embeddings[k](inputs[k]) for k in self.additional_embeddings],
                         axis=1)


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
    def __init__(self, layer_sizes, movies, unique_movie_titles, n_unique_user_ids, embedding_size,
                 additional_features, additional_feature_info):
        super().__init__()
        self.additional_features = additional_features
        self.query_model = QueryCandidateModel(layer_sizes, UserModel(n_unique_user_ids,
                                                                      embedding_size=embedding_size,
                                                                      additional_features=self.additional_features,
                                                                      additional_feature_info=additional_feature_info))
        self.candidate_model = QueryCandidateModel(layer_sizes, MovieModel(unique_movie_titles,
                                                                           embedding_size=embedding_size,
                                                                           additional_features=self.additional_features,
                                                                           additional_feature_info=additional_feature_info))
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=(movies
                            .apply(tf.data.experimental.dense_to_ragged_batch(128))
                            .map(self.candidate_model)),
            ),
        )

    def compute_loss(self, features, training=False):
        query_embeddings = self.query_model({
            'user_id': features['user_id'],
            **{k: features[k] for k in self.additional_features if k in USER_FEATURES}
        })
        movie_embeddings = self.candidate_model({
            'movie_title': features['movie_title'],
            **{k: features[k] for k in self.additional_features if k in MOVIE_FEATURES}
        })
        return self.task(query_embeddings, movie_embeddings, compute_metrics=not training)


def get_movie_length_wrapper(movie_title):
    return get_movie_length(movie_title.numpy().decode())


@cachier()
def get_movie_length(movie_title_str):
    movielens_year = int(re.search('\((\d+)\)', movie_title_str).groups(0)[0])
    try:
        movie = [x for x in ia.search_movie(movie_title_str) if 'year' in x and x['year'] == movielens_year][0]
    except IndexError:
        try:
            movie = ia.search_movie(re.search('(.*?) \(', movie_title_str).groups(0)[0])[0]
        except IndexError:
            return 90
    ia.update(movie, ['technical'])
    try:
        runtime_str = movie.get('tech')['runtime']
    except KeyError:
        return 90
    try:
        return int(re.search('\((\d+)', runtime_str[0]).groups(0)[0])
    except AttributeError:
        try:
            return int(re.search('(\d+)', runtime_str[0]).groups(0)[0])
        except AttributeError:
            return 90


class MovieLensTrainer:
    def __init__(self, num_epochs, embedding_size, layer_sizes, additional_feature_sets, retrain):
        self.num_epochs = num_epochs
        self.embedding_size = embedding_size
        self.layer_sizes = tuple(layer_sizes)
        self.additional_feature_sets = additional_feature_sets
        self.retrain = retrain
        self.movies = (tfds.load("movielens/1m-movies", split="train")
                       .map(lambda x: {**x,
                                       'movie_title_text': x['movie_title'],
                                       'movie_length': tf.py_function(func=get_movie_length_wrapper,
                                                                      inp=[x['movie_title']],
                                                                      Tout=[tf.int32])}))
        self.ratings = (tfds.load("movielens/1m-ratings", split="train")
                        .map(lambda x: {**x, 'movie_title_text': x['movie_title']})
                        .shuffle(100_000, seed=42, reshuffle_each_iteration=False))
        self.all_ratings = list(self.ratings.map(lambda x: {'movie_title': x["movie_title"],
                                                            'user_id': x['user_id'],
                                                            'bucketized_user_age': x['bucketized_user_age'],
                                                            'movie_genres': x['movie_genres'],
                                                            'timestamp': x['timestamp']})
                                .apply(tf.data.experimental.dense_to_ragged_batch(len(self.ratings))))[0]
        all_movies = list(self.movies.apply(tf.data.experimental.dense_to_ragged_batch(len(self.movies))))[0]
        self.unique_movie_titles = np.unique(all_movies['movie_title'])
        self.unique_movie_genres, _ = tf.unique(all_movies['movie_genres'].flat_values)
        self.unique_user_ids = np.unique(self.all_ratings['user_id'])
        self.max_timestamp = self.all_ratings['timestamp'].numpy().max()
        self.min_timestamp = self.all_ratings['timestamp'].numpy().min()
        self.additional_feature_info = {'timestamp_buckets': np.linspace(self.min_timestamp, self.max_timestamp,
                                                                         num=1000),
                                        'unique_movie_genres': self.unique_movie_genres,
                                        'bucketized_user_age': self.all_ratings['bucketized_user_age']}

    def train_all_models(self):
        models = {}
        for additional_features in self.additional_feature_sets:
            model, history = self.get_movielens_model(tuple(additional_features))
            models[tuple(additional_features)] = (model, history)
        return models

    def get_movielens_model(self, additional_features):
        folder_name = f'saved_models/{self.num_epochs}_{self.embedding_size}_{self.layer_sizes}' \
                      f'_{tuple(sorted(additional_features))}'
        if os.path.exists(folder_name) and not self.retrain:
            model = tf.saved_model.load(f'{folder_name}/model')
            with open(f'{folder_name}/model_history.pkl', 'rb') as f:
                model_history = pickle.load(f)
            return model, model_history
        else:
            return self.train_movielens_model(additional_features, folder_name)

    def train_movielens_model(self, additional_features, folder_name):
        trainset = (self.ratings
                    .take(80_000)
                    .shuffle(100_000)
                    .apply(tf.data.experimental.dense_to_ragged_batch(2048))
                    .cache())
        testset = (self.ratings
                   .skip(80_000)
                   .take(20_000)
                   .apply(tf.data.experimental.dense_to_ragged_batch(2048))
                   .cache())
        model = MovieLensModel(self.layer_sizes, self.movies, self.unique_movie_titles, self.unique_user_ids,
                               self.embedding_size,
                               additional_features=additional_features,
                               additional_feature_info=self.additional_feature_info)
        model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1), run_eagerly=True)
        model_history = model.fit(
            trainset,
            validation_data=testset,
            validation_freq=5,
            epochs=self.num_epochs,
            verbose=1)
        model.task = tfrs.tasks.Retrieval()
        model.compile()
        tf.saved_model.save(model, f'{folder_name}/model')
        with open(f'{folder_name}/model_history.pkl', 'wb') as f:
            pickle.dump(model_history.history, f)
        return model, model_history.history


def plot_training_runs(model_histories):
    first_key = list(model_histories.keys())[0]
    num_validation_runs = len(model_histories[first_key]["val_factorized_top_k/top_100_categorical_accuracy"])
    epochs = [(x + 1) * 5 for x in range(num_validation_runs)]
    fig = go.Figure()
    for k, v in model_histories.items():
        fig.add_trace(go.Scatter(x=epochs, y=v["val_factorized_top_k/top_100_categorical_accuracy"],
                                 mode='lines',
                                 name='_'.join(k)))
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--embedding_size', type=int, default=32)
    parser.add_argument('--layer_sizes', nargs='+', default=[32])
    parser.add_argument('--additional_feature_sets', nargs='+', help='options: timestamp', action='append')
    parser.add_argument('--generate_recommendations_for_user', type=int, default=42)
    parser.add_argument('--retrain', action='store_true')
    args = parser.parse_args()

    tf.data.experimental.enable_debug_mode
    if ['None'] in args.additional_feature_sets:
        args.additional_feature_sets.remove(['None'])
        args.additional_feature_sets.append([])
    else:
        args.additional_feature_sets = [x for x in args.additional_feature_sets if len(x) > 0]

    movielens_trainer = MovieLensTrainer(args.num_epochs,
                                         args.embedding_size,
                                         args.layer_sizes,
                                         args.additional_feature_sets,
                                         args.retrain)
    models = movielens_trainer.train_all_models()
    fig = plot_training_runs({k: v[1] for k, v in models.items()})
    fig.show()

    print(f'Run settings: {vars(args)}')
    if args.generate_recommendations_for_user:
        for model_name, model_obj in models.items():
            rating_idx = tf.where(movielens_trainer.all_ratings['user_id'] ==
                                  str(args.generate_recommendations_for_user).encode()).numpy().squeeze()[0]
            user_details_for_query = {'user_id': [movielens_trainer.all_ratings['user_id'][rating_idx].numpy().decode()],
                                      **{col_name: [movielens_trainer.all_ratings[col_name][rating_idx].numpy()]
                                         for col_name in model_name}}
            model = model_obj[0]
            index = tfrs.layers.factorized_top_k.BruteForce()
            index.index_from_dataset(movielens_trainer.movies.apply(tf.data.experimental.dense_to_ragged_batch(100)).map(model.candidate_model))
            _, titles = index(model.query_model(user_details_for_query))
            title_names = np.array([movielens_trainer.unique_movie_titles[x] for x in titles.numpy().squeeze()])
            print(f"Recommendations for model {model_name}, user {args.generate_recommendations_for_user}:\n"
                  f" {title_names[:10]}")


