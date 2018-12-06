import argparse
import tensorflow as tf


CSV_COLUMNS = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin',
               'Embarked']
CSV_COLUMN_DEFAULTS = [[''], [''], [0], [''], [''], [0.0], [0], [0], [''], [0.0], [''], ['']]
LABEL_COLUMN = 'Survived'
LABELS = ['0', '1']

INPUT_COLUMNS = [
    tf.feature_column.categorical_column_with_vocabulary_list('Pclass', [1, 2, 3]),
    tf.feature_column.categorical_column_with_vocabulary_list('Sex', ['female', 'male']),
    tf.feature_column.categorical_column_with_vocabulary_list('Embarked', ['C', 'Q', 'S']),
    tf.feature_column.numeric_column('Age', dtype=tf.float32),
    tf.feature_column.numeric_column('SibSp'),
    tf.feature_column.numeric_column('Parch'),
    tf.feature_column.numeric_column('Fare', dtype=tf.float32),
    tf.feature_column.categorical_column_with_hash_bucket('PassengerId', hash_bucket_size=100, dtype=tf.string),
    tf.feature_column.categorical_column_with_hash_bucket('Name', hash_bucket_size=100, dtype=tf.string)
]

UNUSED_COLUMNS = set(CSV_COLUMNS) - {col.name for col in INPUT_COLUMNS} - {LABEL_COLUMN}


def parse_csv(rows_string_tensor):
    row_columns = tf.expand_dims(rows_string_tensor, -1)
    columns = tf.decode_csv(row_columns, record_defaults=CSV_COLUMN_DEFAULTS)
    features = dict(zip(CSV_COLUMNS, columns))
    for col in UNUSED_COLUMNS:
        features.pop(col)
    return features


def parse_label_column(label_string_tensor):
    table = tf.contrib.lookup.index_table_from_tensor(tf.constant(LABELS))
    return table.lookup(label_string_tensor)


def json_serving_input_fn():
    inputs = {}
    for feat in INPUT_COLUMNS:
        inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def input_fn(filenames, num_epochs=None, shuffle=True, skip_header_lines=0, batch_size=200):
    filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle:
        filename_dataset = filename_dataset.shuffle(len(filenames))

    dataset = filename_dataset.flat_map(lambda filename: tf.data.TextLineDataset(filename).skip(skip_header_lines))
    dataset = dataset.map(parse_csv)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 10)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    return features, parse_label_column(features.pop(LABEL_COLUMN))


def build_estimator(config, embedding_size=8, hidden_units=None):
    (pclass, sex, embarked, age, sibsp, parch, fare, passenger_id, name) = INPUT_COLUMNS
    age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    fare_buckets = tf.feature_column.bucketized_column(age, boundaries=[100, 200, 300, 400, 500])
    wide_columns = [
        tf.feature_column.crossed_column([fare_buckets, 'Pclass'], hash_bucket_size=int(1e4)),
        pclass, sex, embarked, age_buckets]
    deep_columns = [
        tf.feature_column.indicator_column(pclass),
        tf.feature_column.indicator_column(sex),
        tf.feature_column.indicator_column(embarked),
        tf.feature_column.embedding_column(passenger_id, dimension=embedding_size),
        tf.feature_column.embedding_column(name, dimension=embedding_size),
        age, sibsp, parch, fare
    ]
    return tf.estimator.DNNLinearCombinedClassifier(config=config, linear_feature_columns=wide_columns,
                                                    dnn_feature_columns=deep_columns,
                                                    dnn_hidden_units=hidden_units or [100, 70, 50, 25])


def run(_know_args=None):
    train_input = lambda: input_fn(_know_args.train_files, num_epochs=_know_args.train_steps, skip_header_lines=1,
                                   batch_size=_know_args.train_batch_size)
    eval_input = lambda: input_fn(_know_args.eval_files, batch_size=_know_args.eval_batch_size, skip_header_lines=1,
                                  shuffle=False)
    train_spec = tf.estimator.TrainSpec(train_input, max_steps=_know_args.train_steps)
    exporter = tf.estimator.FinalExporter('titanic', json_serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(eval_input, steps=_know_args.eval_steps, exporters=[exporter], name='titanic-eval')
    run_config = tf.estimator.RunConfig(model_dir=_know_args.job_dir)
    estimator = build_estimator(embedding_size=_know_args.embedding_size,
                                hidden_units=[max(2, int(_know_args.first_layer_size * _know_args.scale_factor ** i))
                                              for i in range(_know_args.num_layers)],
                                config=run_config)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


# --train-files data/train.csv --train-steps 1000 --eval-files data/eval.csv --job-dir titanic_ltung
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', help='GCS location to write checkpoints and export models', required=True)
    parser.add_argument('--train-files', dest='train_files', nargs='+', required=True)
    parser.add_argument('--train-steps', dest='train_steps', type=int, required=True)
    parser.add_argument('--eval-files', dest='eval_files', nargs='+', required=True)
    parser.add_argument('--train-batch-size', dest='train_batch_size', help='Batch size for training steps', type=int, default=40)
    parser.add_argument('--eval-batch-size', dest='eval_batch_size', help='Batch size for evaluation steps', type=int, default=40)
    parser.add_argument('--eval-steps', dest='eval_steps', help='Number of steps to run evalution for at each checkpoint', type=int, default=100)
    parser.add_argument('--embedding-size', help='Number of embedding dimensions for categorical columns', type=int, default=8)
    parser.add_argument('--first-layer-size', help='Number of nodes in the first layer of the DNN', type=int, default=100)
    parser.add_argument('--num-layers', help='Number of layers in the DNN', type=int, default=4)
    parser.add_argument('--scale-factor', help='How quickly should the size of the layers in the DNN decay', type=float, default=0.7)
    know_args, _ = parser.parse_known_args()
    run(_know_args=know_args)
