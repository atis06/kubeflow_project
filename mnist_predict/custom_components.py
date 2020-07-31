import kfp
from kfp.components import InputPath, OutputPath
from typing import NamedTuple

class CustomComponents(object):

    @staticmethod
    def download_dataset(data_dir: OutputPath(str)):
        """Download the MNIST data set to the KFP volume to share it among all steps"""

        import tensorflow_datasets as tfds

        tfds.load(name="mnist", data_dir=data_dir)
        
    @staticmethod 
    def train_model(data_dir: InputPath(str), model_dir: OutputPath(str)):
        """Trains a single-layer CNN for 5 epochs using a pre-downloaded dataset.
        Once trained, the model is persisted to `model_dir`."""

        import os
        import tensorflow as tf
        import tensorflow_datasets as tfds

        def normalize_image(image, label):
            """Normalizes images: `uint8` -> `float32`"""
            return tf.cast(image, tf.float32) / 255.0, label

        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(0.001),
            metrics=["accuracy"],
        )

        print(model.summary())
        ds_train, ds_info = tfds.load(
            "mnist",
            split="train",
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
            download=False,
            data_dir=data_dir,
        )

        # See: https://www.tensorflow.org/datasets/keras_example#build_training_pipeline
        ds_train = ds_train.map(
            normalize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
        ds_train = ds_train.batch(128)
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

        model.fit(
            ds_train, epochs=5,
        )

        model.save(model_dir)
        print(f"Model saved {model_dir}")
        print(os.listdir(model_dir))

    @staticmethod  
    def evaluate_model(
        data_dir: InputPath(str), model_dir: InputPath(str), metrics_path: OutputPath(str)
    ) -> NamedTuple("EvaluationOutput", [("mlpipeline_metrics", "Metrics")]):
        """Loads a saved model from file and uses a pre-downloaded dataset for evaluation.
        Model metrics are persisted to `/mlpipeline-metrics.json` for Kubeflow Pipelines
        metadata."""

        import json
        import tensorflow as tf
        import tensorflow_datasets as tfds
        from collections import namedtuple

        def normalize_image(image, label):
            return tf.cast(image, tf.float32) / 255.0, label

        ds_test, ds_info = tfds.load(
            "mnist",
            split="test",
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
            download=False,
            data_dir=data_dir,
        )

        # See: https://www.tensorflow.org/datasets/keras_example#build_training_pipeline
        ds_test = ds_test.map(
            normalize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        ds_test = ds_test.batch(128)
        ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

        model = tf.keras.models.load_model(model_dir)
        (loss, accuracy) = model.evaluate(ds_test)

        metrics = {
            "metrics": [
                {"name": "loss", "numberValue": str(loss), "format": "PERCENTAGE"},
                {"name": "accuracy", "numberValue": str(accuracy), "format": "PERCENTAGE"},
            ]
        }

        with open(metrics_path, "w") as f:
            json.dump(metrics, f)

        out_tuple = namedtuple("EvaluationOutput", ["mlpipeline_metrics"])

        return out_tuple(json.dumps(metrics))

    @staticmethod
    def export_model(
        model_dir: InputPath(str),
        metrics: InputPath(str),
        export_bucket: str,
        model_name: str,
        model_version: int,
    ):
        import os
        import boto3
        from botocore.client import Config

        s3 = boto3.client(
            "s3",
            endpoint_url="http://minio-service.kubeflow:9000",
            aws_access_key_id="minio",
            aws_secret_access_key="minio123",
            config=Config(signature_version="s3v4"),
        )

        # Create export bucket if it does not yet exist
        response = s3.list_buckets()
        export_bucket_exists = False

        for bucket in response["Buckets"]:
            if bucket["Name"] == export_bucket:
                export_bucket_exists = True

        if not export_bucket_exists:
            s3.create_bucket(ACL="public-read-write", Bucket=export_bucket)

        # Save model files to S3
        for root, dirs, files in os.walk(model_dir):
            for filename in files:
                local_path = os.path.join(root, filename)
                s3_path = os.path.relpath(local_path, model_dir)

                s3.upload_file(
                    local_path,
                    export_bucket,
                    f"{model_name}/{model_version}/{s3_path}",
                    ExtraArgs={"ACL": "public-read"},
                )

        response = s3.list_objects(Bucket=export_bucket)
        print(f"All objects in {export_bucket}:")
        for file in response["Contents"]:
            print("{}/{}".format(export_bucket, file["Key"]))