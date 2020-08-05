from kfserving import KFServingClient
import compose_pipeline
import kfp
import tensorflow as tf
import requests

def start_pipeline():
    mnist_pipeline = compose_pipeline.mnist_pipeline

    pipeline_func = mnist_pipeline
    run_name = pipeline_func.__name__ + " run"
    experiment_name = "End-to-End MNIST Pipeline"

    arguments = {
        "model_dir": "/train/model",
        "data_dir": "/train/data",
        "export_bucket": "mnist",
        "model_name": "mnist",
        "model_version": "1",
    }

    client = kfp.Client()
    run_result = client.create_run_from_pipeline_func(
        pipeline_func,
        experiment_name=experiment_name,
        run_name=run_name,
        arguments=arguments,
    )
    
def send_example():
    with open('namespace', 'r') as namespace_file:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_test = x_test / 255.0

        image_index = 1005
        
        tf_serving_req = {"instances": x_test[image_index : image_index + 1].tolist()}
        model = 'mnist'
        namespace = namespace_file.read()
        url = f"http://{model}-predictor-default.{namespace}.svc.cluster.local/v1/models/{model}:predict"

        print(f"Sending {tf_serving_req} to {url}")

        x = requests.post(url, data = tf_serving_req)
        return x

    
def shut_down_server():
    with open('namespace', 'r') as namespace_file:
        KFServing = KFServingClient()
        KFServing.delete('mnist', namespace=namespace_file.read())

def main():
    start_pipeline()

    result = None
    while result is None:
        try:
            # connect
            result = send_example()
        except:
            print("Can't connect...")
            print("Keep trying...")
            pass

    print(f"Got back: {result}")

    shut_down_server()

if __name__ == "__main__":
    main()