from kfserving import KFServingClient
import compose_pipeline
import kfp
import tensorflow as tf
import requests
import time
import argparse
import logging

def read_args():
    parser = argparse.ArgumentParser(description='This software is able to create pipeline for MNIST, run an example and terminate KFServing. Please add your arguments...')
    
    parser.add_argument('--model_dir', dest='model_dir', nargs='?',
                         default="/train/model",
                        help='Directory of the model to save (default: "/train/model")')
    parser.add_argument('--data_dir', dest='data_dir', nargs='?',
                         default="/train/data",
                        help='Directory of the data to save and load from (default: "/train/data")')
    parser.add_argument('--export_bucket', dest='export_bucket', nargs='?',
                         default="mnist",
                        help='Which bucket the model need to be exported to (default: "mnist")')
    parser.add_argument('--model_name', dest='model_name', nargs='?',
                         default="mnist",
                        help='Name of the model (default: "mnist")')
    parser.add_argument('--model_version', dest='model_version', nargs='?',
                         default="1",
                        help='Version of the model (default: 1)')
    parser.add_argument('--experiment_name', dest='experiment_name', nargs='?',
                         default="End-to-End MNIST Pipeline",
                        help='Name of the experiment (default: "End-to-End MNIST Pipeline")')

    return vars(parser.parse_args())
    
args = read_args()

def start_pipeline():
    pipeline_func = compose_pipeline.mnist_pipeline
    run_name = pipeline_func.__name__ + " run"

    arguments = {
        "model_dir": args["model_dir"],
        "data_dir": args["data_dir"],
        "export_bucket": args["export_bucket"],
        "model_name": args["model_name"],
        "model_version": args["model_version"],
    }

    client = kfp.Client()
    run_result = client.create_run_from_pipeline_func(
        pipeline_func,
        experiment_name=args["experiment_name"],
        run_name=run_name,
        arguments=arguments,
    )
    
def send_example():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test / 255.0

    image_index = 1005

    tf_serving_req = {"instances": x_test[image_index : image_index + 1].tolist()}
    model_name = args["model_name"]
    url = f"http://{model_name}-predictor-default.{compose_pipeline.NAMESPACE}.svc.cluster.local/v1/models/{model_name}:predict"
    
    result = None
    while result is None:
        try:
            print(f"Sending {tf_serving_req} to {url}")

            result = requests.post(url, data = tf_serving_req)
        except:
            print("Can't connect...")
            print("Wait 30...")
            time.sleep(30)
            print("Keep trying...")
            pass
   
    return result

    
def shut_down_server():
    KFServing = KFServingClient()
    KFServing.delete(args["model_name"], namespace=compose_pipeline.NAMESPACE)

def main():
    args = read_args()
    print(f"Namespace: {compose_pipeline.NAMESPACE}")
    start_pipeline()

    result = send_example()
    
    print(f"Got back: {result}")
    shut_down_server()

if __name__ == "__main__":
    main()