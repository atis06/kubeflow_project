# Team Getting Started Guide

This guide has been made for Challenge 035 - Data Scientist - Kubeflow project by Attila Kántor.

## What is Kubeflow?
**Kubeflow** is a **Kubernetes** based machine learning platform developed by Google to enhance and make easier the deployment of machine learning pipelines. Kubeflow implements distributed and scalable solutions through creating chain of separated **Docker** containers for elements of the ML pipeline (e.g. training, serving, monitoring and logging) on the Kubernetes cluster called components. The goal of the platform is to leverage the flexibility and reproducibility of Kubernetes and Docker. 

## How to install Kubeflow?
Kubeflow can be installed on Linux, Windows and also on Mac. I will show you how to make it work on Windows, but you can easily do it on your local system by following **this tutorial.**
Setup Kubeflow (MiniKF) on Windows:
You will need to install **Vagrant** and **Virtual Box**.
 1. Create a project folder.
 2. Open command prompt as administrator.
	*vagrant init arrikto/minikf
	vagrant up*
 3. Go to 10.10.10.10 in your browser, then follow the setup instructions.
 4. After these steps you will be able to use Kubeflow platform.

If you have any problems during the setup process please visit **this page**.

## What is mnist_predict?
Mnist_predict is an implementation of a Kubeflow pipeline written in Python. It has several functionality like:
 - creating a Kubeflow pipeline for the MNIST dataset 
 - running the Kubeflow pipeline and saving the resulting model 
 - setting up KFServing inference server, which is an inference engine to serve machine and deep learning models in Kubernetes.
 - deploying the model to KFServing 
 - inference an example 
 - shutting down KFServing inference server
### How to make mnist_predict work?
All you need to do is to go to Kubeflow, create a notebook server and connect to it. Open a terminal window and clone this project from Github. 
You need to install some dependecies in order to run the script properly. Run the following commands in the terminal:

    pip3 install kfp --user
    pip3 install azure=="4.0.0" --user
    pip3 install kubeflow --user --no-cache
    pip3 install kfserving --user
cd to the mnist_predict folder then run the following command to create MinIO object store in Kubernetes.

    kubectl apply -f config/minio_secret.yaml
Then you can actually execute the main script by this command:

    python3 start.py
The software is going to work with the default values, but you can customize them:

    python3 start.py -h
    usage: start.py [-h] [--model_dir [MODEL_DIR]] [--data_dir [DATA_DIR]]
                    [--export_bucket [EXPORT_BUCKET]] [--model_name [MODEL_NAME]]
                    [--model_version [MODEL_VERSION]]
                    [--experiment_name [EXPERIMENT_NAME]]
    
    This software is able to create pipeline for MNIST, run an example and
    terminate KFServing. Please add your arguments...
    
    optional arguments:
      -h, --help            show this help message and exit
      --model_dir [MODEL_DIR]
                            Directory of the model to save (default:
                            "/train/model")
      --data_dir [DATA_DIR]
                            Directory of the data to save and load from (default:
                            "/train/data")
      --export_bucket [EXPORT_BUCKET]
                            Which bucket the model need to be exported to
                            (default: "mnist")
      --model_name [MODEL_NAME]
                            Name of the model (default: "mnist")
      --model_version [MODEL_VERSION]
                            Version of the model (default: 1)
      --experiment_name [EXPERIMENT_NAME]
                            Name of the experiment (default: "End-to-End MNIST
                            Pipeline")

 After the program started, it will create the pipeline and start to request the KFServing inference server with an example until the answer arrives. If everything is working properly, the the software will terminate the inference server in the final step.
 As a result you should see something like this in Kubeflow > Pipelines > Experiments:

If you have any questions regarding this topic, please feel free to contact with me.
