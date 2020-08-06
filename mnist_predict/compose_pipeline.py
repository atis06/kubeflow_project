from custom_components import CustomComponents

import kfp.components as components
import kfp.dsl as dsl
import kubeflow.fairing.utils

NAMESPACE = kubeflow.fairing.utils.get_current_k8s_namespace()

def train_and_serve(
    data_dir: str,
    model_dir: str,
    export_bucket: str,
    model_name: str,
    model_version: int,
):
    # For GPU support, please add the "-gpu" suffix to the base image
    BASE_IMAGE = "mesosphere/kubeflow:1.0.1-0.3.1-tensorflow-2.2.0"

    downloadOp = components.func_to_container_op(
        CustomComponents.download_dataset, base_image=BASE_IMAGE
    )()

    trainOp = components.func_to_container_op(CustomComponents.train_model, base_image=BASE_IMAGE)(
        downloadOp.output
    )

    evaluateOp = components.func_to_container_op(CustomComponents.evaluate_model, base_image=BASE_IMAGE)(
        downloadOp.output, trainOp.output
    )

    exportOp = components.func_to_container_op(CustomComponents.export_model, base_image=BASE_IMAGE)(
        trainOp.output, evaluateOp.output, export_bucket, model_name, model_version
    )

    # Create an inference server from an external component
    kfserving_op = components.load_component_from_file(
        "config/component.yaml"
    )
    kfserving = kfserving_op(
        action="create",
        default_model_uri=f"s3://{export_bucket}/{model_name}",
        model_name=model_name,
        namespace=NAMESPACE,
        framework="tensorflow",
    )

    kfserving.after(exportOp)

# See: https://github.com/kubeflow/kfserving/blob/master/docs/DEVELOPER_GUIDE.md#troubleshooting
def op_transformer(op):
    op.add_pod_annotation(name="sidecar.istio.io/inject", value="false")
    return op


@dsl.pipeline(
    name="End-to-End MNIST Pipeline",
    description="A sample pipeline to demonstrate multi-step model training, evaluation, export, and serving",
)
def mnist_pipeline(
    model_dir: str = "/train/model",
    data_dir: str = "/train/data",
    export_bucket: str = "mnist",
    model_name: str = "mnist",
    model_version: int = 1,
):
    train_and_serve(
        data_dir=data_dir,
        model_dir=model_dir,
        export_bucket=export_bucket,
        model_name=model_name,
        model_version=model_version,
    )
    dsl.get_pipeline_conf().add_op_transformer(op_transformer)
       
