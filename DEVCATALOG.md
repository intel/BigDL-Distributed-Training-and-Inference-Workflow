# Distributed Training & Inference Workflows using BigDL 

## Introduction

Developing distributed training and inference pipelines to handle large datasets in production can be a complex and time-consuming process. However, by adopting BigDL, you can seamlessly scale AI workflows written in Python scripts or notebooks on a single laptop across large clusters to process distributed big data.

Learn to use BigDL to easily build and seamlessly scale distributed training and inference workflows
for TensorFlow and PyTorch. This page takes the recsys workflows for Neural Collaborative Filtering (NCF) as an example.

Check out more workflow examples and reference implementations in the [Developer Catalog](https://developer.intel.com/aireferenceimplementations).

## Solution Technical Overview

Highlights and benefits of BigDL are as follows:

- Easily build efficient in-memory, distributed data analytics and AI pipelines that runs on a single Intel® Xeon cluster.
- Seamlessly scale TensorFlow and PyTorch applications to big data platforms.
- Directly deploy solutions on production Hadoop/YARN and Kubernetes clusters.


For more details, visit the BigDL [GitHub repository](https://github.com/intel-analytics/BigDL/tree/main) and
[documentation page](https://bigdl.readthedocs.io/en/latest/).

## Validated Hardware Requirements

BigDL and the workflow example shown below could run widely on Intel® Xeon® series processors.

|| Supported Hardware         |
|---| ---------------------------- |
|CPU| Intel® Xeon® Scalable processors|
|Memory|>10G|
|Disk|>10G|


## How it Works

<img src="https://github.com/intel-analytics/BigDL/blob/main/docs/readthedocs/image/orca-workflow.png" width="80%" />

The architecture above illustrates how BigDL can build end-to-end, distributed and in-memory pipelines on Intel® Xeon clusters.

- BigDL supports loading data from various distributed data sources and data formats that are widely used in the big data ecosystem.
- BigDL supports distributed data processing with Spark DataFrame, Ray Dataset and provides APIs for distributed data parallel processing of Python libraries.
- BigDL supports seamlessly scaling many popular deep learning frameworks and includes runtime optimizations on Xeon.

---

## Get Started

We will use `~/work` as our working directory:

```bash
export WORKSPACE=~/work
```

### 1. Prerequisites

You are highly recommended to use the toolkit under the following system and software settings:
- OS: Linux (including Ubuntu 18.04/20.04 and CentOS 7) or Mac
- Python: 3.7, 3.8, 3.9

### 2. Download the Workflow Repository
Create a working directory for the example workflow of BigDL and clone the [Main
Repository](https://github.com/intel-analytics/BigDL) repository into your working
directory. This step downloads the example scripts in BigDL to demonstrate the workflow.
Follow the steps in the next section to easily install BigDL via Docker or pip.

```
mkdir -p $WORKSPACE && cd $WORKSPACE
git clone https://github.com/intel-analytics/BigDL.git
cd BigDL
```

### 3. Download the Datasets

This workflow uses the [ml-100k dataset](https://grouplens.org/datasets/movielens/100k/) of [MovieLens](https://movielens.org/). 

```
cd python/orca/tutorial/NCF
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
```

## Supported Runtime Environment
The workflow uses Spark DataFrame to process the movielens data and defines the [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031) model in PyTorch. You can execute the reference workflow using the following environments:
* Docker
* Helm 
* Bare metal

### Run Using Docker

Follow these instructions to set up and run our provided Docker image.
For running the training workflow on bare metal, see the [bare metal instructions](#run-using-bare-metal).

#### Set Up Docker Engine

You'll need to install Docker Engine on your development system.
Note that while **Docker Engine** is free to use, **Docker Desktop** may require
you to purchase a license.  See the [Docker Engine Server installation
instructions](https://docs.docker.com/engine/install/#server) for details.

If the Docker image is run on a cloud service, you may also need
credentials to perform training and inference related operations (such as these
for Azure):
- [Set up the Azure Machine Learning Account](https://azure.microsoft.com/en-us/free/machine-learning)
- [Configure the Azure credentials using the Command-Line Interface](https://docs.microsoft.com/en-us/cli/azure/authenticate-azure-cli)
- [Compute targets in Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/concept-compute-target)
- [Virtual Machine Products Available in Your Region](https://azure.microsoft.com/en-us/explore/global-infrastructure/products-by-region/?products=virtual-machines&regions=us-east)

#### Setup Docker Compose
Ensure you have Docker Compose installed on your machine. If you don't have this tool installed, consult the official [Docker Compose installation documentation](https://docs.docker.com/compose/install/linux/#install-the-plugin-manually).

```bash
DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
mkdir -p $DOCKER_CONFIG/cli-plugins
curl -SL https://github.com/docker/compose/releases/download/v2.7.0/docker-compose-linux-x86_64 -o $DOCKER_CONFIG/cli-plugins/docker-compose
chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose
docker compose version
```

#### Set Up Docker Image

Pull the provided Docker image:
```
docker pull intelanalytics/bigdl-orca:latest
```

#### Run Pipeline with Docker Compose 
```bash
docker compose up bigdl-workflow --build
```

Stop the containers created by docker compose and remove them after the completion of the workflow.

```bash
docker compose down
```

#### Run Docker Image in an Interactive Environment

Create the Docker container for BigDL using the ``docker run`` command, as shown below. If your environment requires a proxy to access the Internet, export your
development system's proxy settings to the Docker environment by adding `--env http_proxy=${http_proxy}` when you create the docker container.
```
docker run -a stdout \
  --name bigdl-workflow \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${PWD}:/workspace \
  --workdir /workspace \
  --privileged --init -it \
  intelanalytics/bigdl-orca:latest \
  bash
```

Run these commands to install additional software used for the workflow in the Docker container:
```
pip install torch torchmetrics==0.10.0 tqdm
```

Use these commands to run the workflow:
- Distributed training:
```
python pytorch_train_spark_dataframe.py --dataset ml-100k
```
- Distributed inference:
```
python pytorch_predict_spark_dataframe.py --dataset ml-100k
```

### Run Using Helm
#### 1. Install Helm
- Install [Helm](https://helm.sh/docs/intro/install/)
```bash
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 && \
chmod 700 get_helm.sh && \
./get_helm.sh
```
#### 2. Launch with Helm
```bash
helm upgrade --install bigdl-workflow kubernetes/
```

#### 3. Clean Up the Helm Chart after Completion
```
helm delete bigdl-workflow
```

### Run Using Bare Metal
Follow these instructions to set up and run this workflow on your own development
system. For running the training workflow with a provided Docker image, see the [Docker
 instructions](#31-install-from-docker).


#### Set Up System Software

Our examples use the ``conda`` package and environment on your local computer.
If you don't have ``conda`` installed, see the [Conda Linux installation
instructions](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html).

#### Set Up Workflow

Run these commands to set up the workflow's ``conda`` environment and install required software:
```
conda create -n bigdl python=3.9 --yes
conda activate bigdl
pip install --pre --upgrade bigdl-orca-spark3
pip install torch torchmetrics==0.10.0 tqdm
```

#### Run Workflow
Use these commands to run the workflow:
- Distributed training:
```
python pytorch_train_spark_dataframe.py --dataset ml-100k
```
- Distributed inference:
```
python pytorch_predict_spark_dataframe.py --dataset ml-100k
```

### Expected Output
Check out the processed data, saved model and predictions of the workflow:
```
ll train_processed_dataframe.parquet
ll test_processed_dataframe.parquet
ll test_predictions_dataframe.parquet
ll NCF_model
```
Check out the logs of the console for training and inference results:

- pytorch_train_spark_dataframe.py:
```
Train results:
num_samples: 400052
val_num_samples: 99948
epoch: 1.0
batch_count: 40.0
train_loss: 0.43055336376741554
last_train_loss: 0.3613356734713937
val_accuracy: 0.7996858358383179
val_precision: 1.0
val_recall: 0.00029959555831737816
val_loss: 0.36779773215466566

num_samples: 400052
val_num_samples: 99948
epoch: 2.0
batch_count: 40.0
train_loss: 0.3421447095760922
last_train_loss: 0.2986053046780199
val_accuracy: 0.863288938999176
val_precision: 0.7882054448127747
val_recall: 0.4344634711742401
val_loss: 0.3191395945899022

Evaluation results:
num_samples: 99948
Accuracy: 0.8642994165420532
Precision: 0.7878707647323608
Recall: 0.44436147809028625
val_loss: 0.3183932923312301
```
- pytorch_predict_spark_dataframe.py:
```
Prediction results of the first 5 rows:
+----+----+-----+-------+------+----------+--------------------+--------+-------------------+
|item|user|label|zipcode|gender|occupation|                 age|category|         prediction|
+----+----+-----+-------+------+----------+--------------------+--------+-------------------+
|   1| 585|  0.0|    772|     1|         7|[0.9393939393939394]|     102|-2.0137369632720947|
|   9| 864|  1.0|    529|     1|         6|[0.30303030303030...|       1| 1.2248330116271973|
|  11| 778|  1.0|    117|     1|         1|[0.4090909090909091]|      24|  1.032607078552246|
|  12| 175|  1.0|    280|     2|        10|[0.2878787878787879]|      24| 2.0041821002960205|
|  13| 189|  1.0|    719|     1|        11|[0.3787878787878788]|       2| 0.5711498260498047|
+----+----+-----+-------+------+----------+--------------------+--------+-------------------+
```

---

## Summary and Next Steps
Now you have successfully tried the recsys workflows of BigDL to build an end-to-end pipeline for Neural Collaborative Filtering model.
You can continue to try other use cases provided in BigDL or build the training and inference workflows on your own dataset!

## Learn More
For more information about BigDL distributed training and inference workflows or to read about other relevant workflow
examples, see these guides and software resources:

- More BigDL workflow examples for TensorFlow: https://github.com/intel-analytics/BigDL/tree/main/python/orca/example/learn/tf2
- More BigDL workflow examples for PyTorch: https://github.com/intel-analytics/BigDL/tree/main/python/orca/example/learn/pytorch
- To scale BigDL workflows to Kubernetes clusters: https://bigdl.readthedocs.io/en/latest/doc/Orca/Tutorial/k8s.html
- [Intel® AI Analytics Toolkit (AI Kit)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html)
- [Azure Machine Learning Documentation](https://learn.microsoft.com/en-us/azure/machine-learning/)

## Troubleshooting
- If you encounter the error `E0129 21:36:55.796060683 1934066 thread_pool.cc:254] Waiting for thread pool to idle before forking` during the training, it may be caused by the installed version of grpc. See [here](https://github.com/grpc/grpc/pull/32196) for more details about this issue. To fix it, a recommended grpc version is 1.43.0:
```bash
pip install grpcio==1.43.0
```

## Support
If you have questions or issues about this workflow, contact the Support Team through [GitHub](https://github.com/intel-analytics/BigDL/issues) or [Google User Group](https://groups.google.com/g/bigdl-user-group).
