# FL-MNAR
Code repository of the paper "Federated Learning under Sample Selection Heterogeneity" (published in IEEE BigData 2024)

Despite having benefits of privacy preservation and secure computation, federated learning (FL) faces the issue of data heterogeneity. Specifically, the performance of FL systems can be degraded due to sample selection heterogeneity. We define the sample selection heterogeneity scenario with two points. First, each local training set is subject to missing-not-at-random (MNAR) sample selection bias where some labels are non-randomly missing. This requires incorporating a sample selection model that utilizes two equations to account for the prediction and selection of samples. Choosing selection features is a challenging task, especially when the number of observed features is large. Second, the sample selection mechanism is not the same across all clients. This implies that clients do not share the same set of selection features. In this work, we propose FL-MNAR to address FL under sample selection heterogeneity. The framework integrates an existing sample selection model that robustly handles sample selection bias for each client. FL-MNAR also trains an assignment function that gives a set of selection features to each client based on how well the features fit selection. Experimental results show that FL-MNAR achieves state-of-the-art performance under sample selection heterogeneity.

## Usage

### Environment

Create conda environment from the requirements.txt file:
```
conda create --name flowerenv --file requirements.txt
```

### FL-MNAR

Run FL-MNAR using the following command:
```
python3 main.py --dataset [DATASET_NAME] --data_dir [DATASET_DIRECTORY] --scheduler [$T_s$] --alpha [LOCAL_LEARNING_RATE]
```
