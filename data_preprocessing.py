import torch
import operator
from torch.utils.data import random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
import torch.nn.functional as F
from BiasCorr.BiasCorr_preprocess import g_s_train, g_y_train

def get_adult(data_dir, split, device):
    # Prepare training and testing sets
    features = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",
                "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
                "Hours per week", "Country", "Target"]
    original_train = pd.read_csv(data_dir + 'adult.data', names=features, sep=r'\s*,\s*',
                                 engine='python', na_values="?")
    original_test = pd.read_csv(data_dir + 'adult.test', names=features, sep=r'\s*,\s*',
                                engine='python', na_values="?")

    original = pd.concat([original_train, original_test], axis=0)
    original = original.dropna().reset_index(drop=True)
    original['Sex'] = original['Sex'].map({'Male': 1, 'Female': 0})
    original['Target'] = original['Target'].replace('<=50K', 0).replace('>50K', 1)
    original['Target'] = original['Target'].replace('<=50K.', 0).replace('>50K.', 1)
    original.Age = original.Age.astype(int)

    # Redundant column
    del original["Education"]

    # Drop some more columns
    del original["fnlwgt"]

    # Adjust martial status column
    original.replace(['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married',
                      'Separated', 'Widowed'],
                     ['not married', 'married', 'married', 'married', 'not married', 'not married', 'not married'],
                     inplace=True)

    original['Marital Status'] = original['Marital Status'].map({'married': 1, 'not married': 0})

    original.replace(['Federal-gov', 'Local-gov', 'State-gov'], ['government', 'government', 'government'],
                     inplace=True)
    original.replace(['Never-worked', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'Without-pay'],
                     ['private', 'private', 'private', 'private', 'private'], inplace=True)

    original['Workclass'] = original['Workclass'].map({'government': 1, 'private': 0})

    # Adjust country column
    counts = original['Country'].value_counts()
    replace = counts[counts <= 150].index
    original['Country'] = original['Country'].replace(replace, 'other')

    del original["Race"]

    def data_transform(df):
        """Normalize features."""
        binary_data = pd.get_dummies(df)
        feature_cols = binary_data
        data = pd.DataFrame(feature_cols, columns=feature_cols.columns)
        return data

    data = data_transform(original)

    sc = StandardScaler()
    X_total_adult = pd.concat([data.iloc[:, 0:1], data.iloc[:, 2:4], data.iloc[:, 5:]], axis=1).values
    X_total_adult = sc.fit_transform(X_total_adult)

    training_set = np.concatenate([
        X_total_adult[:int(len(data) * split), :], data.iloc[:int(len(data) * split), 1].values.reshape(-1, 1),
        np.zeros((int(len(data) * split), 1))
    ], axis=1)
    testing_set = np.concatenate([
        X_total_adult[int(len(data) * split):, :], data.iloc[int(len(data) * split):, 1].values.reshape(-1, 1),
        np.zeros((len(data) - int(len(data) * split), 1))
    ], axis=1)

    return Variable(torch.from_numpy(training_set)).type(torch.FloatTensor).to(device), Variable(
        torch.from_numpy(testing_set)).type(torch.FloatTensor).to(device), data

def get_drug(split, device):
    columns = ["ID", "Age", "Gender", "Education", "Country", "Ethnicity",
               "Nscore", "Escore", "Oscore", "Ascore", "Cscore", "Impulsive", "SS",
               "Alcohol", "Amphet", "Amyl", "Benzos", "Caff", "Cannabis", "Choc", "Coke", "Crack", "Ecstasy",
               "Heroin", "Ketamine", "Legalh", "LSD", "Meth", "Mushrooms", "Nicotine", "Semer", "VSA"]
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data',
                       sep=",", names=columns)
    drug = "Benzos"
    data.loc[(data[drug] == 'CL0') | (data[drug] == 'CL1'), "Drug"] = 0
    data.loc[(data[drug] != 'CL0') & (data[drug] != 'CL1'), "Drug"] = 1

    sc = StandardScaler()
    X_total_drug = data.iloc[:, 1:13].values
    X_total_drug = sc.fit_transform(X_total_drug)

    training_set = np.concatenate([
        X_total_drug[:int(len(data) * split), [1, 2, 3, 4, 6, 0, 5, 7, 8, 9, 10, 11]],
        data.iloc[:int(len(data) * split), 32].values.reshape(-1, 1),
        np.zeros((int(len(data) * split), 1))
    ], axis=1)
    testing_set = np.concatenate([
        X_total_drug[int(len(data) * split):, [1, 2, 3, 4, 6, 0, 5, 7, 8, 9, 10, 11]],
        data.iloc[int(len(data) * split):, 32].values.reshape(-1, 1),
        np.zeros((len(data) - int(len(data) * split), 1))
    ], axis=1)

    return Variable(torch.from_numpy(training_set)).type(torch.FloatTensor).to(device), Variable(
        torch.from_numpy(testing_set)).type(torch.FloatTensor).to(device), data

def assignment_computation(X_train, y, s, J):
    # Establish pi based on the number of prediction features (J)
    psi_params1 = torch.cat((
        torch.ones((1, J)) * np.log(0.001),
        torch.ones((1, J)) * np.log(0.999)
    ))
    psi_params2 = torch.cat((
        torch.ones((1, X_train.shape[1] - J)) * np.log(0.5),
        torch.ones((1, X_train.shape[1] - J)) * np.log(0.5)
    ))
    pi = Variable(
        torch.cat((
            psi_params1, psi_params2
        ), 1), requires_grad=True)

    M = F.gumbel_softmax(logits=pi, tau=1, hard=True, dim=0)
    m = torch.argmax(M, dim=0).reshape(1, -1)
    assignment = np.where(m.reshape(-1).cpu().detach().numpy() == 1)[0]

    # modified_training_set = torch.cat((
    #     X_train[:, assignment], y.reshape(-1, 1), s.reshape(-1, 1)
    # ), 1)

    return assignment

def prepare_modified_data(dataset, data_dir, num_partitions, batch_size, J, feature_assign=False):

    if dataset == "Adult":
        # Get Adult dataset
        training_set, testing_set, original = get_adult(
            data_dir=data_dir,
            split=0.7,
            device='cpu'
        )
    elif dataset == "Drug":
        # Get Drug dataset
        training_set, testing_set, original = get_drug(
            split=0.7,
            device='cpu'
        )

    # Split training set into 'num_partitions' local training sets
    num_samples = len(training_set) // num_partitions
    remainder = len(training_set) - ( num_samples * (num_partitions - 1) )
    partition_len = [num_samples] * (num_partitions - 1)
    partition_len.append(remainder)
    # training_sets = random_split(training_set, partition_len, torch.Generator().manual_seed(2023))
    # training_sets = [training_sets[i].dataset for i in range(num_partitions)]

    # Manually split training set into 'num_partitions' local training sets (non-IID)
    train_original = original.iloc[:int(len(original) * 0.7)]
    if dataset == "Adult":
        bias_categories = [
            'Education-Num', 'Age', 'Marital Status', 'Hours per week',
            'Education-Num', 'Age', 'Marital Status', 'Hours per week',
            'Education-Num', 'Age'
        ]
        bias_values = [
            12, 32, 1, 40,
            12, 32, 1, 40,
            12, 32
        ]
        bias_operators = [
            operator.gt, operator.ge, operator.eq, operator.ge,
            operator.gt, operator.ge, operator.eq, operator.ge,
            operator.gt, operator.ge
        ]
        not_bias_operators = [
            operator.le, operator.lt, operator.ne, operator.lt,
            operator.le, operator.lt, operator.ne, operator.lt,
            operator.le, operator.lt
        ]
    elif dataset == "Drug":
        bias_categories = [
            'Oscore', 'Escore', 'Oscore', 'Escore'
        ]
        bias_values = [
            -0.35, -0.05, -0.35, -0.35
        ]
        bias_operators = [
            operator.lt, operator.lt, operator.lt, operator.lt
        ]
        not_bias_operators = [
            operator.ge, operator.ge, operator.ge, operator.ge
        ]

    # Third setting: Biased local dataset with selection value
    full_local_sets = []
    val_sets = []
    low = 0

    for i in range(len(partition_len)):
        high = low + partition_len[i]
        partition_original = train_original.iloc[low:high]
        select = partition_original.loc[
            bias_operators[i](partition_original[bias_categories[i]], bias_values[i])
        ].index.values

        not_select = partition_original.loc[
            not_bias_operators[i](partition_original[bias_categories[i]], bias_values[i])
        ].index.values

        # Set pseudolabel
        # # Define function that computes pseudolabel after training g_y on D^{(c)}_s
        if dataset == "Adult":
            pseudolabels = g_y_train(
                X_train_s=training_set[select, :J], X_train_u=training_set[not_select, :J],
                train_y_s=training_set[select, -2], spec='LR', tol=0.01, lr=0.01
            )
        elif dataset == "Drug":
            pseudolabels = g_y_train(
                X_train_s=training_set[select, :J], X_train_u=training_set[not_select, :J],
                train_y_s=training_set[select, -2], spec='LR', tol=0.01, lr=0.005
            )
        training_set[not_select, -2] = torch.Tensor(pseudolabels)

        # Set s'_i
        training_set[select, -1] = 1

        if feature_assign:
            if dataset == "Adult":
                assignment = assignment_computation(
                    X_train=training_set[low:high, :-2], y=training_set[low:high, -2],
                    s=training_set[low:high, -1], J=J
                )
                X_train_sel_u = training_set[not_select]
                sel_var = g_s_train(
                    X_train_sel=training_set[low:high, assignment], X_train_sel_u=X_train_sel_u[:, assignment],
                    s=training_set[low:high, -1], spec='LR', tol=0.1, lr=0.01
                )
            elif dataset == "Drug":
                assignment = assignment_computation(
                    X_train=training_set[low:high, :-2], y=training_set[low:high, -2],
                    s=training_set[low:high, -1], J=J
                )
                X_train_sel_u = training_set[not_select]
                sel_var = g_s_train(
                    X_train_sel=training_set[low:high, assignment], X_train_sel_u=X_train_sel_u[:, assignment],
                    s=training_set[low:high, -1], spec='Probit', tol=0.005, lr=0.01
                )
            training_set[not_select, -1] = torch.Tensor([sel_var])
            full_local_set = torch.cat((
                training_set[low:high, assignment],
                training_set[low:high, -2].reshape(-1, 1),
                training_set[low:high, -1].reshape(-1, 1)
            ), 1)
        else:
            full_local_set = training_set[low:high, :]

        full_local_sets.append(full_local_set)
        val_sets.append(training_set[select])
        low = low + partition_len[i]

    return full_local_sets, val_sets, testing_set