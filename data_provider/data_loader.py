import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from sklearn import model_selection
import time

warnings.filterwarnings('ignore')


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, args, all_df, labels_df, class_names, max_seq_len, limit_size=None):
        self.args = args
        self.root_path = args.data_path
        self.class_names = class_names
        self.max_seq_len = max_seq_len
        self.all_df, self.labels_df = all_df, labels_df
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        # print(len(self.all_IDs))

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        # if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
        #     num_samples = len(self.all_IDs)
        #     num_columns = self.feature_df.shape[1]
        #     seq_len = int(self.feature_df.shape[0] / num_samples)
        #     batch_x = batch_x.reshape((1, seq_len, num_columns))
        #     batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

        #     batch_x = batch_x.reshape((1 * seq_len, num_columns))

        return self.instance_norm(torch.from_numpy(batch_x)), \
               torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)

def load_uea_dataset(args, root_path, file_list=None):
    Data = {}
    time_now = time.time()

    all_train_data_df, all_train_labels_df, all_train_class_names, all_train_max_seq_len = load_all(root_path, file_list=file_list, flag='TRAIN')
    test_data_df, test_labels_df, test_class_names, test_max_seq_len = load_all(root_path, file_list=file_list, flag='TEST')

    # splitting the dataset
    if args.val_ratio > 0:
        train_data_df, train_label_df, val_data_df, val_label_df = split_dataset(all_train_data_df, all_train_labels_df, args.val_ratio, args.seed)
    # if val ratio = 0, the validation set is equal to the test set
    else:
        train_data_df, train_label_df = all_train_data_df, all_train_labels_df
        val_data_df, val_label_df     = test_data_df, test_labels_df

    Data['All_train_data']   = all_train_data_df
    Data['All_train_labels'] = all_train_labels_df
    Data['train_data']       = train_data_df
    Data['train_labels']     = train_label_df
    Data['val_data']         = val_data_df
    Data['val_labels']       = val_label_df
    Data['test_data']        = test_data_df
    Data['test_labels']      = test_labels_df

    Data['All_train_max_len'] = Data['train_max_len'] =  Data['val_max_len'] = all_train_max_seq_len
    Data['test_max_len']      = test_max_seq_len
    Data['All_train_class_names'] = Data['train_class_names'] = Data['val_class_names'] = all_train_class_names
    Data['test_class_names']      = test_class_names

    print("{} {} {} samples will be used for training, validation, and testing, respectively.".format(len(Data['train_labels']), len(Data['val_labels']), len(Data['test_labels'])))

    print('load data time :', time.time() - time_now)

    return Data
    
def load_all(root_path, file_list=None, flag=None):
    """
    Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
    Args:
        root_path: directory containing all individual .csv files
        file_list: optionally, provide a list of file paths within `root_path` to consider.
            Otherwise, entire `root_path` contents will be used.
    Returns:
        all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
        labels_df: dataframe containing label(s) for each sample
    """
    # Select paths for training and evaluation
    if file_list is None:
        data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
    else:
        data_paths = [os.path.join(root_path, p) for p in file_list]
    if len(data_paths) == 0:
        raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
    if flag is not None:
        data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
    input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
    if len(input_paths) == 0:
        pattern='*.ts'
        raise Exception("No .ts files found using pattern: '{}'".format(pattern))

    all_df, labels_df, class_names, max_seq_len = load_single(input_paths[0])  # a single file contains dataset

    return all_df, labels_df, class_names, max_seq_len

def load_single(filepath):
    df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                            replace_missing_vals_with='NaN')
    labels = pd.Series(labels, dtype="category")
    class_names = labels.cat.categories
    labels_df = pd.DataFrame(labels.cat.codes,
                                dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

    lengths = df.applymap(
        lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

    horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

    if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
        df_trimmed = df.apply(lambda row: trim_series(row), axis=1)
        df = pd.DataFrame(df_trimmed.tolist(), columns=df.columns)
        # df = df.applymap(subsample)

    max_seq_len = int(np.max(lengths[:, 0]))
    if max_seq_len > 2500:  # mainly for dataset of EigenWorms
        df = df.applymap(lambda x: subsample(x, limit=0, factor=(max_seq_len // 1500)))

    lengths = df.applymap(lambda x: len(x)).values
    vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
    if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
        max_seq_len = int(np.max(lengths[:, 0]))
    else:
        max_seq_len = lengths[0, 0]

    # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
    # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
    # sample index (i.e. the same scheme as all datasets in this project)

    df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
        pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

    # Replace NaN values
    grp = df.groupby(by=df.index)
    df = grp.transform(interpolate_missing)

    return df, labels_df, class_names, max_seq_len

def split_dataset(data_df, label_df, validation_ratio, seed):
    splitter = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=seed)
    train_indices, val_indices = zip(*splitter.split(X=np.zeros(len(label_df)), y=label_df))
    
    train_data_df = data_df.loc[train_indices]
    train_label_df = label_df.loc[train_indices]
    val_data_df = data_df.loc[val_indices]
    val_label_df = label_df.loc[val_indices]
    return train_data_df, train_label_df, val_data_df, val_label_df

# define a function to trim the Series
def trim_series(series_list):
    #  find the minimum Series length in this row
    min_len = min([len(s) for s in series_list])
    # trim each Series to the minimum length
    return [s.iloc[:min_len] for s in series_list]