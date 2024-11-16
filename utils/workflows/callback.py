import copy

import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl


class Counter(pl.Callback):
    def __init__(self, what="epochs", verbose=True):
        self.what = what
        self.verbose = verbose
        self.state = {"epochs": 0, "batches": 0}

    @property
    def state_key(self) -> str:
        # note: we do not include `verbose` here on purpose
        return f"Counter[what={self.what}]"

    def on_train_epoch_end(self, *args, **kwargs):
        if self.what == "epochs":
            self.state["epochs"] += 1

    def on_train_batch_end(self, *args, **kwargs):
        if self.what == "batches":
            self.state["batches"] += 1

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)

    def state_dict(self):
        return self.state.copy()


class OnlineDACallback(pl.Callback):

    def __init__(self, generator):
        """Initialize online data augmentation callback.

        Parameters
        ----------
        generator : BaseTimeSeriesGenerator
            Time series generator object for data augmentation.
            Must be one of:
            - PureSyntheticGenerator: Creates fully synthetic series without reference
            to a source dataset
            - SemiSyntheticGenerator: Modifies existing series from a source dataset
            - SemiSyntheticTransformer: Applies transformations to series from
            a source dataset

        """
        super().__init__()

        self.generator = generator

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """
        Applying data augmentation after getting a batch of time series for training
        """
        temporal = batch['temporal']
        temporal, _ = torch.sort(temporal, dim=0)

        df_ = self.temporal_to_df(temporal)
        df_synth = self.generator.transform(df_)
        df_aug = pd.concat([df_, df_synth])
        temporal_aug = self.df_to_tensor(df_aug)

        if isinstance(temporal, torch.mps.Tensor):
            temporal_aug = temporal_aug.to('mps')

        batch['temporal'] = temporal_aug

        return batch

    @staticmethod
    def temporal_to_df(temporal: torch.Tensor) -> pd.DataFrame:
        """Convert batch of time series tensors to DataFrame format.

        Transforms a tensor containing multiple time series into a long-format
        DataFrame following the (unique_id, ds, y) structure used in
        Nixtla-based forecasting workflows.

        Parameters
        ----------
        temporal : torch.Tensor
            Batch of time series data.

        Returns
        -------
        pd.DataFrame
            Long-format time series data with columns:
            - unique_id: Series identifier (0 to batch_size-1)
            - ds: Integer index as time indicator
            - y: Series values

        """
        temporal_ = copy.deepcopy(temporal)

        arr_list = []
        for i, arr in enumerate(temporal_):
            if isinstance(arr, torch.mps.Tensor):
                arr = arr.cpu()

            arr_t = arr.cpu().numpy().T

            arr_df = pd.DataFrame(arr_t).copy()
            arr_df.columns = ['y', 'y_mask']
            arr_df['ds'] = np.arange(arr_df.shape[0])
            arr_df['unique_id'] = f'ID{i}'

            arr_list.append(arr_df)

        df = pd.concat(arr_list)
        df = df.query('y_mask>0').drop(columns=['y_mask']).reset_index(drop=True)

        return df

    @staticmethod
    def create_mask(df) -> pd.DataFrame:
        """ Create masked time series datasets

        Transforms variable-length time series into fixed-length format by:
        1. Finding maximum series length
        2. Left-padding shorter series with zeros
        3. Creating binary mask to identify padded values

        Parameters
        ----------
        df : pd.DataFrame
            Time series dataset with required columns:
            - unique_id: Series identifier
            - ds: Timestamp column
            - y: Target values

        Returns
        -------
        pd.DataFrame
            Padded dataset with added columns:
            - y_mask: Boolean, False for padded values
            - All original columns preserved
            All series padded to same length

        """
        uids = df['unique_id'].unique()
        all_ds = np.arange(0, df['ds'].max() + 1)

        df_extended = pd.DataFrame([(uid, ds)
                                    for uid in uids
                                    for ds in all_ds],
                                   columns=['unique_id', 'ds'])

        result = pd.merge(df_extended, df, on=['unique_id', 'ds'], how='left')

        result['y_mask'] = (~result['y'].isna()).astype(int)
        result['y'] = result['y'].fillna(0)

        result = result.sort_values(['unique_id', 'ds'])
        result = result.reset_index(drop=True)

        return result

    @classmethod
    def df_to_tensor(cls, df: pd.DataFrame) -> torch.Tensor:
        """Convert DataFrame of time series data to PyTorch tensor.

        Transforms a time series DataFrame following neuralforecast's format
        into a PyTorch tensor for model training. Handles masked values and
        multiple series.

        Parameters
        ----------
        df : pd.DataFrame
            Time series dataset with required columns:
            - unique_id: Series identifier
            - ds: Timestamp
            - y: Target values

        """

        df_ = cls.create_mask(df).copy()

        arr_list = []
        for _, uid_df in df_.groupby('unique_id'):
            arr_list.append(uid_df[['y', 'y_mask']].values.T)

        arr = np.stack(arr_list, axis=0)

        t = torch.tensor(arr, dtype=torch.float32)

        return t
