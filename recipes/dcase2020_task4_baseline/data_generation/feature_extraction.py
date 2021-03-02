# File containing the organization for the data generation (ETL process):
# E: extraction process
# T: transformation process (normalization)
# L: Loading of the dataset

import inspect
import logging
import os


from utils.Logger import create_logger
from utils.Scaler import Scaler, ScalerPerAudio
from utils.Transforms import get_transforms
from utils_data.DataLoad import ConcatDataset, DataLoadDf
from utils_data.Desed import DESED


# Extraction of datasets
def get_dfs(
    path_dict,
    desed_dataset,
    sample_rate,
    hop_size,
    pooling_time_ratio,
    save_features,
    nb_files=None,
    reduced_dataset=False,
    eval_dataset=False,
    separated_sources=False,
):
    """
    The function initializes and retrieves all the subset of the dataset.

    Args:
        desed_dataset: desed class instance
        sample_rate: int, sample rate
        hop_size: int, window hop size
        pooling_time_ratio: int, pooling time ratio
        save_features: bool, True if features need to be saved, False if features are not going to be saved
        nb_files: int, number of file to be considered (in case you want to consider only part of the dataset)
        separated source: bool, true if you want to consider separated source as well or not
    """

    log = create_logger(
        __name__ + "/" + inspect.currentframe().f_code.co_name,
        terminal_level=logging.INFO,
    )


    train_synth_df = desed_dataset.initialize_and_get_df(
        tsv_path=path_dict["tsv_path_train_synth"],
        audio_dir=path_dict["audio_train_synth"],
        nb_files=2500 if reduced_dataset else nb_files,
        download=False,
        save_features=save_features,
    )

    valid_synth_df = desed_dataset.initialize_and_get_df(
        tsv_path=path_dict["tsv_path_valid_synth"],
        audio_dir=path_dict["audio_valid_synth"],
        nb_files=1000 if reduced_dataset else nb_files,
        download=False,
        save_features=save_features,
    )

    if eval_dataset:
        validation_df = desed_dataset.initialize_and_get_df(
            tsv_path=path_dict["tsv_path_eval_deded"],
            audio_dir=path_dict["audio_evaluation_dir"],
            nb_files=nb_files,
            save_features=save_features,
        )
    else:
        validation_df = desed_dataset.initialize_and_get_df(
            tsv_path=path_dict["tsv_path_valid"],
            audio_dir=path_dict["audio_validation_dir"],
            nb_files=nb_files,
            save_features=save_features,
        )

    # Put train_synth in frames so many_hot_encoder can work.
    # Not doing it for valid, because not using labels (when prediction) and event based metric expect sec.
    
    # ipdb.set_trace()
    # train_synth_df.onset = (
    #      train_synth_df.onset * sample_rate // hop_size // pooling_time_ratio
    # )
    # train_synth_df.offset = (
    #     train_synth_df.offset * sample_rate // hop_size // pooling_time_ratio
    # )
    
    # log.debug(valid_synth_df.event_label.value_counts())

    data_dfs = {
        "train_synthetic": train_synth_df,
        "valid_synthetic": valid_synth_df,
        "validation": validation_df,  
    }

    return data_dfs


def get_dataset(
    base_feature_dir,
    path_dict,
    sample_rate,
    n_window,
    hop_size,
    n_mels,
    mel_min_max_freq,
    pooling_time_ratio,
    save_features,
    eval_dataset=False,
    reduced_dataset=False,
    nb_files=None,
):
    """
        Function to get the dataset

    Args:
        base_feature_dir: features directory
        path_dict: dict, dictionary containing all the necessary paths
        sample_rate: int, sample rate
        n_window: int, window length
        hop_size: int, hop size
        n_mels: int, number of mels
        mel_min_max_freq: tuple, min and max frequency to consider for the mel filter
        nb_files: int, number of files to retrieve and process (in case only part of dataset is used)

    Return:
        desed_dataset: DESED instance
        dfs: dict, dictionary containing the different subset of the datasets.

    """
    desed_dataset = DESED(
        sample_rate=sample_rate,
        n_window=n_window,
        hop_size=hop_size,
        n_mels=n_mels,
        mel_min_max_freq=mel_min_max_freq,
        base_feature_dir=base_feature_dir,
        compute_log=False,
    )

    
    dfs = get_dfs(
        path_dict=path_dict,
        sample_rate=sample_rate,
        hop_size=hop_size,
        pooling_time_ratio=pooling_time_ratio,
        desed_dataset=desed_dataset,
        save_features=save_features,
        reduced_dataset=reduced_dataset,
        nb_files=nb_files,
        eval_dataset=eval_dataset,
        separated_sources=False,
    )
    return desed_dataset, dfs


def get_compose_transforms(
    datasets,
    scaler_type,
    max_frames,
    add_axis_conv,
    noise_snr,
    encode_label_kwargs,
    ext,
):
    """
    The function performs all the operation needed to normalize the dataset.

    Args:
        dfs: dict, dataset
        encod_funct: encode labels function
        scaler_type: str, which type of scaler to consider, per audio or the full dataset
        max_frames: int, maximum number of frames
        add_axis_conv: int, axis to squeeze


    Return:
        transforms: transforms to apply to training dataset
        transforms_valid: transforms to apply to validation dataset

    """

    log = create_logger(
        __name__ + "/" + inspect.currentframe().f_code.co_name,
        terminal_level=logging.INFO,
    )

    if scaler_type == "dataset":
        transforms = get_transforms(
            encode_label_kwargs=encode_label_kwargs,
            frames=max_frames,
            add_axis=add_axis_conv,
        )

        train_synth_data = datasets["synthetic"]
        train_synth_data.transforms = transforms

        # scaling, only on real data since that's our final goal and test data are real
        scaler_args = []
        scaler = Scaler()
        scaler.calculate_scaler(ConcatDataset([train_synth_data]), ext)
    else:
        scaler_args = ["global", "min-max"]
        scaler = ScalerPerAudio(*scaler_args)

    transforms = get_transforms(
        encode_label_kwargs=encode_label_kwargs,
        frames=max_frames,
        scaler=scaler,
        add_axis=add_axis_conv,
        noise_dict_params={"mean": 0.0, "snr": noise_snr},
    )

    transforms_valid = get_transforms(
        encode_label_kwargs=encode_label_kwargs,
        frames=max_frames,
        scaler=scaler,
        add_axis=add_axis_conv,
    )

    return transforms, transforms_valid, scaler, scaler_args
