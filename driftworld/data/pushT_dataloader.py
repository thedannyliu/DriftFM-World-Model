import torch
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import zarr
from utils import TrainDataset, FullVideoDataset, list_zarr_paths


def get_pushT_loader(cfg, rank=0, world_size=1):
    """
    Returns dataloader for Push-T dataset.
    When world_size > 1, a DistributedSampler shards the dataset across ranks;
    otherwise plain shuffling is used. batch_size is the PER-GPU batch size.
    """
    dataset_list = []
    combined_stats = []
    num_datasets = 0
    all_data_stats = {'agent_pos': {'min': np.array([2.0407837e-04, 1.0189312e+00], dtype=np.float32), 'max': np.array([509.08173, 509.43417], dtype=np.float32)}, 'action': {'min': np.array([0., 0.], dtype=np.float32), 'max': np.array([511., 511.], dtype=np.float32)}}

    train_demos = cfg.data.num_train_demos
    if cfg.validation.enabled:
        train_demos -= cfg.validation.num_demos
        if train_demos < 1:
            raise ValueError("Validation holdout leaves no training demos")
        if cfg.validation.demo_start != train_demos:
            raise ValueError(
                "validation.demo_start must equal the number of retained training demos"
            )

    for full_path in list_zarr_paths(cfg.data.dataset_path_dir):
        # create dataset from file
        dataset = TrainDataset(
            dataset_path=full_path,
            pred_horizon=cfg.data.pred_horizon,
            obs_horizon=cfg.data.obs_horizon,
            action_horizon=cfg.data.action_horizon,
            id = num_datasets,
            num_demos = train_demos,
            resize_scale = cfg.data.resize_scale,
            stats = all_data_stats
        )
        num_datasets += 1
        # save training data statistics (min, max) for each dim
        stats = dataset.stats
        dataset_list.append(dataset)
        combined_stats.append(stats)

    combined_dataset = ConcatDataset(dataset_list)

    loader_generator = torch.Generator().manual_seed(cfg.train.seed)
    if world_size > 1:
        sampler = DistributedSampler(
            combined_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        dataloader = torch.utils.data.DataLoader(
            combined_dataset,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.dataloader.num_workers,
            prefetch_factor=cfg.dataloader.prefetch_factor,
            sampler=sampler,
            pin_memory=cfg.dataloader.pin_memory,
            persistent_workers=True,
            generator=loader_generator,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            combined_dataset,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.dataloader.num_workers,
            prefetch_factor=cfg.dataloader.prefetch_factor,
            shuffle=True,
            pin_memory=cfg.dataloader.pin_memory,
            persistent_workers=True,
            generator=loader_generator,
        )
    return dataloader


def get_pushT_validation_loader(cfg):
    """Return the fixed held-out loader used only for checkpoint selection."""
    dataset_list = []
    all_data_stats = {'agent_pos': {'min': np.array([2.0407837e-04, 1.0189312e+00], dtype=np.float32), 'max': np.array([509.08173, 509.43417], dtype=np.float32)}, 'action': {'min': np.array([0., 0.], dtype=np.float32), 'max': np.array([511., 511.], dtype=np.float32)}}

    for dataset_id, full_path in enumerate(list_zarr_paths(cfg.data.dataset_path_dir)):
        total_demos = zarr.open(full_path, mode='r')['meta']['episode_ends'].shape[0]
        if total_demos < cfg.validation.demo_start + cfg.validation.num_demos:
            continue
        dataset_list.append(TrainDataset(
            dataset_path=full_path,
            pred_horizon=cfg.data.pred_horizon,
            obs_horizon=cfg.data.obs_horizon,
            action_horizon=cfg.data.action_horizon,
            id=dataset_id,
            num_demos=cfg.validation.num_demos,
            resize_scale=cfg.data.resize_scale,
            stats=all_data_stats,
            demo_start=cfg.validation.demo_start,
            require_num_demos=True,
        ))

    if not dataset_list:
        raise ValueError("No dataset contains the requested validation episodes")

    combined_dataset = ConcatDataset(dataset_list)
    num_workers = cfg.validation.num_workers
    loader_kwargs = {
        'dataset': combined_dataset,
        'batch_size': cfg.validation.batch_size,
        'num_workers': num_workers,
        'shuffle': False,
        'pin_memory': cfg.dataloader.pin_memory,
        'persistent_workers': num_workers > 0,
        'generator': torch.Generator().manual_seed(cfg.validation.seed),
    }
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = cfg.dataloader.prefetch_factor
    return torch.utils.data.DataLoader(**loader_kwargs)

def get_pushT_loader_shuffleFalse(cfg):
    """
    Returns dataloader for Push-T dataset with shuffle=False.
    """
    dataset_list = []
    num_datasets = 0
    all_data_stats = {'agent_pos': {'min': np.array([2.0407837e-04, 1.0189312e+00], dtype=np.float32), 'max': np.array([509.08173, 509.43417], dtype=np.float32)}, 'action': {'min': np.array([0., 0.], dtype=np.float32), 'max': np.array([511., 511.], dtype=np.float32)}}

    for full_path in list_zarr_paths(cfg.data.dataset_path_dir):
        dataset = TrainDataset(
            dataset_path=full_path,
            pred_horizon=cfg.data.pred_horizon,
            obs_horizon=cfg.data.obs_horizon,
            action_horizon=cfg.data.action_horizon,
            id = num_datasets,
            num_demos = cfg.data.num_train_demos,
            resize_scale = cfg.data.resize_scale,
            stats = all_data_stats
        )
        num_datasets += 1
        dataset_list.append(dataset)

    combined_dataset = ConcatDataset(dataset_list)

    dataloader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.dataloader.num_workers,
        prefetch_factor=cfg.dataloader.prefetch_factor,
        shuffle=False,
        pin_memory=cfg.dataloader.pin_memory,
        persistent_workers=True
    )
    return dataloader

def get_pushT_full_loader(cfg):
    """
    Returns a dataloader that yields full-length episodes for Push-T.
    Uses batch_size=1 and shuffle=False.
    """
    dataset_list = []
    num_datasets = 0
    all_data_stats = {'agent_pos': {'min': np.array([2.0407837e-04, 1.0189312e+00], dtype=np.float32), 'max': np.array([509.08173, 509.43417], dtype=np.float32)}, 'action': {'min': np.array([0., 0.], dtype=np.float32), 'max': np.array([511., 511.], dtype=np.float32)}}

    for full_path in list_zarr_paths(cfg.data.dataset_path_dir):
        dataset = FullVideoDataset(
            dataset_path=full_path,
            id=num_datasets,
            num_demos=cfg.data.num_train_demos,
            resize_scale=cfg.data.resize_scale,
            stats=all_data_stats
        )
        num_datasets += 1
        dataset_list.append(dataset)

    combined_dataset = ConcatDataset(dataset_list)

    dataloader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=1,
        num_workers=cfg.dataloader.num_workers,
        prefetch_factor=cfg.dataloader.prefetch_factor,
        shuffle=False,
        pin_memory=cfg.dataloader.pin_memory,
        persistent_workers=True
    )
    return dataloader
