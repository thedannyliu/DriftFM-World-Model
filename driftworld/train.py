"""
Training loop for DriftWorld on Push-T
"""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.set_float32_matmul_precision('high')
import numpy as np
import wandb
import logging
import random
import json
import shutil

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from data.pushT_dataloader import get_pushT_loader, get_pushT_validation_loader
from utils_model import create_model

log = logging.getLogger(__name__)

def ddp_setup():
    """
    Initialize torch.distributed from env vars set by torchrun.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def is_main():
    return int(os.environ.get("RANK", 0)) == 0


def barrier(world_size):
    if world_size > 1 and dist.is_initialized():
        dist.barrier()


def set_seed(seed, rank=0):
    seed = seed + rank
    if is_main():
        log.info(f"Seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return seed


def rng_state_dict():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def load_rng_state_dict(state):
    if not state:
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])


def gather_rng_states(world_size):
    local_state = rng_state_dict()
    if world_size == 1:
        return [local_state]
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_state)
    return gathered


def load_initial_model(model, state_dict):
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing = all('time_embed.' in key for key in incompatible.missing_keys)
    if incompatible.unexpected_keys or not allowed_missing:
        raise RuntimeError(
            f"Incompatible initial checkpoint: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return incompatible.missing_keys


def save_checkpoint_atomic(checkpoint, latest_path):
    temporary_path = f"{latest_path}.tmp"
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, latest_path)


def copy_checkpoint_atomic(source_path, destination_path):
    temporary_path = f"{destination_path}.tmp"
    shutil.copyfile(source_path, temporary_path)
    os.replace(temporary_path, destination_path)


@torch.no_grad()
def evaluate_validation(model, dataloader, cfg, device):
    """Evaluate a fixed stochastic objective without advancing training RNG."""
    training_rng_state = rng_state_dict()
    was_training = model.training
    random.seed(cfg.validation.seed)
    np.random.seed(cfg.validation.seed)
    torch.manual_seed(cfg.validation.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.validation.seed)

    totals = {}
    num_batches = 0
    model.eval()
    try:
        for nbatch in dataloader:
            if num_batches >= cfg.validation.max_batches:
                break
            if cfg.data.normalize_img:
                nbatch['image'] = (nbatch['image'] - 0.5) / 0.5
            loss, metrics = model(nbatch, device)
            metrics = dict(metrics)
            metrics['loss'] = loss.item()
            for key, value in metrics.items():
                if torch.is_tensor(value):
                    value = value.item()
                totals[key] = totals.get(key, 0.0) + float(value)
            num_batches += 1
    finally:
        model.train(was_training)
        load_rng_state_dict(training_rng_state)

    if num_batches == 0:
        raise RuntimeError("Validation loader produced no batches")
    return {f"validation/{key}": value / num_batches for key, value in totals.items()}

def train(cfg):
    """
    Train model, given hydra config cfg. Multi-GPU via torchrun + DDP.
    """
    local_rank, rank, world_size = ddp_setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main():
        log.info(f"Using device: {device} | world_size={world_size}")

    set_seed(cfg.train.seed, rank)

    if is_main():
        log.info("Creating dataloader")
    dataloader = get_pushT_loader(cfg, rank=rank, world_size=world_size)
    validation_loader = None
    if cfg.validation.enabled and is_main():
        log.info(
            f"Creating fixed validation loader: demos "
            f"[{cfg.validation.demo_start}, "
            f"{cfg.validation.demo_start + cfg.validation.num_demos})"
        )
        validation_loader = get_pushT_validation_loader(cfg)

    if is_main():
        log.info("Creating model")
    denoiser = create_model(cfg, device)

    if world_size > 1:
        denoiser = DDP(
            denoiser,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
        inner = denoiser.module
    else:
        inner = denoiser

    if is_main():
        log.info("Creating optimizer")
    optimizer = torch.optim.AdamW(
        params=inner.parameters(),
        lr=cfg.opt.lr,
        betas=(0.9, cfg.opt.beta2),
        weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-6/cfg.opt.lr,
        end_factor=1.0,
        total_iters=cfg.opt.warmup_steps
    )

    actual_step = 0 # current step
    to_skip = False # Whether to skip to the correct location in dataloader
    best_val_loss = float('inf')
    best_val_step = None

    if is_main():
        log.info("Creating model / Restoring checkpoint")
        os.makedirs(f"{cfg.output_dir}/ckpt_save", exist_ok=True)
        legacy_second_latest = os.path.join(cfg.output_dir, "ckpt-2nd-latest.pth")
        if os.path.exists(legacy_second_latest):
            os.remove(legacy_second_latest)
            log.info(f"Removed legacy checkpoint {legacy_second_latest}")
    barrier(world_size)

    # Load checkpoint
    if os.path.exists(cfg.path_ckpt_latest):
        ckpt = torch.load(cfg.path_ckpt_latest, map_location="cpu", weights_only=False)
        inner.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        actual_step = ckpt['step'] + 1
        best_val_loss = ckpt.get('best_val_loss', best_val_loss)
        best_val_step = ckpt.get('best_val_step', best_val_step)
        rank_states = ckpt.get('rng_state_by_rank')
        if rank_states is not None:
            if len(rank_states) != world_size:
                raise RuntimeError(
                    f"Checkpoint has {len(rank_states)} RNG states for world_size={world_size}"
                )
            load_rng_state_dict(rank_states[rank])
        else:
            load_rng_state_dict(ckpt.get('rng_state'))
        del ckpt
        if is_main():
            log.info(f"Restored from step {actual_step} ckpt")
        if actual_step % len(dataloader) != 0:
            to_skip = True
    elif cfg.train.get('init_checkpoint') and os.path.exists(cfg.train.init_checkpoint):
        ckpt = torch.load(cfg.train.init_checkpoint, map_location="cpu", weights_only=False)
        missing = load_initial_model(inner, ckpt['model'])
        if is_main():
            log.info(
                f"Initialized from {cfg.train.init_checkpoint}; "
                f"zero-initialized keys kept: {missing}"
            )
        del ckpt
    elif cfg.model.is_phase_2 == True and os.path.exists(cfg.path_ckpt_phase1):
        # Just started phase 2, so load from phase 1 checkpoint
        ckpt = torch.load(cfg.path_ckpt_phase1, map_location="cpu", weights_only=False)
        inner.load_state_dict(ckpt['model'])
        if is_main():
            log.info(f"Restored from phase 1's step {ckpt['step']} ckpt to begin phase 2 training")
        del ckpt
    barrier(world_size)

    # Set up wandb run
    if is_main():
        training_rng_state = rng_state_dict()
        if cfg.wandb_info.get('key'):
            wandb.login(key=cfg.wandb_info.key)
        if not os.path.exists(cfg.wandb_info.saved_run_id):
            run = wandb.init(
                entity=cfg.wandb_info.entity,
                project=cfg.wandb_info.project,
                name=cfg.wandb_info.name,
                dir=cfg.output_dir
            )
            run_id = run.id

            with open(cfg.wandb_info.saved_run_id, 'w') as f:
                json.dump({'run_id': run_id}, f)

            log.info(f"Started new wandb run {run_id}")
        else:
            log.info(f"Resuming wandb run")
            with open(cfg.wandb_info.saved_run_id, 'r') as f:
                run_id = json.load(f)['run_id']

            run = wandb.init(
                entity=cfg.wandb_info.entity,
                project=cfg.wandb_info.project,
                id=run_id,
                resume="allow",
                dir=cfg.output_dir
            )

        n_params = sum(p.numel() for p in inner.parameters() if p.requires_grad)
        wandb.log({'num_params': n_params, 'seed': cfg.train.seed}, step=actual_step)
        load_rng_state_dict(training_rng_state)
    barrier(world_size)

    start_ep = actual_step // len(dataloader) + 1
    first_pass = True
    max_steps = cfg.train.get('max_steps')
    training_complete = False
    last_validation_step = None
    for epoch_idx in range(start_ep, cfg.train.num_epochs + 1):
        dataloader.generator.manual_seed(
            cfg.train.seed + epoch_idx * world_size + rank
        )
        if world_size > 1 and isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch_idx)

        if is_main():
            log.info(f"(epoch {epoch_idx}) start")
        for batch_idx, nbatch in enumerate(dataloader):
            if max_steps is not None and actual_step >= max_steps:
                training_complete = True
                break
            # Reach actual_step by skipping forward in dataloader
            if to_skip:
                cur_step = len(dataloader) * (epoch_idx - 1) + batch_idx
                if cur_step < actual_step:
                    continue
                elif cur_step == actual_step:
                    to_skip = False

            if cfg.data.normalize_img:
                nbatch['image'] = (nbatch['image'] - 0.5) / 0.5 # to [-1,1] range
            if first_pass and is_main():
                log.info(f"batch[image]: {nbatch['image'].shape} | {nbatch['image'].min()} | {nbatch['image'].max()}")
                log.info(f"batch[action]: {nbatch['action'].shape} | {nbatch['action'].min()} | {nbatch['action'].max()}")
            first_pass = False

            loss, metrics = denoiser(nbatch, device)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(inner.parameters(), max_norm=cfg.opt.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            inner.update_ema()

            metrics['lr'] = scheduler.get_last_lr()[0]

            if is_main():
                wandb.log(metrics, step=actual_step)
                log.info(f"(epoch {epoch_idx}) (batch {batch_idx}/{len(dataloader)})")
                for k, v in metrics.items():
                    if k.startswith("loss_") or k == "lr":
                        log.info(f"{k}: {v}")

            validation_improved = False
            should_validate = (
                cfg.validation.enabled
                and actual_step > 0
                and actual_step % cfg.validation.every == 0
            )
            if should_validate:
                if is_main():
                    validation_metrics = evaluate_validation(
                        inner, validation_loader, cfg, device
                    )
                    validation_loss = validation_metrics['validation/loss']
                    if validation_loss < best_val_loss:
                        best_val_loss = validation_loss
                        best_val_step = actual_step
                        validation_improved = True
                    validation_metrics['validation/best_loss'] = best_val_loss
                    validation_metrics['validation/best_step'] = best_val_step
                    wandb.log(validation_metrics, step=actual_step)
                    log.info(
                        f"validation/loss: {validation_loss:.8f} | "
                        f"best: {best_val_loss:.8f} at step {best_val_step}"
                    )
                last_validation_step = actual_step
                barrier(world_size)

            if actual_step % cfg.train.ckpt_every == 0 or should_validate:
                rng_states = gather_rng_states(world_size)
                if is_main():
                    log.info(f"(epoch {epoch_idx}) Saving latest ckpt at step {actual_step}")
                    checkpoint = {
                        'step': actual_step,
                        'model': inner.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'rng_state_by_rank': rng_states,
                        'best_val_loss': best_val_loss,
                        'best_val_step': best_val_step,
                    }
                    save_checkpoint_atomic(
                        checkpoint,
                        cfg.path_ckpt_latest,
                    )
                    if validation_improved:
                        copy_checkpoint_atomic(cfg.path_ckpt_latest, cfg.path_ckpt_best)
                        log.info(f"Saved best ckpt at step {actual_step}")
                barrier(world_size)
            actual_step += 1
        if training_complete:
            break

    needs_final_checkpoint = (
        max_steps is not None
        and actual_step > 0
        and (
            (actual_step - 1) % cfg.train.ckpt_every != 0
            or (cfg.validation.enabled and last_validation_step != actual_step - 1)
        )
    )
    if needs_final_checkpoint:
        final_step = actual_step - 1
        validation_improved = False
        if cfg.validation.enabled and last_validation_step != final_step:
            if is_main():
                validation_metrics = evaluate_validation(inner, validation_loader, cfg, device)
                validation_loss = validation_metrics['validation/loss']
                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                    best_val_step = final_step
                    validation_improved = True
                validation_metrics['validation/best_loss'] = best_val_loss
                validation_metrics['validation/best_step'] = best_val_step
                wandb.log(validation_metrics, step=final_step)
                log.info(
                    f"validation/loss: {validation_loss:.8f} | "
                    f"best: {best_val_loss:.8f} at step {best_val_step}"
                )
            barrier(world_size)
        rng_states = gather_rng_states(world_size)
        if is_main():
            log.info(f"Saving final checkpoint at step {final_step}")
            checkpoint = {
                'step': final_step,
                'model': inner.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'rng_state_by_rank': rng_states,
                'best_val_loss': best_val_loss,
                'best_val_step': best_val_step,
            }
            save_checkpoint_atomic(
                checkpoint,
                cfg.path_ckpt_latest,
            )
            if validation_improved:
                copy_checkpoint_atomic(cfg.path_ckpt_latest, cfg.path_ckpt_best)
                log.info(f"Saved best ckpt at step {final_step}")
        barrier(world_size)

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()
