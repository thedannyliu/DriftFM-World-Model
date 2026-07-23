"""Hydra entry point for Drift Flow transport diagnostics."""

import json
import logging

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="configs/train",
    config_name="pushT_driftflow",
)
def main(cfg: DictConfig):
    from eval.transport_audit import evaluate_transport_audit

    summary = evaluate_transport_audit(cfg)
    metrics = summary["metrics"]
    wandb_project = cfg.audit.get("wandb_project")
    if wandb_project:
        import wandb

        run = wandb.init(
            entity=cfg.audit.get("wandb_entity"),
            project=wandb_project,
            name=cfg.audit.run_name,
            job_type="transport-audit",
            config={
                "checkpoint": summary["checkpoint"],
                "checkpoint_step": summary["checkpoint_step"],
                "seed": summary["seed"],
                "num_batches": summary["num_batches"],
                "particles": summary["particles"],
                "gpu": summary["gpu"],
            },
        )
        wandb.log(metrics)
        summary["wandb_run_id"] = run.id
        run.finish()
        with open(cfg.audit.output, "w") as output_file:
            json.dump(summary, output_file, indent=2, sort_keys=True)
    log.info(
        "transport audit complete: "
        f"nfe1_mse={metrics['free/nfe1/final/paired_mse']:.6f} "
        f"nfe2_mse={metrics['free/nfe2/final/paired_mse']:.6f} "
        f"nfe4_mse={metrics['free/nfe4/final/paired_mse']:.6f} "
        f"output={cfg.audit.output}"
    )


if __name__ == "__main__":
    main()
