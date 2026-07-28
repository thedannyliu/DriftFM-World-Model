"""Hydra entry point for advantage-aligned transport diagnostics."""

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
    from eval.hypothesis_audit import evaluate_hypothesis_audit

    summary = evaluate_hypothesis_audit(cfg)
    audit_cfg = cfg.hypothesis_audit
    wandb_project = audit_cfg.get("wandb_project")
    if wandb_project:
        import wandb

        run = wandb.init(
            entity=audit_cfg.get("wandb_entity"),
            project=wandb_project,
            name=audit_cfg.run_name,
            job_type="hypothesis-audit",
            config={
                "checkpoint": summary["checkpoint"],
                "checkpoint_step": summary["checkpoint_step"],
                "family": summary["family"],
                "seed": summary["seed"],
                "num_batches": summary["num_batches"],
                "particles": summary["particles"],
                "gpu": summary["gpu"],
            },
        )
        log_values = dict(summary["metrics"])
        log_values.update({
            f"correlation/{name}": value
            for name, value in summary["correlations"].items()
            if value is not None
        })
        for stratum, values in summary["motion_strata"].items():
            log_values.update({
                f"motion_strata/{stratum}/{name}": value
                for name, value in values.items()
            })
        wandb.log(log_values)
        summary["wandb_run_id"] = run.id
        run.finish()
        with open(audit_cfg.output, "w") as output_file:
            json.dump(summary, output_file, indent=2, sort_keys=True)

    risks = summary["metrics"]
    correlations = summary["correlations"]
    log.info(
        "hypothesis audit complete: "
        f"mse_1/2/4/8="
        f"{risks['risk/nfe1/mse']:.6f}/"
        f"{risks['risk/nfe2/mse']:.6f}/"
        f"{risks['risk/nfe4/mse']:.6f}/"
        f"{risks['risk/nfe8/mse']:.6f} "
        f"corr_defect_degradation_2to4="
        f"{correlations['defect_vs_degradation/2_to_4/mse']} "
        f"output={audit_cfg.output}"
    )


if __name__ == "__main__":
    main()
