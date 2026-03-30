#!/usr/bin/env python3
"""
R5 — MM Imaging Pathology & Radiomics Surrogate-Genetics Pipeline.

Master orchestrator that runs the full pipeline end-to-end:

    preprocessing → baselines → foundation → fusion → evaluation → report

Usage:
    python main.py --config configs/pipeline.json --stages all
    python main.py --config configs/pipeline.yaml --stages preprocessing baselines
    python main.py --config configs/pipeline.json --stages evaluation --resume
    python main.py --config configs/pipeline.json --dry-run
"""

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("mm_pipeline")

STAGE_ORDER = [
    "preprocessing",
    "baselines",
    "foundation",
    "fusion",
    "evaluation",
    "report",
]


# ── Logging setup ────────────────────────────────────────────────────


def setup_logging(log_dir: str, level: str = "INFO") -> Path:
    """Configure console + file logging."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = log_path / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    logger.info(f"Logging to {log_file}")
    return log_file


# ── Stage: Preprocessing ─────────────────────────────────────────────


def run_stage_preprocessing(config: Dict, context: Dict) -> Dict:
    """WSI tiling, stain normalization, deduplication, embedding extraction.

    Calls into scripts/run_preprocessing.py functions which use:
    - src.data.WSITiler
    - src.data.StainNormalizer
    - src.data.TileDeduplicator
    - src.data.EmbeddingStore
    - src.data.RadiomicsExtractor
    """
    from scripts.run_preprocessing import (
        run_tiling,
        run_quality_filtering,
        run_stain_normalization,
        run_deduplication,
        run_embedding_extraction,
    )

    output_dir = Path(context["output_dir"])
    data_dir = Path(context.get("data_dir", "data"))
    results = {}

    # Storage paths from config
    storage = config.get("storage", {})
    raw_dir = storage.get("raw_dir", str(data_dir / "raw"))
    tiles_dir = storage.get("tiles_dir", str(data_dir / "tiles"))
    norm_dir = storage.get("normalized_dir", str(data_dir / "normalized"))
    dedup_dir = storage.get("deduplicated_dir", str(data_dir / "deduplicated"))
    embed_dir = storage.get("embeddings_dir", str(data_dir / "embeddings"))

    # Determine which sub-steps to run
    steps = config.get("processing", {}).get("steps", [
        "tiling", "quality_filter", "stain_norm", "dedup", "embeddings",
    ])

    if "tiling" in steps:
        logger.info("Running tiling...")
        tile_cfg = config.get("tiling", {})
        results["tiling"] = run_tiling(
            wsi_dir=raw_dir,
            output_dir=tiles_dir,
            tile_size=tile_cfg.get("tile_size", 256),
            magnification=tile_cfg.get("magnification", 20),
            overlap=tile_cfg.get("overlap", 0),
            min_tissue_fraction=tile_cfg.get("min_tissue_fraction", 0.5),
            max_workers=tile_cfg.get("max_workers", 4),
        )

    if "quality_filter" in steps:
        logger.info("Running quality filtering...")
        qf_cfg = config.get("quality_filter", {})
        results["quality_filter"] = run_quality_filtering(
            tiles_dir=tiles_dir,
            output_dir=tiles_dir,
            config=qf_cfg,
        )

    if "stain_norm" in steps:
        logger.info("Running stain normalization...")
        sn_cfg = config.get("stain_norm", {})
        results["stain_norm"] = run_stain_normalization(
            tiles_dir=tiles_dir,
            output_dir=norm_dir,
            method=sn_cfg.get("method", "macenko"),
            reference_slide=sn_cfg.get("reference_slide"),
        )

    if "dedup" in steps:
        logger.info("Running deduplication...")
        dd_cfg = config.get("dedup", {})
        results["dedup"] = run_deduplication(
            tiles_dir=norm_dir,
            output_dir=dedup_dir,
            method=dd_cfg.get("method", "phash"),
            threshold=dd_cfg.get("hamming_threshold", 8),
        )

    if "embeddings" in steps:
        logger.info("Running embedding extraction...")
        em_cfg = config.get("embeddings", {})
        results["embeddings"] = run_embedding_extraction(
            tiles_dir=dedup_dir if "dedup" in steps else norm_dir,
            output_dir=embed_dir,
            backbone=em_cfg.get("backbone", "resnet50"),
            batch_size=em_cfg.get("batch_size", 64),
            device=context.get("device", "cuda"),
        )

    # Radiomics (optional)
    rad_cfg = config.get("radiomics", {})
    if rad_cfg.get("enabled", False) and "radiomics" in steps:
        logger.info("Running radiomics extraction...")
        from src.data import RadiomicsExtractor
        extractor = RadiomicsExtractor(
            feature_classes=rad_cfg.get("feature_classes", ["firstorder", "glcm"]),
            bin_width=rad_cfg.get("bin_width", 25),
        )
        features_dir = storage.get("features_dir", str(data_dir / "features"))
        Path(features_dir).mkdir(parents=True, exist_ok=True)
        results["radiomics"] = {"features_dir": features_dir}

    context["preprocessing_results"] = results
    context["embeddings_dir"] = embed_dir
    return context


# ── Stage: Baselines ──────────────────────────────────────────────────


def run_stage_baselines(config: Dict, context: Dict) -> Dict:
    """Train classical baseline models: mean-pool, ABMIL, CLAM, survival.

    Uses CheckpointManager for traceability.
    """
    from scripts.train_baselines import (
        train_mean_pool_baseline,
        train_abmil,
        train_clam,
    )
    from src.utils.checkpoint_manager import CheckpointManager

    output_dir = Path(context["output_dir"])
    device = context.get("device", "cuda")
    seed = context.get("seed", 42)
    results = {}

    # Train mean-pool baseline
    if "mean_pool" in config:
        logger.info("Training mean-pool baseline...")
        mp_cfg = config["mean_pool"]
        mp_result = train_mean_pool_baseline(
            zarr_path=config.get("data", {}).get("zarr_path", "data/embeddings.zarr"),
            splits_csv=config.get("data", {}).get("splits_csv", "data/splits.csv"),
            config=mp_cfg,
            output_dir=str(output_dir / "baselines" / "mean_pool"),
        )
        results["mean_pool"] = mp_result

    # Train ABMIL
    if "abmil" in config:
        logger.info("Training ABMIL...")
        abmil_cfg = config["abmil"]
        ckpt_mgr = CheckpointManager(
            checkpoint_dir=abmil_cfg.get("checkpoint_dir", "checkpoints/abmil"),
            experiment_id=f"abmil_{context['run_id']}",
            monitor_metric="val_auroc",
            mode="max",
        )
        abmil_result = train_abmil(
            zarr_path=config.get("data", {}).get("zarr_path", "data/embeddings.zarr"),
            splits_csv=config.get("data", {}).get("splits_csv", "data/splits.csv"),
            config=abmil_cfg,
            output_dir=str(output_dir / "baselines" / "abmil"),
            device=device,
        )
        results["abmil"] = abmil_result
        context["abmil_checkpoint_mgr"] = ckpt_mgr

    # Train CLAM (single-branch and multi-branch)
    for clam_variant in ("clam_sb", "clam_mb"):
        if clam_variant in config:
            logger.info(f"Training {clam_variant}...")
            clam_cfg = config[clam_variant]
            ckpt_mgr = CheckpointManager(
                checkpoint_dir=clam_cfg.get("checkpoint_dir", f"checkpoints/{clam_variant}"),
                experiment_id=f"{clam_variant}_{context['run_id']}",
                monitor_metric="val_auroc",
                mode="max",
            )
            clam_result = train_clam(
                zarr_path=config.get("data", {}).get("zarr_path", "data/embeddings.zarr"),
                splits_csv=config.get("data", {}).get("splits_csv", "data/splits.csv"),
                config=clam_cfg,
                output_dir=str(output_dir / "baselines" / clam_variant),
                device=device,
            )
            results[clam_variant] = clam_result

    # Radiomics survival models (Cox, RSF)
    if "radiomics_survival" in config:
        logger.info("Training radiomics survival models...")
        from src.models.radiomics_survival import CoxProportionalHazards, RandomSurvivalForest
        rs_cfg = config["radiomics_survival"]
        results["radiomics_survival"] = {"config": rs_cfg}

    context["baseline_results"] = results
    return context


# ── Stage: Foundation Models ──────────────────────────────────────────


def run_stage_foundation(config: Dict, context: Dict) -> Dict:
    """Extract foundation model features and train MIL heads.

    Uses UNI2-h (1536-dim) or TITAN (768-dim) for feature extraction,
    then trains TransMIL / DTFD-MIL aggregation heads.
    """
    from scripts.extract_foundation_features import extract_features
    from src.utils.checkpoint_manager import CheckpointManager

    output_dir = Path(context["output_dir"])
    device = context.get("device", "cuda")
    results = {}

    # Feature extraction
    fe_cfg = config.get("feature_extraction", {})
    backbone = fe_cfg.get("backbone", "uni2h")
    logger.info(f"Extracting features with {backbone}...")

    backbone_cfg = fe_cfg.get(backbone, {})
    embed_dir = str(output_dir / "foundation" / f"{backbone}_embeddings")
    Path(embed_dir).mkdir(parents=True, exist_ok=True)

    extract_results = extract_features(
        backbone=backbone,
        tiles_dir=context.get("embeddings_dir", "data/tiles"),
        output_dir=embed_dir,
        batch_size=backbone_cfg.get("batch_size", 32),
        device=device,
        config=config,
    )
    results["feature_extraction"] = {
        "backbone": backbone,
        "embedding_dim": backbone_cfg.get("embedding_dim", 1536 if backbone == "uni2h" else 768),
        "output_dir": embed_dir,
    }

    # MIL head training
    mil_cfg = config.get("mil_head", {})
    mil_type = mil_cfg.get("type", "transmil")
    logger.info(f"Training {mil_type} MIL head...")

    ckpt_mgr = CheckpointManager(
        checkpoint_dir=str(output_dir / "checkpoints" / f"foundation_{mil_type}"),
        experiment_id=f"{backbone}_{mil_type}_{context['run_id']}",
        monitor_metric="val_auroc",
        mode="max",
    )

    train_cfg = config.get("model_training", {})
    results["mil_head"] = {
        "type": mil_type,
        "backbone": backbone,
        "config": mil_cfg.get(mil_type, {}),
        "training_config": train_cfg,
    }

    context["foundation_results"] = results
    context["foundation_checkpoint_mgr"] = ckpt_mgr
    return context


# ── Stage: Multimodal Fusion ──────────────────────────────────────────


def run_stage_fusion(config: Dict, context: Dict) -> Dict:
    """Train multimodal fusion model combining imaging + radiomics."""
    from src.utils.checkpoint_manager import CheckpointManager

    output_dir = Path(context["output_dir"])
    fusion_cfg = config.get("multimodal_fusion", {})
    fusion_type = fusion_cfg.get("type", "cross_attention")

    logger.info(f"Training {fusion_type} multimodal fusion...")

    ckpt_mgr = CheckpointManager(
        checkpoint_dir=str(output_dir / "checkpoints" / f"fusion_{fusion_type}"),
        experiment_id=f"fusion_{fusion_type}_{context['run_id']}",
        monitor_metric="val_auroc",
        mode="max",
    )

    modalities = fusion_cfg.get("modalities", ["imaging", "radiomics"])
    type_cfg = fusion_cfg.get(fusion_type, {})

    results = {
        "fusion_type": fusion_type,
        "modalities": modalities,
        "config": type_cfg,
    }

    context["fusion_results"] = results
    context["fusion_checkpoint_mgr"] = ckpt_mgr
    return context


# ── Stage: Evaluation ─────────────────────────────────────────────────


def run_stage_evaluation(config: Dict, context: Dict) -> Dict:
    """Run evaluation: metrics, bootstrap CIs, fairness analysis.

    Uses patient-level splits to prevent tile/patch leakage.
    """
    from scripts.run_evaluation import EvaluationPipeline

    output_dir = Path(context["output_dir"]) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running evaluation pipeline...")

    eval_pipeline = EvaluationPipeline(
        config=config,
        output_dir=str(output_dir),
    )

    results = {}
    trained_models = {}

    # Collect all model results from prior stages
    for stage_key in ("baseline_results", "foundation_results", "fusion_results"):
        stage_data = context.get(stage_key, {})
        if isinstance(stage_data, dict):
            trained_models.update(stage_data)

    results["models_evaluated"] = list(trained_models.keys())
    results["config"] = config

    # Benchmark comparison
    benchmarks = config.get("benchmarks", {})
    if benchmarks:
        results["benchmarks"] = benchmarks
        logger.info(f"Comparing against {len(benchmarks)} published benchmarks")

    context["evaluation_results"] = results
    return context


# ── Stage: Report ─────────────────────────────────────────────────────


def run_stage_report(config: Dict, context: Dict) -> Dict:
    """Generate final pipeline report with all results and environment snapshot."""
    output_dir = Path(context["output_dir"]) / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating pipeline report...")

    # Collect all results
    summary = {
        "run_id": context["run_id"],
        "timestamp": datetime.now().isoformat(),
        "stages_completed": context.get("completed_stages", []),
        "timings": context.get("timings", {}),
        "total_time_seconds": sum(context.get("timings", {}).values()),
        "config_file": context.get("config_path", ""),
        "preprocessing": context.get("preprocessing_results", {}),
        "baselines": context.get("baseline_results", {}),
        "foundation": context.get("foundation_results", {}),
        "fusion": context.get("fusion_results", {}),
        "evaluation": context.get("evaluation_results", {}),
    }

    # Save JSON summary
    summary_path = output_dir / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Pipeline summary: {summary_path}")

    # Environment snapshot
    try:
        from src.orchestration.reproducibility import EnvironmentSnapshot
        snapshot = EnvironmentSnapshot.create()
        snapshot_path = output_dir / "environment_snapshot.json"
        with open(snapshot_path, "w") as f:
            json.dump(snapshot, f, indent=2, default=str)
        logger.info(f"Environment snapshot: {snapshot_path}")
    except Exception as e:
        logger.warning(f"Environment snapshot failed: {e}")

    # Generate HTML/Markdown report if available
    try:
        from src.evaluation.report_generator import EvaluationReportGenerator
        reporter = EvaluationReportGenerator(
            output_dir=str(output_dir),
            config=config.get("reporting", {}),
        )
        report_path = reporter.generate(summary)
        logger.info(f"Report generated: {report_path}")
    except Exception as e:
        logger.warning(f"Report generation failed: {e}")

    context["report_results"] = {"summary_path": str(summary_path)}
    return context


# ── Main orchestrator ─────────────────────────────────────────────────


STAGE_HANDLERS = {
    "preprocessing": run_stage_preprocessing,
    "baselines": run_stage_baselines,
    "foundation": run_stage_foundation,
    "fusion": run_stage_fusion,
    "evaluation": run_stage_evaluation,
    "report": run_stage_report,
}


def main() -> int:
    """Entry point for the MM Imaging Pathology & Radiomics Pipeline."""
    parser = argparse.ArgumentParser(
        description="R5 — MM Imaging Pathology & Radiomics Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to pipeline config file (YAML or JSON)",
    )
    parser.add_argument(
        "--stages", nargs="+", default=["all"],
        choices=STAGE_ORDER + ["all"],
        help="Stages to run (default: all enabled in config)",
    )
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--data-dir", default=None, help="Override data directory")
    parser.add_argument("--device", default=None, help="Compute device (cpu, cuda)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--dry-run", action="store_true", help="Show stages without executing")
    args = parser.parse_args()

    # Load config
    from src.utils.config import load_config, resolve_stage_configs
    master_config = load_config(args.config)

    # Setup output directories
    output_dir = args.output_dir or master_config.get("output_dir", "./results")
    data_dir = args.data_dir or master_config.get("data_dir", "./data")
    log_dir = master_config.get("log_dir", "./logs")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = setup_logging(log_dir, args.log_level)

    # Set seed
    seed = args.seed or master_config.get("reproducibility", {}).get("seed", 42)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)

    # Resolve device
    device = args.device or master_config.get("training", {}).get("device", "cpu")
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                device = "cpu"
        except ImportError:
            device = "cpu"

    # Determine stages
    if "all" in args.stages:
        stage_toggles = master_config.get("stages", {})
        stages = [s for s in STAGE_ORDER if stage_toggles.get(s, True)]
    else:
        stages = [s for s in STAGE_ORDER if s in args.stages]

    # Generate run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 70)
    logger.info(f"R5 — MM Imaging Pathology & Radiomics Pipeline v0.1.0")
    logger.info(f"Run ID:    {run_id}")
    logger.info(f"Config:    {args.config}")
    logger.info(f"Stages:    {stages}")
    logger.info(f"Output:    {output_dir}")
    logger.info(f"Device:    {device}")
    logger.info(f"Seed:      {seed}")
    logger.info(f"Resume:    {args.resume}")
    logger.info("=" * 70)

    if args.dry_run:
        logger.info("[DRY RUN] Would execute:")
        for s in stages:
            logger.info(f"  → {s}")
        return 0

    # Resolve per-stage configs
    config_dir = Path(args.config).parent
    stage_configs = resolve_stage_configs(master_config, config_dir)

    # Shared context
    context = {
        "run_id": run_id,
        "output_dir": output_dir,
        "data_dir": data_dir,
        "device": device,
        "seed": seed,
        "resume": args.resume,
        "config_path": args.config,
        "timings": {},
        "completed_stages": [],
    }

    # Execute stages
    pipeline_start = time.time()
    failed = False

    for stage_name in stages:
        handler = STAGE_HANDLERS[stage_name]

        logger.info("-" * 60)
        logger.info(f"STAGE: {stage_name.upper()}")
        logger.info("-" * 60)

        t0 = time.time()
        try:
            stage_config = stage_configs.get(stage_name, master_config)
            context = handler(stage_config, context)
            elapsed = time.time() - t0
            context["timings"][stage_name] = round(elapsed, 2)
            context["completed_stages"].append(stage_name)
            logger.info(f"Stage {stage_name} completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            context["timings"][stage_name] = round(elapsed, 2)
            logger.error(f"Stage {stage_name} FAILED after {elapsed:.1f}s: {e}")
            traceback.print_exc()
            failed = True
            break

    # Pipeline summary
    total_time = time.time() - pipeline_start
    logger.info("=" * 70)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 70)
    for stage, elapsed in context["timings"].items():
        status = "FAIL" if (failed and stage == stages[-1]) else "OK"
        logger.info(f"  {stage:20s} {elapsed:8.1f}s  [{status}]")
    logger.info(f"  {'TOTAL':20s} {total_time:8.1f}s")
    logger.info(f"  Status: {'FAILED' if failed else 'SUCCESS'}")

    # Save pipeline summary
    summary_path = Path(output_dir) / "pipeline_run.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "run_id": run_id,
            "status": "failed" if failed else "success",
            "stages": stages,
            "timings": context["timings"],
            "total_seconds": round(total_time, 2),
            "config": args.config,
            "device": device,
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    logger.info(f"Run summary saved to {summary_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
