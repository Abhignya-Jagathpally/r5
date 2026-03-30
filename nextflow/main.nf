#!/usr/bin/env nextflow

/*
================================================================================
MM Imaging Pathology & Radiomics Pipeline - Nextflow DSL2
================================================================================

A comprehensive Nextflow pipeline for:
1. WSI tiling and stain normalization
2. Tile deduplication
3. Foundation model embeddings
4. Radiomics feature extraction
5. Multi-modal model training and evaluation

Features:
- Container support (Docker/Singularity)
- HPC/Cloud resource labels
- Error handling and automatic retries
- Publish directories for results

Configuration: nextflow/nextflow.config
================================================================================
*/

nextflow.enable.dsl = 2

// Import processes from modules
include { tile_wsis } from './processes/preprocessing.nf'
include { normalize_tiles } from './processes/preprocessing.nf'
include { deduplicate_tiles } from './processes/preprocessing.nf'
include { extract_embeddings } from './processes/embeddings.nf'
include { extract_radiomics } from './processes/radiomics.nf'
include { create_splits } from './processes/data_preparation.nf'
include { train_baseline_model } from './processes/training.nf'
include { train_foundation_model } from './processes/training.nf'
include { train_fusion_model } from './processes/training.nf'
include { evaluate_models } from './processes/evaluation.nf'
include { generate_report } from './processes/reporting.nf'

workflow {
    log.info """
    ================================================================================
    MM Imaging Radiomics Pipeline
    ================================================================================

    Data directory:          ${params.data_dir}
    Output directory:        ${params.output_dir}
    Embedding backbone:      ${params.embedding_backbone}
    Number of samples:       ${params.num_samples}

    ================================================================================
    """

    // 1. Tile WSIs
    if (params.stages.tiling) {
        wsi_files = Channel
            .fromPath("${params.data_dir}/wsis/**/*.svs")
            .map { file ->
                def patient_id = file.name.replaceAll('.svs', '')
                tuple(patient_id, file)
            }

        tile_wsis(wsi_files)
        tiles = tile_wsis.out.tiles
    } else {
        tiles = Channel
            .fromPath("${params.output_dir}/tiles/**/*.npz")
            .map { file ->
                def patient_id = file.name.replaceAll('.npz', '')
                tuple(patient_id, file)
            }
    }

    // 2. Normalize tiles
    if (params.stages.stain_norm) {
        normalize_tiles(tiles)
        normalized = normalize_tiles.out.normalized
    } else {
        normalized = tiles
    }

    // 3. Deduplicate tiles
    if (params.stages.dedup) {
        deduplicate_tiles(normalized)
        deduplicated = deduplicate_tiles.out.deduplicated
    } else {
        deduplicated = normalized
    }

    // 4. Extract embeddings
    if (params.stages.embeddings) {
        extract_embeddings(deduplicated)
        embeddings = extract_embeddings.out.embeddings
    } else {
        embeddings = Channel
            .fromPath("${params.output_dir}/embeddings/**/*.h5")
            .map { file ->
                def patient_id = file.name.replaceAll('.h5', '')
                tuple(patient_id, file)
            }
    }

    // 5. Extract radiomics
    if (params.stages.radiomics) {
        imaging_files = Channel
            .fromPath("${params.data_dir}/imaging/*")
            .filter { it.isDirectory() }
            .map { dir ->
                def patient_id = dir.name
                def image = file("${dir}/image.nii.gz")
                def mask = file("${dir}/mask.nii.gz")
                tuple(patient_id, image, mask)
            }

        extract_radiomics(imaging_files)
        radiomics = extract_radiomics.out.features
    } else {
        radiomics = Channel.empty()
    }

    // 6. Create data splits
    metadata = file("${params.data_dir}/metadata.csv")
    create_splits(metadata)

    // 7. Train baseline models
    baseline_models = Channel.from(params.baseline_models)

    embeddings_collected = embeddings.collect()

    train_baseline_model(
        baseline_models,
        embeddings_collected,
        create_splits.out.train_split,
        create_splits.out.val_split
    )

    // 8. Train foundation models
    foundation_models = Channel.from(params.foundation_models)

    train_foundation_model(
        foundation_models,
        embeddings_collected,
        create_splits.out.train_split,
        create_splits.out.val_split
    )

    // 9. Train fusion model
    radiomics_collected = radiomics.collect()

    train_fusion_model(
        embeddings_collected,
        radiomics_collected,
        create_splits.out.train_split,
        create_splits.out.val_split
    )

    // 10. Evaluate all models
    baseline_models_out = train_baseline_model.out.model.collect()
    foundation_models_out = train_foundation_model.out.model.collect()
    fusion_model_out = train_fusion_model.out.model.collect()

    evaluate_models(
        baseline_models_out,
        foundation_models_out,
        fusion_model_out,
        create_splits.out.test_split,
        embeddings_collected,
        radiomics_collected
    )

    // 11. Generate report
    generate_report(
        evaluate_models.out.results,
        evaluate_models.out.curves
    )
}

workflow.onComplete {
    log.info """
    ================================================================================
    Pipeline Complete
    ================================================================================
    Status:          ${ workflow.success ? 'SUCCESS' : 'FAILED' }
    Duration:        $workflow.duration
    Results:         ${params.output_dir}
    ================================================================================
    """
}

workflow.onError {
    log.error """
    ================================================================================
    Pipeline Failed
    ================================================================================
    Error message:  $workflow.errorMessage
    Error report:   ${workflow.errorReport}
    ================================================================================
    """
}
