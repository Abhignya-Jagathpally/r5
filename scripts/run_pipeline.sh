#!/bin/bash

################################################################################
# Master Pipeline Execution Script — Snakemake/Nextflow Orchestration Wrapper
# MM Imaging Pathology & Radiomics Pipeline
#
# NOTE: For the Python pipeline entry point, prefer:
#     python main.py --help
#   which supports --config, --stages, --dry-run, --output-dir, --seed,
#   --list-stages, and --verbose flags directly.
#   This shell script is a wrapper for Snakemake/Nextflow orchestration.
#
# Usage: ./scripts/run_pipeline.sh [OPTIONS]
#
# Options:
#   -e, --engine         Execution engine: snakemake or nextflow (default: snakemake)
#   -p, --profile        Execution profile: local, slurm, cloud (default: local)
#   -c, --config         Config file path (default: configs/pipeline.yaml)
#   -s, --stages         Comma-separated stages to run (default: all)
#   -d, --dry-run        Run in dry-run mode (don't execute)
#   -j, --jobs           Number of parallel jobs (default: 4)
#   -h, --help           Show this help message
################################################################################

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Default parameters
ENGINE="snakemake"
PROFILE="local"
CONFIG="${REPO_ROOT}/configs/pipeline.yaml"
STAGES=""
DRY_RUN=0
NUM_JOBS=4
VERBOSE=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ ERROR: $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ WARNING: $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

show_help() {
    grep "^# " "$0" | grep -v "#!/bin/bash" | sed 's/^# //' | head -20
}

check_dependencies() {
    local missing_deps=0

    print_info "Checking dependencies..."

    # Check Python
    if ! command -v python &> /dev/null; then
        print_error "Python not found"
        missing_deps=1
    else
        python_version=$(python --version 2>&1 | awk '{print $2}')
        print_success "Python ${python_version}"
    fi

    # Check git
    if ! command -v git &> /dev/null; then
        print_error "Git not found"
        missing_deps=1
    else
        print_success "Git $(git --version | awk '{print $3}')"
    fi

    # Check selected engine
    if [ "$ENGINE" = "snakemake" ]; then
        if ! command -v snakemake &> /dev/null; then
            print_error "Snakemake not found. Install with: pip install snakemake"
            missing_deps=1
        else
            snakemake_version=$(snakemake --version)
            print_success "Snakemake ${snakemake_version}"
        fi
    elif [ "$ENGINE" = "nextflow" ]; then
        if ! command -v nextflow &> /dev/null; then
            print_error "Nextflow not found. Install with: curl -s https://get.nextflow.io | bash"
            missing_deps=1
        else
            nextflow_version=$(nextflow -v 2>&1 | head -1)
            print_success "${nextflow_version}"
        fi
    fi

    # Check GPU availability
    if command -v nvidia-smi &> /dev/null; then
        gpu_count=$(nvidia-smi --list-gpus | wc -l)
        print_success "GPU(s) available: ${gpu_count}"
    else
        print_warning "No NVIDIA GPU detected (GPU support disabled)"
    fi

    if [ $missing_deps -eq 1 ]; then
        print_error "Missing dependencies. Please install them and retry."
        return 1
    fi

    return 0
}

check_config_file() {
    if [ ! -f "$CONFIG" ]; then
        print_error "Config file not found: $CONFIG"
        return 1
    fi
    print_success "Config file found: $CONFIG"
    return 0
}

setup_environment() {
    print_info "Setting up environment..."

    # Create necessary directories
    mkdir -p "${REPO_ROOT}/results"
    mkdir -p "${REPO_ROOT}/logs"
    mkdir -p "${REPO_ROOT}/.conda"

    # Activate Python environment if available
    if [ -d "${REPO_ROOT}/venv" ]; then
        source "${REPO_ROOT}/venv/bin/activate"
        print_success "Python venv activated"
    elif [ -d "${REPO_ROOT}/.conda/mm_pipeline" ]; then
        eval "$(conda shell.bash hook)"
        conda activate mm_pipeline
        print_success "Conda environment activated"
    fi

    # Export repo root to environment
    export REPO_ROOT

    return 0
}

run_snakemake() {
    print_header "Running Pipeline with Snakemake"

    local snakemake_opts=(
        "--configfile" "${CONFIG}"
        "--jobs" "${NUM_JOBS}"
        "--keep-going"
        "--rerun-incomplete"
        "--printshellcmds"
        "--reason"
        "--stats" "${REPO_ROOT}/results/snakemake_stats.json"
    )

    # Add profile-specific options
    case "$PROFILE" in
        slurm)
            snakemake_opts+=("--profile" "slurm")
            ;;
        local)
            snakemake_opts+=("--cores" "${NUM_JOBS}")
            ;;
    esac

    # Add dry-run if requested
    if [ $DRY_RUN -eq 1 ]; then
        snakemake_opts+=("--dryrun")
        print_warning "DRY RUN MODE - No files will be created"
    fi

    # Add verbose if requested
    if [ $VERBOSE -eq 1 ]; then
        snakemake_opts+=("--verbose")
    fi

    print_info "Snakemake command:"
    print_info "snakemake ${snakemake_opts[@]} $@"
    echo ""

    cd "${REPO_ROOT}"
    snakemake "${snakemake_opts[@]}" "$@"

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        print_success "Snakemake pipeline completed successfully"
    else
        print_error "Snakemake pipeline failed with exit code $exit_code"
    fi

    return $exit_code
}

run_nextflow() {
    print_header "Running Pipeline with Nextflow"

    local nf_opts=(
        "-c" "${REPO_ROOT}/nextflow/nextflow.config"
        "-profile" "${PROFILE}"
        "--output_dir" "${REPO_ROOT}/results"
    )

    # Add dry-run if requested
    if [ $DRY_RUN -eq 1 ]; then
        nf_opts+=("-preview")
        print_warning "DRY RUN MODE - Preview only, no execution"
    fi

    # Add verbose if requested
    if [ $VERBOSE -eq 1 ]; then
        nf_opts+=("-v")
    fi

    print_info "Nextflow command:"
    print_info "nextflow run ${REPO_ROOT}/nextflow/main.nf ${nf_opts[@]} $@"
    echo ""

    cd "${REPO_ROOT}"
    NXF_OPTS="-Xms500m -Xmx8g" nextflow run "${REPO_ROOT}/nextflow/main.nf" "${nf_opts[@]}" "$@"

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        print_success "Nextflow pipeline completed successfully"
    else
        print_error "Nextflow pipeline failed with exit code $exit_code"
    fi

    return $exit_code
}

generate_report() {
    print_info "Generating pipeline report..."

    local report_dir="${REPO_ROOT}/results/reports"
    mkdir -p "$report_dir"

    cat > "${report_dir}/pipeline_summary.txt" <<EOF
================================================================================
MM Imaging Radiomics Pipeline Execution Report
================================================================================

Execution Time: $(date)
Engine: $ENGINE
Profile: $PROFILE
Config: $CONFIG
Dry Run: $DRY_RUN
Parallel Jobs: $NUM_JOBS

Repository: $REPO_ROOT
Results Directory: ${REPO_ROOT}/results
Logs Directory: ${REPO_ROOT}/logs

Git Status:
$(cd "$REPO_ROOT" && git status --short 2>/dev/null || echo "Not a git repo")

Git Hash:
$(cd "$REPO_ROOT" && git rev-parse HEAD 2>/dev/null || echo "Not available")

Python Environment:
$(python --version 2>&1)

Python Packages:
$(pip list 2>/dev/null | head -10)

================================================================================
EOF

    print_success "Report generated: ${report_dir}/pipeline_summary.txt"
}

################################################################################
# Main Script
################################################################################

main() {
    print_header "MM Imaging Radiomics Pipeline"

    # Parse command-line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--engine)
                ENGINE="$2"
                shift 2
                ;;
            -p|--profile)
                PROFILE="$2"
                shift 2
                ;;
            -c|--config)
                CONFIG="$2"
                shift 2
                ;;
            -s|--stages)
                STAGES="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN=1
                shift
                ;;
            -j|--jobs)
                NUM_JOBS="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=1
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Validate inputs
    if [ "$ENGINE" != "snakemake" ] && [ "$ENGINE" != "nextflow" ]; then
        print_error "Invalid engine: $ENGINE. Must be 'snakemake' or 'nextflow'"
        exit 1
    fi

    if [ "$PROFILE" != "local" ] && [ "$PROFILE" != "slurm" ] && [ "$PROFILE" != "cloud" ]; then
        print_error "Invalid profile: $PROFILE. Must be 'local', 'slurm', or 'cloud'"
        exit 1
    fi

    print_info "Configuration:"
    print_info "  Engine:         $ENGINE"
    print_info "  Profile:        $PROFILE"
    print_info "  Config:         $CONFIG"
    print_info "  Parallel Jobs:  $NUM_JOBS"
    echo ""

    # Validate environment
    if ! check_dependencies; then
        exit 1
    fi
    echo ""

    if ! check_config_file; then
        exit 1
    fi
    echo ""

    if ! setup_environment; then
        exit 1
    fi
    echo ""

    # Run pipeline
    local exit_code=0
    if [ "$ENGINE" = "snakemake" ]; then
        run_snakemake || exit_code=$?
    elif [ "$ENGINE" = "nextflow" ]; then
        run_nextflow || exit_code=$?
    fi

    echo ""
    generate_report

    if [ $exit_code -eq 0 ]; then
        print_success "Pipeline completed successfully!"
        print_info "Results saved to: ${REPO_ROOT}/results"
        exit 0
    else
        print_error "Pipeline execution failed"
        print_info "Check logs in: ${REPO_ROOT}/logs"
        exit 1
    fi
}

# Run main function
main "$@"
