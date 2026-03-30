"""
Test suite for orchestration layer.

Tests for:
- Agentic tuner (locked/editable surfaces, budget management)
- Hyperparameter search
- Parallel processing
- Reproducibility infrastructure
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.orchestration.agentic_tuner import (
    AgenticTuner,
    AgenticTunerConfig,
    EditableSurface,
    ExperimentResult,
    LockedSurface,
)
from src.orchestration.hyperparameter_search import (
    HyperparameterSearchConfig,
    HyperparameterSearcher,
)
from src.orchestration.reproducibility import (
    DockerfileGenerator,
    EnvironmentSnapshot,
    ExperimentJournal,
    SeedManager,
    SingularityGenerator,
)


class TestAgenticTuner:
    """Test suite for AgenticTuner class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def locked_surface(self):
        """Create locked surface definition."""
        return LockedSurface(
            locked_files={"src/data/loader.py", "src/evaluation/metrics.py"},
            locked_functions={"load_data", "compute_auroc"},
            preprocessing_contract_hash="abc123",
        )

    @pytest.fixture
    def editable_surface(self):
        """Create editable surface definition."""
        return EditableSurface(
            editable_files={"configs/model_config.yaml"},
            editable_functions={"train_step", "optimizer_step"},
            editable_config_keys={"learning_rate", "batch_size", "hidden_dim"},
        )

    @pytest.fixture
    def config(self, temp_dir):
        """Create agentic tuner config."""
        return AgenticTunerConfig(
            metric="auroc",
            metric_mode="max",
            budget_type="trials",
            max_trials=5,
            experiment_name="test_tuning",
            log_dir=temp_dir / "experiments",
        )

    @pytest.fixture
    def tuner(self, config, locked_surface, editable_surface):
        """Create agentic tuner instance."""
        return AgenticTuner(config, locked_surface, editable_surface)

    def test_initialization(self, tuner, config):
        """Test AgenticTuner initialization."""
        assert tuner.config == config
        assert tuner.best_metric_value == -np.inf
        assert len(tuner.experiments) == 0

    def test_locked_surface_serialization(self, locked_surface):
        """Test LockedSurface serialization."""
        data = locked_surface.to_dict()
        restored = LockedSurface.from_dict(data)
        assert restored.locked_files == locked_surface.locked_files
        assert restored.locked_functions == locked_surface.locked_functions

    def test_editable_surface_serialization(self, editable_surface):
        """Test EditableSurface serialization."""
        data = editable_surface.to_dict()
        restored = EditableSurface.from_dict(data)
        assert restored.editable_files == editable_surface.editable_files
        assert restored.editable_functions == editable_surface.editable_functions

    def test_config_diff_computation(self, tuner):
        """Test configuration diff computation."""
        config_a = {"lr": 1e-4, "batch_size": 32, "hidden_dim": 256}
        config_b = {"lr": 2e-4, "batch_size": 32, "hidden_dim": 512}

        diff = tuner._compute_config_diff(config_a, config_b)

        assert "lr" in diff
        assert diff["lr"] == (1e-4, 2e-4)
        assert "hidden_dim" in diff
        assert "batch_size" not in diff

    def test_is_better_max_mode(self):
        """Test is_better with max mode."""
        config = AgenticTunerConfig(metric_mode="max")
        tuner = AgenticTuner(
            config,
            LockedSurface(),
            EditableSurface(),
        )

        assert tuner._is_better(0.9, 0.8) is True
        assert tuner._is_better(0.7, 0.8) is False

    def test_is_better_min_mode(self):
        """Test is_better with min mode."""
        config = AgenticTunerConfig(metric_mode="min")
        tuner = AgenticTuner(
            config,
            LockedSurface(),
            EditableSurface(),
        )

        assert tuner._is_better(0.1, 0.2) is True
        assert tuner._is_better(0.3, 0.2) is False

    def test_suspicious_improvement_detection(self, tuner):
        """Test detection of suspiciously large improvements."""
        tuner.best_metric_value = 0.8
        tuner.config.metric_threshold = 0.05

        assert tuner._is_suspicious_improvement(0.95) is True  # 0.15 > 0.05
        assert tuner._is_suspicious_improvement(0.82) is False  # 0.02 < 0.05

    def test_budget_exhaustion_trials(self, tuner):
        """Test budget exhaustion with trial count limit."""
        start_time = 0  # Arbitrary start

        assert tuner._is_budget_exhausted(start_time, 4) is False
        assert tuner._is_budget_exhausted(start_time, 5) is True
        assert tuner._is_budget_exhausted(start_time, 6) is True

    def test_preprocessing_hash_computation(self, tuner, temp_dir):
        """Test preprocessing contract hash computation."""
        # Create temporary file
        test_file = temp_dir / "test.py"
        test_file.write_text("print('hello')")

        tuner.locked.locked_files.add(str(test_file))
        hash_val = tuner._compute_preprocessing_hash()

        assert isinstance(hash_val, str)
        assert len(hash_val) == 64  # SHA256 hex string length

    def test_experiment_recording(self, tuner, temp_dir):
        """Test experiment result recording."""
        config = {"lr": 1e-4, "batch_size": 32}
        baseline_config = {"lr": 2e-4, "batch_size": 32}

        result = tuner._record_experiment(
            trial_id=1,
            candidate_config=config,
            baseline_config=baseline_config,
            metric_value=0.85,
            wall_clock_seconds=120.5,
        )

        assert result.trial_id == 1
        assert result.metric_value == 0.85
        assert result.wall_clock_seconds == 120.5
        assert len(tuner.experiments) == 1

        # Check that result was saved
        result_file = tuner.log_dir / "trial_0001.json"
        assert result_file.exists()

        # Verify saved content
        with open(result_file) as f:
            data = json.load(f)
        assert data["trial_id"] == 1
        assert data["metric_value"] == 0.85

    def test_candidate_generation(self, tuner):
        """Test candidate configuration generation."""
        baseline = {"lr": 1e-4, "batch_size": 32, "hidden_dim": 256}

        candidate = tuner._generate_candidate(baseline, None)

        assert isinstance(candidate, dict)
        # Some keys should have changed
        assert any(candidate[k] != baseline[k] for k in candidate.keys())


class TestHyperparameterSearcher:
    """Test suite for HyperparameterSearcher."""

    @pytest.fixture
    def config(self):
        """Create search config."""
        return HyperparameterSearchConfig(
            max_trials=10,
            num_samples=5,
            search_algorithm="random",
            metric="auroc",
            metric_mode="max",
        )

    @pytest.fixture
    def searcher(self, config):
        """Create searcher instance."""
        return HyperparameterSearcher(config)

    def test_initialization(self, searcher):
        """Test HyperparameterSearcher initialization."""
        assert len(searcher.search_spaces) > 0
        assert "abmil" in searcher.search_spaces
        assert "clam" in searcher.search_spaces
        assert "transmil" in searcher.search_spaces
        assert "dsmil" in searcher.search_spaces
        assert "fusion" in searcher.search_spaces

    def test_list_model_types(self, searcher):
        """Test listing available model types."""
        model_types = searcher.list_model_types()
        assert isinstance(model_types, list)
        assert len(model_types) > 0
        assert "abmil" in model_types

    def test_get_search_space(self, searcher):
        """Test getting search space for model."""
        space = searcher.get_search_space("abmil")
        assert isinstance(space, dict)
        assert "learning_rate" in space
        assert "dropout_rate" in space

    def test_unknown_model_type(self, searcher):
        """Test error on unknown model type."""
        with pytest.raises(ValueError):
            searcher.get_search_space("unknown_model")

    def test_search_space_keys_abmil(self, searcher):
        """Test ABMIL search space has expected keys."""
        space = searcher.get_search_space("abmil")
        expected_keys = {
            "learning_rate",
            "weight_decay",
            "dropout_rate",
            "hidden_dim",
            "batch_size",
        }
        assert expected_keys.issubset(set(space.keys()))

    def test_search_space_keys_fusion(self, searcher):
        """Test fusion search space has expected keys."""
        space = searcher.get_search_space("fusion")
        expected_keys = {
            "learning_rate",
            "fusion_method",
            "pathology_weight",
            "radiomics_weight",
        }
        assert expected_keys.issubset(set(space.keys()))


class TestReproducibility:
    """Test suite for reproducibility infrastructure."""

    def test_environment_snapshot_creation(self):
        """Test environment snapshot creation."""
        snapshot = EnvironmentSnapshot.create()

        assert snapshot.timestamp is not None
        assert snapshot.python_version is not None
        assert snapshot.platform_info is not None
        assert "system" in snapshot.platform_info
        assert "machine" in snapshot.platform_info

    def test_environment_snapshot_serialization(self):
        """Test snapshot serialization."""
        snapshot = EnvironmentSnapshot.create()
        data = snapshot.to_dict()

        assert isinstance(data, dict)
        assert "timestamp" in data
        assert "python_version" in data
        assert "platform_info" in data

    def test_environment_snapshot_saving(self, tmp_path):
        """Test saving snapshot to file."""
        snapshot = EnvironmentSnapshot.create()
        save_path = tmp_path / "snapshot.json"

        snapshot.save(save_path)

        assert save_path.exists()
        with open(save_path) as f:
            data = json.load(f)
        assert "timestamp" in data

    def test_seed_manager_set_seed(self):
        """Test setting random seed."""
        SeedManager.set_seed(42)

        # Verify reproducibility
        arr1 = np.random.rand(5)
        SeedManager.set_seed(42)
        arr2 = np.random.rand(5)

        np.testing.assert_array_equal(arr1, arr2)

    def test_seed_manager_get_config(self):
        """Test getting seed configuration."""
        config = SeedManager.get_seed_config(42)

        assert isinstance(config, dict)
        assert config["seed"] == 42
        assert "numpy_seed" in config
        assert "torch_seed" in config

    def test_dockerfile_generator(self, tmp_path):
        """Test Dockerfile generation."""
        snapshot = EnvironmentSnapshot.create()
        generator = DockerfileGenerator(snapshot)

        dockerfile_path = tmp_path / "Dockerfile"
        content = generator.generate(dockerfile_path)

        assert dockerfile_path.exists()
        assert "FROM" in content
        assert "RUN" in content
        assert "pip install" in content

    def test_singularity_generator(self, tmp_path):
        """Test Singularity definition generation."""
        snapshot = EnvironmentSnapshot.create()
        generator = SingularityGenerator(snapshot)

        def_path = tmp_path / "Singularity.def"
        content = generator.generate(def_path)

        assert def_path.exists()
        assert "Bootstrap:" in content
        assert "%post" in content
        assert "pip install" in content

    def test_experiment_journal_add_entry(self, tmp_path):
        """Test adding entries to experiment journal."""
        journal = ExperimentJournal(tmp_path)

        journal.add_entry(
            experiment_id="exp_001",
            model_type="abmil",
            config={"lr": 1e-4},
            metrics={"auroc": 0.85},
            git_hash="abc123def456",
            notes="Test experiment",
        )

        assert len(journal.entries) == 1
        entry = journal.entries[0]
        assert entry["experiment_id"] == "exp_001"
        assert entry["model_type"] == "abmil"

    def test_experiment_journal_save(self, tmp_path):
        """Test saving experiment journal."""
        journal = ExperimentJournal(tmp_path)

        journal.add_entry(
            experiment_id="exp_001",
            model_type="abmil",
            config={"lr": 1e-4},
            metrics={"auroc": 0.85},
        )

        journal.save()

        journal_file = tmp_path / "experiment_journal.json"
        assert journal_file.exists()

    def test_experiment_journal_markdown_report(self, tmp_path):
        """Test generating markdown report from journal."""
        journal = ExperimentJournal(tmp_path)

        journal.add_entry(
            experiment_id="exp_001",
            model_type="abmil",
            config={"lr": 1e-4},
            metrics={"auroc": 0.85},
            notes="Test experiment",
        )

        report = journal.generate_markdown_report()

        assert "# Experiment Journal" in report
        assert "exp_001" in report
        assert "abmil" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
