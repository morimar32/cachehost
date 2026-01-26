"""Tests for Config and CLI argument parsing."""

import os
import tempfile

import pytest

from cachehost.config import Config, parse_args


class TestConfigDefaults:
    """Tests for Config dataclass defaults."""

    def test_required_model_path(self, temp_model_file):
        """Test that model_path is required."""
        config = Config(model_path=temp_model_file)
        assert config.model_path == temp_model_file

    def test_default_n_ctx(self, temp_model_file):
        """Test default context window size."""
        config = Config(model_path=temp_model_file)
        assert config.n_ctx == 4096

    def test_default_n_threads(self, temp_model_file):
        """Test default thread count."""
        config = Config(model_path=temp_model_file)
        # Should be CPU count or 4
        assert config.n_threads >= 1

    def test_default_sampling_params(self, temp_model_file):
        """Test default sampling parameters."""
        config = Config(model_path=temp_model_file)
        assert config.top_k == 20
        assert config.repeat_penalty == 1.0
        assert config.temperature == 0.7
        assert config.min_p == 0.05
        assert config.seed == 3407

    def test_default_server_settings(self, temp_model_file):
        """Test default server settings."""
        config = Config(model_path=temp_model_file)
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.timeout == 300

    def test_default_backend_and_api(self, temp_model_file):
        """Test default backend and API format."""
        config = Config(model_path=temp_model_file)
        assert config.backend == "auto"
        assert config.api_format == "openai"

    def test_default_logging(self, temp_model_file):
        """Test default logging settings."""
        config = Config(model_path=temp_model_file)
        assert config.log_level == "INFO"
        assert config.log_file is None

    def test_default_cache_dir(self, temp_model_file):
        """Test default cache directory."""
        config = Config(model_path=temp_model_file)
        assert config.cache_dir == "./cache"


class TestConfigModelName:
    """Tests for the model_name property."""

    def test_extracts_filename_without_extension(self, temp_dir):
        """Test that model_name extracts filename without extension."""
        model_path = os.path.join(temp_dir, "my_model.gguf")
        with open(model_path, "wb") as f:
            f.write(b"fake")

        config = Config(model_path=model_path)
        assert config.model_name == "my_model"

    def test_replaces_dots_with_underscores(self, temp_dir):
        """Test that dots in filename are replaced with underscores."""
        model_path = os.path.join(temp_dir, "model.v1.2.gguf")
        with open(model_path, "wb") as f:
            f.write(b"fake")

        config = Config(model_path=model_path)
        assert config.model_name == "model_v1_2"

    def test_handles_path_with_directories(self, temp_dir):
        """Test that only the filename is used, not the full path."""
        subdir = os.path.join(temp_dir, "models", "llama")
        os.makedirs(subdir, exist_ok=True)
        model_path = os.path.join(subdir, "my_model.gguf")
        with open(model_path, "wb") as f:
            f.write(b"fake")

        config = Config(model_path=model_path)
        assert config.model_name == "my_model"

    def test_handles_no_extension(self, temp_dir):
        """Test model name when there's no extension."""
        model_path = os.path.join(temp_dir, "model_no_ext")
        with open(model_path, "wb") as f:
            f.write(b"fake")

        config = Config(model_path=model_path)
        assert config.model_name == "model_no_ext"


class TestParseArgs:
    """Tests for parse_args function."""

    def test_minimal_args(self, temp_model_file):
        """Test parsing with only required arguments."""
        args = ["-m", temp_model_file]
        config = parse_args(args)

        assert config.model_path == temp_model_file
        # All defaults should be set
        assert config.n_ctx == 4096
        assert config.backend == "auto"

    def test_long_form_model_path(self, temp_model_file):
        """Test --model-path long form."""
        args = ["--model-path", temp_model_file]
        config = parse_args(args)

        assert config.model_path == temp_model_file

    def test_all_options(self, temp_model_file, temp_dir):
        """Test parsing all available options."""
        log_file = os.path.join(temp_dir, "test.log")
        cache_dir = os.path.join(temp_dir, "cache")

        args = [
            "-m", temp_model_file,
            "--n-ctx", "8192",
            "--n-threads", "8",
            "--top-k", "50",
            "--repeat-penalty", "1.2",
            "--temperature", "0.5",
            "--min-p", "0.1",
            "--seed", "42",
            "--host", "127.0.0.1",
            "--port", "9000",
            "--timeout", "600",
            "--backend", "llama_cpp",
            "--api-format", "anthropic",
            "--log-level", "DEBUG",
            "--log-file", log_file,
            "--cache-dir", cache_dir,
        ]
        config = parse_args(args)

        assert config.model_path == temp_model_file
        assert config.n_ctx == 8192
        assert config.n_threads == 8
        assert config.top_k == 50
        assert config.repeat_penalty == 1.2
        assert config.temperature == 0.5
        assert config.min_p == 0.1
        assert config.seed == 42
        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.timeout == 600
        assert config.backend == "llama_cpp"
        assert config.api_format == "anthropic"
        assert config.log_level == "DEBUG"
        assert config.log_file == log_file
        assert config.cache_dir == cache_dir

    def test_backend_choices(self, temp_model_file):
        """Test that backend only accepts valid choices."""
        for backend in ["auto", "llama_cpp", "mlx"]:
            args = ["-m", temp_model_file, "--backend", backend]
            config = parse_args(args)
            assert config.backend == backend

    def test_api_format_choices(self, temp_model_file):
        """Test that api-format only accepts valid choices."""
        for fmt in ["openai", "anthropic"]:
            args = ["-m", temp_model_file, "--api-format", fmt]
            config = parse_args(args)
            assert config.api_format == fmt

    def test_log_level_choices(self, temp_model_file):
        """Test that log-level only accepts valid choices."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            args = ["-m", temp_model_file, "--log-level", level]
            config = parse_args(args)
            assert config.log_level == level

    def test_missing_model_path_raises(self):
        """Test that missing model path raises an error."""
        with pytest.raises(SystemExit):
            parse_args([])

    def test_nonexistent_model_path_raises(self, temp_dir):
        """Test that non-existent model path raises an error."""
        fake_path = os.path.join(temp_dir, "nonexistent.gguf")
        with pytest.raises(SystemExit):
            parse_args(["-m", fake_path])

    def test_invalid_backend_raises(self, temp_model_file):
        """Test that invalid backend raises an error."""
        with pytest.raises(SystemExit):
            parse_args(["-m", temp_model_file, "--backend", "invalid"])

    def test_invalid_api_format_raises(self, temp_model_file):
        """Test that invalid api-format raises an error."""
        with pytest.raises(SystemExit):
            parse_args(["-m", temp_model_file, "--api-format", "invalid"])
