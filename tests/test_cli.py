"""Integration testing with the CLI."""

import os
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner
from PIL import Image

from cli.cli import cli


@pytest.fixture
def runner() -> CliRunner:
    """Create a Click CLI runner for testing.

    Returns:
        CliRunner: Isolated filesystem runner for invoking CLI commands.
    """
    return CliRunner()


@pytest.fixture
def temp_image() -> str:
    """Create a temporary 100x100 blue PNG image for testing.

    Yields:
        str: Path to the temporary image file.

    Cleanup:
        The file is removed after the fixture is torn down.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(tmp.name, "PNG")
        yield tmp.name

    if os.path.exists(tmp.name):
        os.remove(tmp.name)


def test_help() -> None:
    """Test that the CLI help message is displayed correctly."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "Show this message and exit." in result.output


def test_predict_command_success(runner: CliRunner, temp_image: str) -> None:
    """Test the 'predict' command with a valid image.

    Args:
        runner: CliRunner fixture.
        temp_image: Path to a valid test image.
    """
    result = runner.invoke(cli, ["predict", temp_image])

    assert result.exit_code == 0
    assert "Predicted class:" in result.output


def test_predict_command_invalid_path(runner: CliRunner) -> None:
    """Test the 'predict' command with a non-existent image path.

    Args:
        runner: CliRunner fixture.
    """
    result = runner.invoke(cli, ["predict", "nonexistent.png"])

    assert result.exit_code != 0


def test_resize_command_success(runner: CliRunner, temp_image: str) -> None:
    """Test the 'resize' command with valid dimensions.

    Args:
        runner: CliRunner fixture.
        temp_image: Path to a valid test image.
    """
    result = runner.invoke(cli, ["resize", temp_image, "50", "50"])

    assert result.exit_code == 0
    assert "Image resized and saved to:" in result.output


def test_resize_command_with_output(runner: CliRunner, temp_image: str) -> None:
    """Test the 'resize' command using the '--output' option.

    Args:
        runner: CliRunner fixture.
        temp_image: Path to a valid test image.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_out:
        output_path = tmp_out.name

    try:
        result = runner.invoke(
            cli, ["resize", temp_image, "50", "50", "--output", output_path]
        )
        assert result.exit_code == 0
        assert os.path.exists(output_path)

        with Image.open(output_path) as img:
            assert img.size == (50, 50)
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


def test_resize_command_invalid_dimensions(runner: CliRunner, temp_image: str) -> None:
    """Test the 'resize' command with invalid (non-numeric) dimensions.

    Args:
        runner: CliRunner fixture.
        temp_image: Path to a valid test image.
    """
    result = runner.invoke(cli, ["resize", temp_image, "invalid", "50"])

    assert result.exit_code != 0


def test_grayscale_command_success(runner: CliRunner, temp_image: str) -> None:
    """Test the 'grayscale' command on a valid image.

    Args:
        runner: CliRunner fixture.
        temp_image: Path to a valid test image.
    """
    result = runner.invoke(cli, ["grayscale", temp_image])

    assert result.exit_code == 0
    assert "Image converted to grayscale and saved to:" in result.output


def test_grayscale_command_with_output(runner: CliRunner, temp_image: str) -> None:
    """Test the 'grayscale' command with the '--output' option.

    Args:
        runner: CliRunner fixture.
        temp_image: Path to a valid test image.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_out:
        output_path = tmp_out.name

    try:
        result = runner.invoke(cli, ["grayscale", temp_image, "--output", output_path])
        assert result.exit_code == 0
        assert os.path.exists(output_path)

        with Image.open(output_path) as img:
            assert img.mode == "L"
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


def test_info_command_success(runner: CliRunner, temp_image: str) -> None:
    """Test the 'info' command displays image metadata correctly.

    Args:
        runner: CliRunner fixture.
        temp_image: Path to a valid test image.
    """
    result = runner.invoke(cli, ["info", temp_image])

    assert result.exit_code == 0
    assert "Image Information:" in result.output
    assert "size:" in result.output
    assert "width:" in result.output
    assert "height:" in result.output


def test_info_command_invalid_path(runner: CliRunner) -> None:
    """Test the 'info' command with a non-existent image path.

    Args:
        runner: CliRunner fixture.
    """
    result = runner.invoke(cli, ["info", "nonexistent.png"])

    assert result.exit_code != 0
