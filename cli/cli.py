"""Command Line Interface for image processing operations."""

import click
from PIL import Image
from pathlib import Path
from mylib.operations import (
    predict_class,
    resize_image,
    convert_to_grayscale,
    get_image_info,
)


@click.group()
def cli():
    """Image processing CLI for MLOps Lab1."""
    pass


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
def predict(image_path):
    """Predict the class of an image (random for Lab1)."""
    try:
        img = Image.open(image_path)
        predicted_class = predict_class(img)
        click.echo(f"Predicted class: {predicted_class}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.argument("width", type=int)
@click.argument("height", type=int)
@click.option("--output", "-o", default=None, help="Output path for resized image")
def resize(image_path, width, height, output):
    """Resize an image to specified dimensions."""
    try:
        img = Image.open(image_path)
        resized_img = resize_image(img, width, height)

        if output:
            resized_img.save(output)
            click.echo(f"Image resized and saved to: {output}")
        else:
            output_path = (
                Path(image_path).stem
                + f"_resized_{width}x{height}"
                + Path(image_path).suffix
            )
            resized_img.save(output_path)
            click.echo(f"Image resized and saved to: {output_path}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output path for grayscale image")
def grayscale(image_path, output):
    """Convert an image to grayscale."""
    try:
        img = Image.open(image_path)
        gray_img = convert_to_grayscale(img)

        if output:
            gray_img.save(output)
            click.echo(f"Image converted to grayscale and saved to: {output}")
        else:
            output_path = Path(image_path).stem + "_grayscale" + Path(image_path).suffix
            gray_img.save(output_path)
            click.echo(f"Image converted to grayscale and saved to: {output_path}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
def info(image_path):
    """Get information about an image."""
    try:
        img = Image.open(image_path)
        image_info = get_image_info(img)
        click.echo("Image Information:")
        for key, value in image_info.items():
            click.echo(f"  {key}: {value}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


if __name__ == "__main__":
    cli()
