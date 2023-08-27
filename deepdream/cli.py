import os
from pathlib import Path
import click
from PIL import Image
import logging
from .dream import DreamMaker, make_dream
from .models import VGG16

# Configure the logger
logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(message)s',
    level=logging.INFO,
)

@click.command()
@click.argument("target", type=click.Path(exists=True))
@click.argument("style", type=click.Path(exists=True))
@click.argument("out", type=click.Path())
@click.option(
    "-n", "--n-epochs", 
    default=5, show_default=True, 
    help="Number of epochs to run when making the dream image"
)
def dream(
        target: os.PathLike,
        style: os.PathLike,
        out: os.PathLike,
        n_epochs: int
    ):
    
    target_img_filepath = Path(target)
    style_img_filepath = Path(style)
    dream_img_filepath = Path(out)
    
    target_img = Image.open(target_img_filepath)
    style_img = Image.open(style_img_filepath)
    
    # Make dream image
    logging.info(f"Begine making the dream image")
    dream_img = make_dream(
        target_img=target_img,
        style_img=style_img,
        dream_maker=DreamMaker(
            model=VGG16().watch_layers({
                'relu1_2',
                'relu2_2',
                'relu3_3',
                'relu4_2', 'relu4_3',
                'relu5_3'
            })
        ),
        n_epochs=n_epochs
    )
    
    # Save dream image
    with open(dream_img_filepath, "w") as f:
        dream_img.save(f)
        logging.info(f"Dream image is saved at {str(dream_img_filepath)}")
