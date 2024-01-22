from fastai.vision.all import *
import gradio as gr

# Import untar_data
path = untar_data(URLs.PETS)/'images'


def is_BullTerrier(x):
    """Check if it's a Bull Terrier dog."""
    if "BULL_TERRIER" in x.upper():
        return x
    else:
        return False


# Define item transforms
item_tfms = Resize(192)

# Create ImageDataLoaders
dls = ImageDataLoaders.from_name_func('.',
                                      get_image_files(path), valid_pct=0.2,
                                      seed=42, label_func=is_BullTerrier,
                                      item_tfms=item_tfms)

# Create vision learner
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)
dls.show_batch()

# Use is_BullTerrier instead of greet in Gradio interface
iface = gr.Interface(fn=is_BullTerrier, inputs="text", outputs="text")
iface.launch()
