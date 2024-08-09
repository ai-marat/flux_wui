import torch
import accelerate
import sentencepiece
import uuid
from diffusers import FluxPipeline
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

def setup_pipeline_and_widgets():
    # Initialize the pipeline once
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe.vae.enable_tiling()
    pipe.enable_sequential_cpu_offload()
    
    clear_output()
    
    # Define the function to set the generator based on the random_seed checkbox
    def set_generator(random, seed_value):
        return torch.Generator("cpu").manual_seed(0) if random else torch.Generator("cpu").manual_seed(seed_value)


    html_widget = widgets.HTML(
    value=(
        "<i>Made by <a href='https://www.youtube.com/@marat_ai' target='_blank'>marat_ai</a>, "
        "<a href='https://www.patreon.com/marat_ai' target='_blank'>more_notebooks</a></i>"
    ),
    placeholder='Some HTML')

    # Define the prompt textarea
    prompt = widgets.Textarea(
        value='a cat',
        description='Prompt',
        disabled=False,
        layout=widgets.Layout(width='40%', height='100px')
    )

    # Define the width slider
    width = widgets.IntSlider(
        value=1024,
        min=8,
        max=2048,
        step=8,
        description='Width',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )

    # Define the height slider
    height = widgets.IntSlider(
        value=1024,
        min=8,
        max=2048,
        step=8,
        description='Height',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )

    # Define the number of inference steps input
    num_inference_steps = widgets.IntText(
        value=4,
        min=0,
        max=5,
        step=1,
        description='Steps',
        disabled=False
    )
    
    guidance_scale = widgets.IntText(
        value=0.0,
        min=0,
        max=10,
        step=0.5,
        description='SFG scale',
        disabled=False
    )

    # Define the seed input
    seed = widgets.IntText(
        value=1,
        min=0,
        max=9999999999999999999999999,
        step=1,
        description='Seed',
        disabled=False
    )

    # Define the random seed checkbox
    random_seed = widgets.Checkbox(
        value=False,
        description='Random Seed',
        disabled=False,
        indent=False
    )

    # Define the generate button
    generate_button = widgets.Button(
        description='Generate',
        disabled=False,
        button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Generate image',
        icon='check'  # (FontAwesome names without the `fa-` prefix)
    )

    # Define an output widget for the image
    output = widgets.Output()

    # Define the function to generate and display the image
    def generate_image(button):
        with output:
            clear_output(wait=True)
            generator = set_generator(random_seed.value, seed.value)
            image = pipe(
                prompt=prompt.value, 
                num_inference_steps=num_inference_steps.value, 
                guidance_scale=guidance_scale.value, 
                generator=generator,
                width=width.value, 
                height=height.value
            ).images[0]
            uid = uuid.uuid4()
            image.save(f"{uid}.png")

            plt.imshow(image)
            plt.axis('off')
            plt.show()

    # Bind the function to the generate button click event
    generate_button.on_click(generate_image)

    # Display the widgets and output
    display(html_widget, prompt, num_inference_steps, guidance_scale, width, height, seed, random_seed, generate_button, output)

