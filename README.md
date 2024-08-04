## Flux Integration for Jupyter-like Environments

This project provides a simple user interface, based on Jupyter notebook widgets, for running Flux in various Jupyter-like services. It has been tested on Colab and Lightning.ai. Testing on Kaggle is pending.

To run the Web User Interface (WUI), simply add the following code to a cell and execute it:

```python
!git clone https://github.com/ai-marat/flux_wui
!pip install -r flux_wui/requirements.txt
from flux_wui.main import setup_pipeline_and_widgets
setup_pipeline_and_widgets()
```
<img width="837" alt="image" src="https://github.com/user-attachments/assets/e499c72d-ad88-416e-8cf4-43e8492fed98">


### About the Model

For detailed information about the Flux model, visit the [Flux GitHub repository](https://github.com/black-forest-labs/flux).

### Additional Resources

- **Notebooks**: Explore more notebooks on [Patreon](https://www.patreon.com/marat_ai).
- **YouTube Channel**: Check out our [YouTube channel](https://www.youtube.com/@marat_ai) for tutorials and updates.




