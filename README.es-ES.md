

## Integración de Flux para Entornos Tipo Jupyter

Este proyecto proporciona una interfaz de usuario sencilla, basada en widgets de cuadernos Jupyter, para ejecutar Flux en diversos servicios tipo Jupyter. Ha sido probado en Colab y Lightning.ai. Las pruebas en Kaggle están pendientes.

Para ejecutar la Interfaz de Usuario Web (WUI), simplemente agregue el siguiente código a una celda y ejecútelo:

```python
!git clone https://github.com/ai-marat/flux_wui
!pip install -r flux_wui/requirements.txt
from flux_wui.main import setup_pipeline_and_widgets
setup_pipeline_and_widgets()
```
<img width="837" alt="image" src="https://github.com/user-attachments/assets/e499c72d-ad88-416e-8cf4-43e8492fed98">


### Acerca del Modelo

Para obtener información detallada sobre el modelo Flux, visite el [repositorio de Flux en GitHub](https://github.com/black-forest-labs/flux).

### Recursos Adicionales

- **Cuadernos**: Explore más cuadernos en [Patreon](https://www.patreon.com/marat_ai).
- **Canal de YouTube**: Visite nuestro [canal de YouTube](https://www.youtube.com/@marat_ai) para ver tutoriales y actualizaciones.
