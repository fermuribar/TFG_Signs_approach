# TFG Signs Approach - Signal Processing for Heart Rate Detection

Este repositorio forma parte del Trabajo Fin de Grado (TFG) titulado **"Dispositivo para la detección de parámetros fisiológicos sin contacto mediante técnicas de visión artificial"**, desarrollado por Fernando Muriano Barbosa.

## Acerca del Proyecto

El objetivo de este proyecto es implementar un enfoque basado en el procesamiento clásico de señales para detectar la frecuencia cardíaca. Este método emplea técnicas como la Transformada Rápida de Fourier (FFT) y el Análisis de Componentes Independientes (ICA), aplicados a datos obtenidos mediante regiones de interés (ROI) extraídas de videos.

## Contenido del Repositorio

El repositorio incluye los siguientes componentes:

1. **`signal_processing.py`**: Script principal para procesar señales y calcular la frecuencia cardíaca.
2. **`roi_detection.py`**: Código para la detección de regiones de interés (frente y mejillas) a partir de videos.
3. **`data_analysis.ipynb`**: Notebook para realizar análisis detallado de las señales procesadas.
4. **Pruebas**: Scripts de prueba para evaluar y validar el rendimiento del sistema en diversas condiciones.

## Cómo Ejecutar el Proyecto

1. **Clona el repositorio**:
   ```bash
   git clone https://github.com/fermuribar/TFG_Signs_approach.git
   cd TFG_Signs_approach
   ```

2. **Instala las dependencias necesarias**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecuta el script principal**:
   ```bash
   python rppg.py
   ```

## Tecnologías y Herramientas Utilizadas

- **Python**: Lenguaje principal del proyecto.
- **OpenCV**: Para el procesamiento de imágenes.
- **NumPy y SciPy**: Para análisis y filtrado de señales.
- **Tkinter**: Desarrollo de una interfaz gráfica para la interacción del usuario.
- **Matplotlib**: Visualización de datos.

## Licencia

Este proyecto está bajo la Licencia MIT. Puedes usar, modificar y distribuir el código siempre que incluyas una copia de la licencia original. Consulta el archivo `LICENSE` para más detalles.

## Contribuciones

Contribuciones, sugerencias y mejoras son bienvenidas. Si tienes una idea o encuentras algún problema, por favor abre un issue o envía un pull request.

---

**Autor**: Fernando Muriano Barbosa  
**Parte del Trabajo Fin de Grado en la Universidad de Granada.**
```
