# Red de Hopfield para Reconocimiento de Patrones

Este proyecto implementa una **red neuronal de Hopfield** que se entrena para el reconocimiento de patrones binarios. Se utiliza tanto el **método de Hebb** como la **pseudoinversa** para almacenar y recuperar patrones. Las imágenes procesadas son de **10x10 píxeles** y contienen valores binarios (-1, 1). El objetivo es recuperar patrones distorsionados o con ruido.

## Requisitos

- **Python 3.x**
- **NumPy** (para manejo eficiente de matrices y operaciones matemáticas)

Para instalar NumPy:

```bash
pip install numpy
```

## Descripción del Proyecto

Este prototipo simula una red de Hopfield para **reconocimiento de patrones**. La red se entrena con tres patrones binarios aleatorios (10x10 píxeles) y luego intenta recuperar un patrón que ha sido distorsionado (añadiendo ruido). Se implementan dos métodos de entrenamiento:

1. **Regla de Hebb**: Ajusta los pesos entre las neuronas usando el producto externo de los patrones.
2. **Matriz Pseudoinversa**: Utiliza la pseudoinversa para calcular los pesos, permitiendo una recuperación más robusta cuando se trata de varios patrones.

### Funciones Principales:

- **`hebbian_learning(patterns)`**: Implementa la regla de Hebb para ajustar los pesos de la red.
- **`pseudoinverse_learning(patterns)`**: Utiliza la pseudoinversa para calcular los pesos entre las neuronas.
- **`update(state, weights)`**: Actualiza el estado de la red según los pesos entrenados.
- **`generate_image()`**: Genera una imagen aleatoria de 10x10 píxeles.

### Proceso:

1. **Generar Patrones**: Se generan tres imágenes aleatorias de 10x10 píxeles, representadas por valores binarios (-1, 1).
2. **Entrenamiento**:
   - Se entrena la red utilizando la regla de Hebb y la pseudoinversa.
3. **Añadir Ruido**: Se introduce ruido en uno de los patrones para distorsionarlo.
4. **Recuperación del Patrón**: Se utilizan los pesos entrenados para recuperar el patrón distorsionado, primero con Hebb y luego con la pseudoinversa.
5. **Impresión de Resultados**: Se imprimen la imagen original, la imagen distorsionada y las imágenes recuperadas con ambos métodos.

## Ejecución del Programa

Para ejecutar el programa, basta con ejecutar el archivo `.py` desde la terminal:

```bash
python trabajo_practico_3.py
```

### Ejemplo de Salida

```plaintext
Imagen original:
[[ 1 -1  1 -1  1 -1  1 -1  1 -1]
 [ 1 -1  1 -1  1 -1  1 -1  1 -1]
 ...]

Imagen con ruido:
[[ 1 -1  1 -1 -1 -1  1 -1  1 -1]
 [ 1 -1  1 -1  1  1  1 -1  1 -1]
 ...]

Imagen recuperada (Hebb):
[[ 1 -1  1 -1  1 -1  1 -1  1 -1]
 [ 1 -1  1 -1  1 -1  1 -1  1 -1]
 ...]

Imagen recuperada (Pseudoinversa):
[[ 1 -1  1 -1  1 -1  1 -1  1 -1]
 [ 1 -1  1 -1  1 -1  1 -1  1 -1]
 ...]
```

## Casos de Uso

### 1. **Reconocimiento de Imágenes Ruidosas**:

Este prototipo permite reconocer imágenes que han sido distorsionadas. El sistema puede tomar un patrón ruidoso y restaurarlo al original usando la red de Hopfield.

### 2. **Memoria Asociativa**:

Puedes usar la red de Hopfield como una **memoria asociativa**. Almacena patrones y permite que la red recupere una imagen original aunque esté incompleta o distorsionada.

### 3. **Comparación de Métodos de Aprendizaje**:

Este código te permite comparar la **regla de Hebb** (más simple pero menos precisa) y la **pseudoinversa** (más robusta, capaz de almacenar más patrones) para el entrenamiento de redes de Hopfield.

## Extensiones Posibles

1. **Aumentar el Tamaño de las Imágenes**: Este prototipo trabaja con imágenes de 10x10 píxeles, pero puedes ampliarlo a imágenes más grandes para problemas más complejos.
2. **Añadir Visualización Gráfica**: Podrías agregar una interfaz gráfica para ver las imágenes de manera más visual, en lugar de imprimir matrices de números.

3. **Más Patrones**: Explora cómo la red de Hopfield se comporta al almacenar y recuperar más patrones, aumentando el número de patrones en el conjunto de entrenamiento.

## Posibles Dificultades

1. **Capacidad Limitada**: La red de Hopfield tiene una capacidad limitada de almacenamiento de patrones, especialmente cuando se utiliza la regla de Hebb.
2. **Recuperación con Ruido**: La calidad de la recuperación de los patrones puede verse afectada cuando el ruido es significativo.

## Conclusiones

Este proyecto demuestra cómo una **red de Hopfield** puede ser usada para el reconocimiento de patrones, incluso con distorsiones o ruido. Al comparar los dos métodos de aprendizaje (Hebb y pseudoinversa), puedes ver cómo la pseudoinversa ofrece una mayor capacidad de almacenamiento y robustez en la recuperación de patrones.
