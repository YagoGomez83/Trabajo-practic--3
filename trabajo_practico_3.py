import numpy as np

# Función para entrenar la red con la regla de Hebb
def hebbian_learning(patterns):
    n = patterns.shape[1]  # Número de neuronas
    weights = np.zeros((n, n))

    for p in patterns:
        weights += np.outer(p, p)
    np.fill_diagonal(weights, 0)  # Evitar autoconexiones
    return weights / len(patterns)

# Función para entrenar usando la matriz pseudoinversa
def pseudoinverse_learning(patterns):
    pseudo_inv = np.linalg.pinv(patterns)  # Calculamos la pseudoinversa de los patrones
    return np.dot(patterns.T, pseudo_inv.T)  # Multiplicamos la transpuesta correctamente

# Actualización de la red (Hopfield)
def update(state, weights):
    return np.sign(np.dot(weights, state))

# Generar imagen simple de 10x10 píxeles
def generate_image():
    pattern = np.random.choice([-1, 1], size=(10, 10))
    return pattern.flatten()

# Prototipo principal
def main():
    # Crear patrones de entrenamiento
    patterns = np.array([generate_image() for _ in range(3)])  # Tres patrones de 100 neuronas

    # Entrenamos la red con Hebb
    hebb_weights = hebbian_learning(patterns)

    # Entrenamos con la pseudoinversa
    pseudo_weights = pseudoinverse_learning(patterns)

    # Imagen con ruido
    noisy_image = patterns[0].copy()
    noisy_image[::10] = -noisy_image[::10]  # Añadimos ruido a la imagen

    # Recuperación con Hebb
    recovered_image_hebb = update(noisy_image, hebb_weights)
    
    # Recuperación con pseudoinversa
    recovered_image_pseudo = update(noisy_image, pseudo_weights)

    # Imprimir los resultados
    print("Imagen original:")
    print(patterns[0].reshape(10, 10))

    print("\nImagen con ruido:")
    print(noisy_image.reshape(10, 10))

    print("\nImagen recuperada (Hebb):")
    print(recovered_image_hebb.reshape(10, 10))

    print("\nImagen recuperada (Pseudoinversa):")
    print(recovered_image_pseudo.reshape(10, 10))

if __name__ == "__main__":
    main()
