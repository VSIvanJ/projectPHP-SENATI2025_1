import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
"""
# Crear un conjunto de datos ficticio
# Tamaño en m²
size = np.array([50, 60, 80, 100, 120, 150, 180])
# Precio en miles de dólares
price = np.array([150, 180, 240, 300, 360, 450, 540])

# Convertir a DataFrame para visualización y análisis
data = pd.DataFrame({'Tamaño (m²)': size, 'Precio (mil USD)': price})

print("Datos:")
print(data)

# Función para entrenar un modelo de regresión lineal simple
def entrenar_modelo(x, y):
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerador = np.sum((x - x_mean) * (y - y_mean))
    denominador = np.sum((x - x_mean) ** 2)

    pendiente = numerador / denominador
    intercepto = y_mean - pendiente * x_mean

    return pendiente, intercepto

# Entrenar el modelo
pendiente, intercepto = entrenar_modelo(size, price)

print(f"\nModelo entrenado: Precio = {pendiente:.2f} * Tamaño + {intercepto:.2f}")

# Función para hacer predicciones
def predecir(tamaño):
    return pendiente * tamaño + intercepto

# Ejemplo de predicción
nuevo_tamaño = 210
prediccion = predecir(nuevo_tamaño)
print(f"\nPredicción: una casa de {nuevo_tamaño} m² costará aproximadamente ${prediccion:.2f} mil USD")

"""
import numpy as np
import pandas as pd

# Crear un conjunto de datos de diferentes objetos con una característica relevante
datos = {
    'Tipo': ['Casa', 'Casa', 'Auto', 'Auto', 'Laptop', 'Laptop'],
    'Característica': [100, 150, 120, 200, 8, 16],  # m², HP, GB RAM
    'Valor': [300, 450, 150, 250, 400, 800]         # Precio en mil USD
}

df = pd.DataFrame(datos)

print("Datos iniciales:")
print(df)

# Entrenar modelos diferentes para cada tipo
modelos = {}

def entrenar_modelo(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    pendiente = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    intercepto = y_mean - pendiente * x_mean
    return pendiente, intercepto

# Entrenar un modelo para cada tipo de objeto
for tipo in df['Tipo'].unique():
    datos_tipo = df[df['Tipo'] == tipo]
    x = datos_tipo['Característica'].values
    y = datos_tipo['Valor'].values
    pendiente, intercepto = entrenar_modelo(x, y)
    modelos[tipo] = (pendiente, intercepto)
    print(f"\nModelo para {tipo}: Valor = {pendiente:.2f} * Característica + {intercepto:.2f}")

# Función para predecir
def predecir(tipo, caracteristica):
    if tipo not in modelos:
        print(f"No hay modelo entrenado para '{tipo}'.")
        return None
    pendiente, intercepto = modelos[tipo]
    return pendiente * caracteristica + intercepto

# Ejemplo de predicción
tipo_objeto = 'Auto'
nueva_caracteristica = 180  # 180 HP
prediccion = predecir(tipo_objeto, nueva_caracteristica)
if prediccion is not None:
    print(f"\nPredicción: un {tipo_objeto} con {nueva_caracteristica} de característica tendrá un valor estimado de ${prediccion:.2f} mil USD")


# Visualizar modelos y predicciones
colores = {'Casa': 'blue', 'Auto': 'green', 'Laptop': 'orange'}
fig, axs = plt.subplots(1, len(modelos), figsize=(15, 5))

for i, tipo in enumerate(modelos):
    datos_tipo = df[df['Tipo'] == tipo]
    x = datos_tipo['Característica'].values
    y = datos_tipo['Valor'].values
    pendiente, intercepto = modelos[tipo]
    
    # Crear línea de regresión
    x_linea = np.linspace(min(x), max(x), 100)
    y_linea = pendiente * x_linea + intercepto
    
    # Dibujar
    axs[i].scatter(x, y, color=colores[tipo], label='Datos reales')
    axs[i].plot(x_linea, y_linea, color='black', label='Modelo lineal')
    
    # Añadir predicción si corresponde
    if tipo == tipo_objeto:
        y_pred = predecir(tipo, nueva_caracteristica)
        axs[i].scatter([nueva_caracteristica], [y_pred], color='red', label='Predicción', zorder=5)
        axs[i].annotate(f"${y_pred:.2f}", (nueva_caracteristica, y_pred), textcoords="offset points", xytext=(0,10), ha='center')
    
    axs[i].set_title(f"Modelo para {tipo}")
    axs[i].set_xlabel("Característica")
    axs[i].set_ylabel("Valor (mil USD)")
    axs[i].legend()

plt.tight_layout()
plt.show()
