import numpy as np
import matplotlib.pyplot as plt

# 1. FUNCIONES DE MEMBRESÍA BASE ##############################################
def trimf(x, params):
    """Función de membresía triangular"""
    a, b, c = params
    y = np.zeros_like(x)
    idx1 = np.logical_and(a < x, x <= b)
    idx2 = np.logical_and(b < x, x < c)
    if a != b: y[idx1] = (x[idx1] - a) / (b - a)
    if b != c: y[idx2] = (c - x[idx2]) / (c - b)
    y[x == b] = 1.0
    return y

def trapmf(x, params):
    """Función de membresía trapezoidal"""
    a, b, c, d = params
    y = np.zeros_like(x)
    idx1 = np.logical_and(a < x, x < b)
    idx2 = np.logical_and(b <= x, x <= c)
    idx3 = np.logical_and(c < x, x < d)
    if a != b: y[idx1] = (x[idx1] - a) / (b - a)
    y[idx2] = 1.0
    if c != d: y[idx3] = (d - x[idx3]) / (d - c)
    return y

def gaussmf(x, params):
    """Función de membresía gaussiana"""
    sigma, c = params
    return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))

# 2. CREACIÓN DE 3 VARIABLES LINGÜÍSTICAS (3 o más términos c/u) ##############
# Definición de los universos de discurso
x_servicio = np.linspace(0, 10, 100)
x_comida = np.linspace(0, 10, 100)
x_propina = np.linspace(0, 30, 100)

# Variable 1: Servicio (Términos: Malo, Regular, Bueno, Excelente)
servicio_malo = gaussmf(x_servicio, [1.5, 0])
servicio_regular = gaussmf(x_servicio, [1.5, 3.3])
servicio_bueno = gaussmf(x_servicio, [1.5, 6.6])
servicio_excelente = gaussmf(x_servicio, [1.5, 10])

# Variable 2: Comida (Términos: Rancia, Pasable, Deliciosa)
comida_rancia = trapmf(x_comida, [0, 0, 1, 3])
comida_pasable = trimf(x_comida, [2, 5, 8])
comida_deliciosa = trapmf(x_comida, [7, 9, 10, 10])

# Variable 3: Propina (Términos: Baja, Mas o menos, Generosa)
propina_baja = trimf(x_propina, [0, 5, 10])
propina_masomenos = trimf(x_propina, [10, 15, 20])
propina_generosa = trimf(x_propina, [20, 25, 30])


# 3. OPERADORES DE CONJUNTOS DIFUSOS (Unión, Intersección, Complemento) #######
def union_difusa(A, B):
    """Implementa el operador OR (Máximo)"""
    return np.maximum(A, B)

def interseccion_difusa(A, B):
    """Implementa el operador AND (Mínimo)"""
    return np.minimum(A, B)

def complemento_difuso(A):
    """Implementa el operador NOT (Negación)"""
    return 1.0 - A


# 4. PROCESO DE COMPOSICIÓN (Max-Min) #########################################
def composicion_max_min(R, S):
    """Composición Max-Min estándar entre dos relaciones matriciales."""
    R = np.array(R)
    S = np.array(S)
    C = np.zeros((R.shape[0], S.shape[1]))
    for i in range(R.shape[0]):
        for j in range(S.shape[1]):
            # Max-Min
            C[i, j] = np.max(np.minimum(R[i, :], S[:, j]))
    return C

R = [[1, 0], 
     [0, 1]]

S = [[1, 0, 0], 
     [1, 0, 1]]

resultado_composicion = composicion_max_min(R, S)
print("--- Resultado de la Composición Max-Min ---")
print(resultado_composicion)
print("\n")


# 5. COMPOSICIÓN DEL ÚLTIMO EJEMPLO CON T-NORMA = PRODUCTO
# El ejemplo evalúa Servicio=10 y Comida=3
val_servicio = 10
val_comida = 3

# 5.1 Fusificación (Obtenemos los grados de membresía de la entrada crisp)[cite: 18, 19]
mu_serv_excelente = gaussmf(np.array([val_servicio]), [1.5, 10])[0] # ~1.0
mu_comida_deliciosa = trapmf(np.array([val_comida]), [7, 9, 10, 10])[0] # ~0.0

# 5.2 Evaluación de reglas con T-Norma de Producto.
# Regla 3: SI Servicio es excelente O Comida es deliciosa, ENTONCES Propina es generosa[cite: 41].
# Para el operador OR, la T-Conorma probabilística comúnmente asociada al producto es (A + B - A*B). 
# Sin embargo, si mantenemos OR como Máximo y solo cambiamos la T-Norma (Implicación) a Producto propuesto por Larsen[cite: 13]:
fuerza_disparo_r3 = np.maximum(mu_serv_excelente, mu_comida_deliciosa)

# Implicación por producto (Larsen)[cite: 13]: mu_A(x) * mu_B(y)
implicacion_r3_producto = fuerza_disparo_r3 * propina_generosa

print("--- Evaluación con T-Norma = Producto (Ejemplo de Propina) ---")
print(f"Entradas Crisp -> Servicio: {val_servicio}, Comida: {val_comida}")
print(f"Fuerza de disparo Regla 3 (Max): {fuerza_disparo_r3:.4f}")

# Defusificación (Centroide)
def defusificar_centroide(x, mfx):
    numerador = np.sum(mfx * x)
    denominador = np.sum(mfx)
    if denominador == 0: return 0
    return numerador / denominador

propina_resultado = defusificar_centroide(x_propina, implicacion_r3_producto)
print(f"Propina calculada (Centroide con Implicación por Producto): {propina_resultado:.2f}%")

# (Opcional) Visualización rápida del resultado
plt.plot(x_propina, propina_generosa, '--', label='Conjunto Generosa')
plt.fill_between(x_propina, 0, implicacion_r3_producto, alpha=0.5, label='Área Implicación (Producto)')
plt.axvline(propina_resultado, color='k', linestyle='--', label=f'Centroide: {propina_resultado:.2f}')
plt.title('Salida Defusificada con Implicación de Producto (Larsen)')
plt.legend()
plt.show()