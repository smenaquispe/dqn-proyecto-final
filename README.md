# 🤖 DQN Robot Navigation con CUDA

**Proyecto Final - Computación Paralela**

Sistema de navegación de robots utilizando Deep Q-Network (DQN) implementado en **CUDA C++** para entrenamiento acelerado en **NVIDIA Jetson Xavier**.

![Training Results](images/WhatsApp%20Image%202025-12-19%20at%201.35.59%20PM.jpeg)

---

## 📋 Descripción

Este proyecto implementa un agente de aprendizaje por refuerzo (Reinforcement Learning) que aprende a navegar un robot hacia un objetivo evitando obstáculos. La característica principal es que **todo el entrenamiento de la red neuronal se ejecuta en CUDA C++** en el Jetson Xavier, aprovechando la GPU para acelerar el proceso.

### Características Principales

- 🧠 **DQN en CUDA C++**: Forward pass y backpropagation implementados con kernels CUDA
- 🎮 **Simulación en Python/Pygame**: Entorno visual de navegación
- 🌐 **Arquitectura Distribuida**: Cliente (laptop) ↔ Servidor (Jetson) via TCP
- ⚡ **Entrenamiento en GPU**: Aceleración usando los cores CUDA del Xavier

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────┐     ┌─────────────────────────────────────┐
│            LAPTOP (CPU)             │     │         JETSON XAVIER (GPU)         │
│                                     │     │                                     │
│  ┌─────────────────────────────┐   │     │   ┌─────────────────────────────┐   │
│  │     robot_environment.py    │   │     │   │    dqn_server_cuda.cu       │   │
│  │  - Simulación física        │   │     │   │  - Red Neuronal DQN         │   │
│  │  - Sensores del robot       │   │     │   │  - Forward Pass (CUDA)      │   │
│  │  - Sistema de recompensas   │   │     │   │  - Backward Pass (CUDA)     │   │
│  └─────────────────────────────┘   │     │   │  - Replay Buffer            │   │
│              ▼                      │     │   └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │     │              ▲                      │
│  │      client_cuda.py         │◄──┼─────┼──────────────┘                      │
│  │  - Envía estados            │   │TCP  │                                     │
│  │  - Recibe acciones          │   │5556 │   Kernels CUDA:                     │
│  │  - Envía feedback           │───┼─────┼──►  - simple_forward_kernel         │
│  └─────────────────────────────┘   │     │     - backward_linear_kernel        │
│              ▼                      │     │     - relu_backward_kernel          │
│  ┌─────────────────────────────┐   │     │     - update_weights_kernel         │
│  │         Pygame              │   │     │     - compute_gradients_kernel      │
│  │    Visualización            │   │     │                                     │
│  └─────────────────────────────┘   │     │                                     │
└─────────────────────────────────────┘     └─────────────────────────────────────┘
```

---

## 📸 Resultados

### Entrenamiento en Progreso

| Episodios Iniciales | Entrenamiento Avanzado |
|---------------------|------------------------|
| ![Inicio](images/WhatsApp%20Image%202025-12-19%20at%201.34.07%20PM.jpeg) | ![Avanzado](images/WhatsApp%20Image%202025-12-19%20at%201.35.38%20PM.jpeg) |

### Servidor CUDA en Jetson Xavier

![Servidor](images/WhatsApp%20Image%202025-12-19%20at%201.35.50%20PM.jpeg)

### Resultados Finales

![Resultados](images/WhatsApp%20Image%202025-12-19%20at%201.35.59%20PM.jpeg)

---

## 🧠 Implementación DQN en CUDA

### Red Neuronal

```
Input (15) → Dense(128) → ReLU → Dense(128) → ReLU → Dense(64) → ReLU → Output (8)
```

| Capa | Entrada | Salida | Parámetros |
|------|---------|--------|------------|
| FC1  | 15      | 128    | 1,920 + 128 = 2,048 |
| FC2  | 128     | 128    | 16,384 + 128 = 16,512 |
| FC3  | 128     | 64     | 8,192 + 64 = 8,256 |
| FC4  | 64      | 8      | 512 + 8 = 520 |
| **Total** | | | **27,336 parámetros** |

### Kernels CUDA Implementados

```cpp
// Forward Pass
__global__ void simple_forward_kernel(float *output, float *input, float *W, float *b, ...);
__global__ void final_layer_kernel(float *output, float *input, float *W, float *b, ...);

// Backward Pass  
__global__ void relu_backward_kernel(float *d_input, float *d_output, float *input, int size);
__global__ void compute_gradients_kernel(float *d_W, float *d_b, float *d_out, float *input, ...);
__global__ void backward_linear_kernel(float *d_input, float *d_output, float *W, ...);

// Optimizer (SGD)
__global__ void update_weights_kernel(float *W, float *d_W, float lr, int size);
__global__ void update_bias_kernel(float *b, float *d_b, float lr, int size);
```

### Estado del Robot (15 dimensiones)

- Posición del robot (x, y) normalizada
- Posición del objetivo (x, y) normalizada
- Distancia al objetivo
- Distancia al obstáculo más cercano
- Ángulo hacia el objetivo
- 8 sensores de proximidad (raycast en 8 direcciones)

### Acciones (8 posibles)

0=Norte, 1=Sur, 2=Oeste, 3=Este, 4=NE, 5=NO, 6=SE, 7=SO

---

## 🚀 Instalación y Uso

### Requisitos

**Laptop:**
- Python 3.8+
- pygame, numpy

**Jetson Xavier:**
- JetPack 4.x o 5.x
- CUDA Toolkit
- nvcc compiler

### Instalación

```bash
# Clonar repositorio
git clone https://github.com/smenaquispe/dqn-proyecto-final.git
cd dqn-proyecto-final

# Instalar dependencias Python (laptop)
pip install pygame numpy

# Copiar archivos al Jetson
scp -r jetson_files/* jetson@<JETSON_IP>:~/dqn_cuda_project/

# Compilar en Jetson
ssh jetson@<JETSON_IP>
cd ~/dqn_cuda_project
make
```

### Ejecutar Entrenamiento

**Terminal 1 - Jetson (Servidor):**
```bash
./bin/dqn_server_cuda
```

**Terminal 2 - Laptop (Cliente):**
```bash
python3 client_cuda.py --host <JETSON_IP> --episodes 200
```

---

## 📊 Hiperparámetros

| Parámetro | Valor |
|-----------|-------|
| Learning Rate | 0.001 |
| Gamma (descuento) | 0.99 |
| Epsilon inicial | 1.0 |
| Epsilon final | 0.01 |
| Epsilon decay | 0.995 |
| Replay Buffer | 100,000 |
| Batch Size | 4 |

---

## 📁 Estructura del Proyecto

```
dqn-proyecto-final/
├── client_cuda.py           # Cliente de entrenamiento
├── robot_environment.py     # Entorno de simulación
├── jetson_files/
│   ├── dqn_server_cuda.cu   # Servidor con DQN CUDA
│   ├── dqn_cuda.cu          # Implementación DQN
│   ├── Makefile             # Compilación
│   └── ...
├── images/                  # Capturas de resultados
├── config/                  # Configuración
└── docs/                    # Documentación técnica
```

---

## 👥 Autores

- **LeoUNSA** - Desarrollo e implementación

## 📄 Licencia

Este proyecto es parte del curso de Computación Paralela - UNSA 2025.
