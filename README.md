# Proyecto DQN CUDA - Guía Rápida

## 📁 Estructura del Proyecto

```
proyecto-final/
├── client/              # Archivos Python para la laptop
│   ├── client_cuda.py        # Cliente de entrenamiento
│   ├── client_inference.py   # Cliente de inferencia
│   └── robot_environment.py  # Entorno de simulación
├── jetson_files/        # Archivos CUDA para el Jetson
│   ├── dqn_server_cuda.cu        # Servidor entrenamiento
│   ├── dqn_inference_server.cu   # Servidor inferencia
│   ├── dqn_cuda.cu               # Implementación DQN
│   ├── device_query.cu           # Query CUDA
│   └── Makefile                  # Compilación
├── config/              # Configuración
│   └── config.ini            # Configuración centralizada
└── docs/                # Documentación
```

## ⚙️ Configuración

Edita `config/config.ini` para cambiar:

- **Tamaño del escenario**: `width`, `height` (default: 400x400)
- **Obstáculos**: `num_obstacles` (default: 5)
- **Steps máximos**: `max_steps` (default: 200)
- **Episodios de entrenamiento**: `episodes` (default: 200)
- **IP del Jetson**: `jetson_ip` (default: 172.22.234.49)

## 🚀 Uso

### 1️⃣ ENTRENAMIENTO

**En el Jetson:**

```bash
ssh -i ~/.ssh/id_jetson jetson@172.22.234.49
cd ~/dqn_cuda_project
./bin/dqn_server_cuda
```

**En tu Laptop:**

```bash
cd client
python3 client_cuda.py
```

**Parámetros opcionales:**

```bash
# Cambiar configuración sin editar config.ini
python3 client_cuda.py --episodes 300 --obstacles 8 --width 500 --height 500
```

Los modelos se guardan automáticamente en `~/dqn_cuda_project/models/` cada 50 episodios.

---

### 2️⃣ INFERENCIA (Usar modelo entrenado)

**En el Jetson:**

```bash
ssh -i ~/.ssh/id_jetson jetson@172.22.234.49
cd ~/dqn_cuda_project
./bin/dqn_inference_server models/dqn_model_final.bin
```

**En tu Laptop:**

```bash
cd client
python3 client_inference.py
```

**Parámetros opcionales:**

```bash
# Cambiar episodios y velocidad
python3 client_inference.py --episodes 20 --delay 0.1
```

---

## 📊 Parámetros Disponibles

### Cliente de Entrenamiento (client_cuda.py)

| Parámetro         | Descripción                | Default       |
| ----------------- | -------------------------- | ------------- |
| `--host`          | IP del Jetson              | 172.22.234.49 |
| `--port`          | Puerto                     | 5556          |
| `--episodes`      | Episodios de entrenamiento | 200           |
| `--save-interval` | Guardar cada N episodios   | 50            |
| `--width`         | Ancho del escenario        | 400           |
| `--height`        | Alto del escenario         | 400           |
| `--obstacles`     | Número de obstáculos       | 5             |
| `--max-steps`     | Steps máximos por episodio | 200           |
| `--no-render`     | Sin visualización          | -             |

**Ejemplo:**

```bash
python3 client_cuda.py --episodes 500 --width 300 --height 300 --obstacles 3
```

### Cliente de Inferencia (client_inference.py)

| Parámetro     | Descripción             | Default       |
| ------------- | ----------------------- | ------------- |
| `--host`      | IP del Jetson           | 172.22.234.49 |
| `--port`      | Puerto                  | 5557          |
| `--episodes`  | Episodios a ejecutar    | 10            |
| `--delay`     | Delay entre pasos (seg) | 0.05          |
| `--width`     | Ancho del escenario     | 400           |
| `--height`    | Alto del escenario      | 400           |
| `--obstacles` | Número de obstáculos    | 5             |
| `--max-steps` | Steps máximos           | 200           |
| `--no-render` | Sin visualización       | -             |

**Ejemplo:**

```bash
python3 client_inference.py --episodes 20 --delay 0.1
```

---

## 🔧 Setup Inicial

### En el Jetson (Primera vez)

1. **Copiar archivos:**

```bash
scp -i ~/.ssh/id_jetson -r jetson_files/* jetson@172.22.234.49:~/dqn_cuda_project/
```

2. **Crear carpeta de modelos:**

```bash
ssh -i ~/.ssh/id_jetson jetson@172.22.234.49 "mkdir -p ~/dqn_cuda_project/models"
```

3. **Compilar:**

```bash
ssh -i ~/.ssh/id_jetson jetson@172.22.234.49
cd ~/dqn_cuda_project
export PATH=/usr/local/cuda/bin:$PATH
make
```

### En tu Laptop

```bash
# Instalar dependencias
pip install pygame numpy
```

---

## 💡 Tips para Mejor Entrenamiento

1. **Escenario pequeño**: 300x300 o 400x400 funciona mejor
2. **Pocos obstáculos**: 3-5 obstáculos facilitan el aprendizaje
3. **Más episodios**: 300-500 episodios para mejor convergencia
4. **Evaluar progreso**: Usa inferencia cada 100 episodios para ver mejora

---

## 📁 Modelos Guardados

Los modelos se guardan en el Jetson en:

```
~/dqn_cuda_project/models/
  ├── dqn_model_ep50.bin
  ├── dqn_model_ep100.bin
  ├── dqn_model_ep150.bin
  └── dqn_model_final.bin
```

Para usar un modelo específico en inferencia:

```bash
./bin/dqn_inference_server models/dqn_model_ep100.bin
```

---

## 🔍 Verificar Estado

**Ver modelos en Jetson:**

```bash
ssh -i ~/.ssh/id_jetson jetson@172.22.234.49 "ls -lh ~/dqn_cuda_project/models/"
```

**Descargar modelo a laptop:**

```bash
scp -i ~/.ssh/id_jetson jetson@172.22.234.49:~/dqn_cuda_project/models/dqn_model_final.bin .
```

---

## ❓ Ayuda Rápida

**Cliente:**

```bash
python3 client_cuda.py --help
python3 client_inference.py --help
```

**Ver configuración actual:**

```bash
cat config/config.ini
```
