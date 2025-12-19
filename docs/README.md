# Proyecto DQN con CUDA - Robot Navigation

Este proyecto implementa un algoritmo DQN (Deep Q-Network) usando CUDA para entrenar un robot que navega en un entorno con obstáculos.

## 🏗️ Arquitectura del Proyecto

```
┌─────────────┐                    ┌──────────────────┐
│   LAPTOP    │                    │  JETSON XAVIER   │
│             │                    │                  │
│  Pygame     │  ── Estados ────>  │  CUDA Server    │
│  Environment│                    │  (DQN Training) │
│             │  <── Acciones ──   │                  │
└─────────────┘                    └──────────────────┘
```

### Componentes:

- **Laptop**:

  - `robot_environment.py`: Entorno de simulación con Pygame
  - `client_cuda.py`: Cliente que se conecta al servidor CUDA
  - Muestra visualización del entrenamiento en tiempo real

- **Jetson Xavier** (IP: 172.22.234.49):
  - `dqn_server_cuda.cu`: Servidor que ejecuta DQN con CUDA
  - Procesa estados y devuelve acciones
  - Entrena la red neuronal con GPU

## 🚀 Configuración Inicial

### Prerequisitos

#### En la Laptop:

```bash
# Python 3.8+
sudo apt-get install python3 python3-pip

# Pygame
pip install pygame numpy
```

#### En el Jetson Xavier:

```bash
# CUDA Toolkit (usualmente pre-instalado)
nvcc --version

# Si no está instalado:
sudo apt-get install cuda-toolkit-11-4

# Build tools
sudo apt-get install build-essential
```

### Configurar SSH

En tu laptop, asegúrate de tener configurada la clave SSH:

```bash
# Verificar que existe
ls -la ~/.ssh/id_jetson

# Si no existe, generar una
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_jetson

# Copiar al Jetson
ssh-copy-id -i ~/.ssh/id_jetson jetson@172.22.234.49

# Probar conexión
ssh -i ~/.ssh/id_jetson jetson@172.22.234.49
```

## 📦 Instalación y Deployment

### Opción 1: Deployment Automático (Recomendado)

```bash
# Hacer ejecutables los scripts
chmod +x *.sh

# Deployar todo al Jetson
./quick_deploy.sh
```

Este script:

1. Copia todos los archivos al Jetson
2. Compila el código CUDA
3. Pregunta si deseas iniciar el servidor

### Opción 2: Deployment Manual

```bash
# 1. Hacer ejecutable
chmod +x deploy_cuda.sh

# 2. Deployar
./deploy_cuda.sh 172.22.234.49

# 3. SSH al Jetson
ssh -i ~/.ssh/id_jetson jetson@172.22.234.49

# 4. Compilar
cd ~/dqn_cuda_project
make

# 5. Iniciar servidor
./bin/dqn_server_cuda
```

## 🎮 Uso del Sistema

### Fase 1: Entrenamiento

#### Paso 1: Iniciar el servidor en el Jetson

En una terminal SSH conectada al Jetson:

```bash
ssh -i ~/.ssh/id_jetson jetson@172.22.234.49
cd ~/dqn_cuda_project
./bin/dqn_server_cuda
```

Deberías ver:

```
==============================================
   DQN Server with CUDA C
   Jetson Xavier Implementation
==============================================

CUDA Device: Xavier
Server listening on port 5556
Waiting for client connection...
```

#### Paso 2: Iniciar el cliente en la Laptop

En otra terminal en tu laptop:

```bash
# Opción A: Script automático
./run_training.sh

# Opción B: Manual
python3 client_cuda.py --host 172.22.234.49 --episodes 100
```

Verás una ventana de Pygame con:

- Robot (azul)
- Objetivo (verde)
- Obstáculos (negro)
- Información del entrenamiento

### Fase 2: Prueba/Inferencia

Para modo de solo inferencia (sin entrenamiento):

```bash
# En el Jetson, el servidor sigue igual
./bin/dqn_server_cuda

# En la laptop, usar modo test
python3 client_cuda.py --host 172.22.234.49 --episodes 10 --test
```

## 🔧 Scripts Útiles

| Script               | Descripción                            |
| -------------------- | -------------------------------------- |
| `quick_deploy.sh`    | Deploy rápido al Jetson                |
| `test_connection.sh` | Verificar conectividad                 |
| `run_training.sh`    | Iniciar entrenamiento desde laptop     |
| `start_server.sh`    | Script para iniciar servidor en Jetson |

## 📊 Monitoreo

### Ver logs del servidor

El servidor muestra:

- Dispositivo CUDA usado
- Requests procesados
- Epsilon actual
- Episodios completados

### Métricas del cliente

La ventana de Pygame muestra:

- Episode actual
- Steps en el episodio
- Reward acumulado
- Distancia al objetivo

## 🐛 Solución de Problemas

### Error: "Connection refused"

```bash
# 1. Verificar que el servidor está corriendo
ssh -i ~/.ssh/id_jetson jetson@172.22.234.49 "ps aux | grep dqn_server"

# 2. Verificar firewall en Jetson
ssh -i ~/.ssh/id_jetson jetson@172.22.234.49 "sudo ufw status"

# 3. Verificar puerto
./test_connection.sh
```

### Error: "nvcc not found"

```bash
# En el Jetson
sudo apt-get update
sudo apt-get install cuda-toolkit-11-4

# Agregar al PATH en ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Error: "pygame not installed"

```bash
# En la laptop
pip install pygame numpy

# O con conda
conda install pygame numpy
```

### Compilación falla en Jetson

```bash
# Verificar arquitectura
cd ~/dqn_cuda_project

# Jetson Xavier usa sm_72
# Si es Jetson Orin, cambiar en Makefile:
# NVCC_FLAGS = -O3 -arch=sm_87 -lcurand

make clean
make
```

## 📝 Archivos del Proyecto

```
proyecto-final/
├── client_cuda.py          # Cliente Python (LAPTOP)
├── robot_environment.py    # Entorno Pygame (LAPTOP)
├── dqn_server_cuda.cu      # Servidor CUDA (JETSON)
├── dqn_cuda.cu             # Implementación DQN completa
├── device_query.cu         # Query de dispositivo CUDA
├── Makefile                # Compilación
├── deploy_cuda.sh          # Script de deployment
├── quick_deploy.sh         # Deploy rápido
├── run_training.sh         # Iniciar entrenamiento
├── test_connection.sh      # Test de conexión
└── README.md               # Este archivo
```

## 🎯 Parámetros de Entrenamiento

Puedes modificar en `client_cuda.py`:

- `--episodes`: Número de episodios (default: 100)
- `--port`: Puerto del servidor (default: 5556)
- `--no-render`: Desactivar visualización

En `robot_environment.py`:

- `width/height`: Tamaño del entorno
- `grid_size`: Tamaño de celdas
- `max_steps`: Steps máximos por episodio

## 📈 Flujo de Datos

1. **Estado del entorno** (15 valores):

   - Posición robot (x, y)
   - Posición objetivo (x, y)
   - Distancia al objetivo
   - Distancia a obstáculo más cercano
   - Ángulo al objetivo
   - 8 sensores de proximidad

2. **Comunicación**:

   - Cliente → Servidor: `STATE:0.5,0.3,...`
   - Servidor → Cliente: `2,0.95` (acción, epsilon)

3. **Acciones** (8 direcciones):
   - 0: Arriba
   - 1: Abajo
   - 2: Izquierda
   - 3: Derecha
   - 4-7: Diagonales

## 🔐 Información de Conexión

- **Jetson IP**: 172.22.234.49
- **Usuario**: jetson
- **Puerto servidor**: 5556
- **SSH Key**: ~/.ssh/id_jetson

## 📚 Referencias

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Jetson Xavier Documentation](https://developer.nvidia.com/embedded/jetson-xavier)
- [DQN Paper](https://arxiv.org/abs/1312.5602)

## 👤 Autor

UNSA - Robótica - Proyecto Final

## 📄 Licencia

Este proyecto es para uso académico.
