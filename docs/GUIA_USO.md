# 🚀 GUÍA DE USO RÁPIDO - DQN CUDA Project

## ⚡ INICIO RÁPIDO (3 pasos)

### 1️⃣ DEPLOYAR al Jetson

```bash
./quick_deploy.sh
```

Esto copia y compila todo en el Jetson automáticamente.

### 2️⃣ INICIAR SERVIDOR en Jetson

En una terminal SSH al Jetson:

```bash
ssh -i ~/.ssh/id_jetson jetson@172.22.234.49
cd ~/dqn_cuda_project
./bin/dqn_server_cuda
```

### 3️⃣ ENTRENAR desde Laptop

En tu laptop:

```bash
./run_training.sh
```

¡Listo! Deberías ver la ventana de Pygame con el robot moviéndose.

---

## 📋 INSTRUCCIONES DETALLADAS

### Pre-requisitos

#### En tu Laptop:

```bash
# Instalar dependencias Python
pip install pygame numpy

# Verificar instalación
python3 -c "import pygame; print('pygame OK')"
```

#### Verificar conexión al Jetson:

```bash
./test_connection.sh
```

Deberías ver:

```
✓ SSH connection OK
✓ CUDA toolkit found
```

---

## 🔄 WORKFLOW COMPLETO

### Paso 1: Primera vez - Setup inicial

```bash
# En tu laptop, desde la carpeta del proyecto

# 1. Verificar conexión
./test_connection.sh

# 2. Deployar código al Jetson
./quick_deploy.sh
```

Espera a que compile. Verás:

```
✓ Compilation successful!
```

### Paso 2: Iniciar el servidor CUDA

**Opción A - Desde el script de deploy:**
Cuando termine el deploy, responde "y" cuando pregunte:

```
Start CUDA server now? (y/n) y
```

**Opción B - Manual:**

```bash
# Conectar por SSH
ssh -i ~/.ssh/id_jetson jetson@172.22.234.49

# Ir al directorio
cd ~/dqn_cuda_project

# Iniciar servidor
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

**⚠️ NO CIERRES ESTA TERMINAL - Déjala corriendo**

### Paso 3: Entrenar el robot

Abre una **NUEVA TERMINAL** en tu laptop:

```bash
# Navegar al proyecto
cd /home/smenaq/Documents/UNSA/robotica/proyecto-final

# Iniciar entrenamiento
./run_training.sh
```

O manualmente:

```bash
python3 client_cuda.py --host 172.22.234.49 --episodes 100
```

### ¿Qué deberías ver?

**En la terminal del servidor (Jetson):**

```
✓ Client connected from 172.22.X.X
Ready to process requests...
Processed 100 requests (epsilon: 0.9950)
=== Episode 1 completed ===
```

**En tu laptop:**

- Una ventana de Pygame mostrando:
  - Robot (cuadrado azul)
  - Objetivo (cuadrado verde)
  - Obstáculos (cuadrados negros)
  - Info: Episode, Steps, Reward, Distance

**En la terminal del cliente (laptop):**

```
✓ Connected to CUDA server at 172.22.234.49:5556
=== Training with CUDA Server ===

Episode 10/100
  Reward: 25.40
  Steps: 156
  Epsilon: 0.9512
```

---

## 🎮 CONTROLES Y OPCIONES

### Durante el entrenamiento:

- **Q**: Salir del entrenamiento
- **Cerrar ventana**: Terminar

### Opciones del cliente:

```bash
# Entrenar 200 episodios
python3 client_cuda.py --host 172.22.234.49 --episodes 200

# Sin visualización (más rápido)
python3 client_cuda.py --host 172.22.234.49 --episodes 100 --no-render

# Solo probar conexión
python3 client_cuda.py --host 172.22.234.49 --test

# Usar otro puerto
python3 client_cuda.py --host 172.22.234.49 --port 5557
```

---

## 🔄 WORKFLOW DIARIO

Una vez todo está configurado, para sesiones posteriores:

### Terminal 1 (SSH al Jetson):

```bash
ssh -i ~/.ssh/id_jetson jetson@172.22.234.49
cd ~/dqn_cuda_project
./bin/dqn_server_cuda
```

### Terminal 2 (En tu Laptop):

```bash
cd /home/smenaq/Documents/UNSA/robotica/proyecto-final
./run_training.sh
```

---

## 🛠️ MANTENIMIENTO

### Recompilar en el Jetson

Si modificas el código CUDA:

```bash
# Desde tu laptop
./quick_deploy.sh

# O manualmente en el Jetson
ssh -i ~/.ssh/id_jetson jetson@172.22.234.49
cd ~/dqn_cuda_project
make clean
make
```

### Ver logs/status del servidor

```bash
# Verificar si está corriendo
ssh -i ~/.ssh/id_jetson jetson@172.22.234.49 "ps aux | grep dqn_server"

# Matar servidor si está colgado
ssh -i ~/.ssh/id_jetson jetson@172.22.234.49 "pkill dqn_server"
```

### Limpiar y reinstalar

```bash
# En el Jetson
ssh -i ~/.ssh/id_jetson jetson@172.22.234.49
cd ~/dqn_cuda_project
make clean
rm -rf bin build

# Desde laptop, re-deployar
./quick_deploy.sh
```

---

## ❌ SOLUCIÓN DE PROBLEMAS

### Error: "Connection refused"

**Causa**: El servidor no está corriendo.

**Solución**:

```bash
# Verificar en Jetson
ssh -i ~/.ssh/id_jetson jetson@172.22.234.49 "cd ~/dqn_cuda_project && ./bin/dqn_server_cuda"
```

### Error: "pygame not found"

**Solución**:

```bash
pip install pygame numpy
```

### Error: "Permission denied" al ejecutar scripts

**Solución**:

```bash
chmod +x *.sh
```

### Error: "nvcc not found" en Jetson

**Solución**:

```bash
ssh -i ~/.ssh/id_jetson jetson@172.22.234.49
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### Servidor no responde

**Solución**:

```bash
# Matar proceso
ssh -i ~/.ssh/id_jetson jetson@172.22.234.49 "pkill dqn_server"

# Reiniciar
ssh -i ~/.ssh/id_jetson jetson@172.22.234.49 "cd ~/dqn_cuda_project && ./bin/dqn_server_cuda"
```

### Ventana de Pygame no aparece

**Verificar DISPLAY**:

```bash
echo $DISPLAY
# Debería mostrar algo como :0 o :1
```

---

## 📊 INTERPRETANDO RESULTADOS

### Métricas importantes:

**Reward positivo**: El robot se acerca al objetivo
**Reward negativo**: Colisiones o alejamiento
**Epsilon**:

- Alto (cerca de 1.0): Exploración (aleatorio)
- Bajo (cerca de 0.01): Explotación (usa lo aprendido)

### Convergencia:

- Episodios 1-30: Movimientos aleatorios
- Episodios 30-70: Comienza a aprender
- Episodios 70+: Comportamiento inteligente

**Buen signo**: Reward aumenta con el tiempo
**Mal signo**: Reward siempre negativo (revisar recompensas)

---

## 💡 TIPS

1. **Primera vez**: Usa 50-100 episodios para ver progreso
2. **Entrenamiento largo**: Usa `--no-render` para más velocidad
3. **Debugging**: Usa `--episodes 5` para pruebas rápidas
4. **Monitorear**: Deja la terminal del servidor visible
5. **Performance**: Cierra otras aplicaciones GPU-intensivas

---

## 📞 COMANDOS DE REFERENCIA RÁPIDA

```bash
# Deployar
./quick_deploy.sh

# Probar conexión
./test_connection.sh

# Servidor (en Jetson)
ssh -i ~/.ssh/id_jetson jetson@172.22.234.49 "cd ~/dqn_cuda_project && ./bin/dqn_server_cuda"

# Cliente (en Laptop)
./run_training.sh

# Recompilar
ssh -i ~/.ssh/id_jetson jetson@172.22.234.49 "cd ~/dqn_cuda_project && make clean && make"
```

---

## 🎯 WORKFLOW PARA PRESENTACIÓN

1. **Preparación**:

   ```bash
   ./test_connection.sh  # Verificar todo funciona
   ```

2. **Demo Entrenamiento** (Terminal 1 - Jetson):

   ```bash
   ssh -i ~/.ssh/id_jetson jetson@172.22.234.49
   cd ~/dqn_cuda_project
   ./bin/dqn_server_cuda
   ```

3. **Demo Visual** (Terminal 2 - Laptop):

   ```bash
   python3 client_cuda.py --host 172.22.234.49 --episodes 50
   ```

4. **Mostrar progreso**: Los primeros episodios son aleatorios, luego mejora

---

¿Problemas? Ejecuta: `./test_connection.sh` y revisa la salida.
