# Comandos Rápidos

## 🎯 ENTRENAMIENTO

### Terminal 1 - Jetson

```bash
ssh -i ~/.ssh/id_jetson jetson@172.22.234.49 "cd ~/dqn_cuda_project && ./bin/dqn_server_cuda"
```

### Terminal 2 - Laptop

```bash
cd client && python3 client_cuda.py
```

---

## 🤖 INFERENCIA

### Terminal 1 - Jetson

```bash
ssh -i ~/.ssh/id_jetson jetson@172.22.234.49 "cd ~/dqn_cuda_project && ./bin/dqn_inference_server models/dqn_model_final.bin"
```

### Terminal 2 - Laptop

```bash
cd client && python3 client_inference.py
```

---

## ⚙️ CON PARÁMETROS PERSONALIZADOS

### Entrenamiento (escenario más pequeño, menos obstáculos)

```bash
python3 client_cuda.py --episodes 300 --width 300 --height 300 --obstacles 3 --max-steps 150
```

### Inferencia (más lenta para ver mejor)

```bash
python3 client_inference.py --episodes 15 --delay 0.15
```

---

## 🔧 SETUP INICIAL (Solo primera vez)

```bash
# 1. Copiar archivos al Jetson
scp -i ~/.ssh/id_jetson -r jetson_files/* jetson@172.22.234.49:~/dqn_cuda_project/

# 2. Crear carpeta models
ssh -i ~/.ssh/id_jetson jetson@172.22.234.49 "mkdir -p ~/dqn_cuda_project/models"

# 3. Compilar en Jetson
ssh -i ~/.ssh/id_jetson jetson@172.22.234.49 "cd ~/dqn_cuda_project && export PATH=/usr/local/cuda/bin:\$PATH && make"
```

---

## 📊 VER MODELOS

```bash
ssh -i ~/.ssh/id_jetson jetson@172.22.234.49 "ls -lh ~/dqn_cuda_project/models/"
```
