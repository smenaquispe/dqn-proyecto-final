╔════════════════════════════════════════════════════════════════╗
║ 🎯 MODO DE INFERENCIA - GUÍA DE USO ║
╚════════════════════════════════════════════════════════════════╝

📅 Fecha: 18 de Diciembre 2025
🎯 Estado: LISTO PARA USAR

═══════════════════════════════════════════════════════════════════
🎮 ¿QUÉ ES EL MODO DE INFERENCIA?
═══════════════════════════════════════════════════════════════════

En este modo, el robot usa el modelo YA ENTRENADO para navegar.
El modelo está en el Jetson y solo hace predicciones (no aprende).

Flujo:

1. Laptop muestra el entorno en Pygame
2. Laptop envía el estado al Jetson
3. Jetson usa el modelo entrenado para decidir la acción
4. Jetson devuelve la acción y los Q-values
5. Laptop ejecuta la acción
6. Repetir hasta llegar al objetivo

═══════════════════════════════════════════════════════════════════
🚀 INICIO RÁPIDO (2 pasos)
═══════════════════════════════════════════════════════════════════

PASO 1: Iniciar servidor de inferencia en el Jetson
────────────────────────────────────────────────────

Terminal 1 (En tu laptop):

    ssh -i ~/.ssh/id_jetson jetson@172.22.234.49
    cd ~/dqn_cuda_project
    ./start_inference_server.sh

O manualmente:

    ./bin/dqn_inference_server dqn_model_final.bin

Deberías ver:
==============================================
DQN Inference Server with CUDA C
Jetson Xavier - Inference Only Mode
==============================================

    CUDA Device: Xavier

    === Loading model ===
    File: dqn_model_final.bin
    Model architecture: 15 -> 128 -> 128 -> 64 -> 8
    ✓ Model loaded successfully

    Inference server listening on port 5557
    Model loaded: dqn_model_final.bin
    Waiting for client connection...

⚠️ DEJA ESTA TERMINAL CORRIENDO

PASO 2: Ejecutar cliente de inferencia desde la laptop
────────────────────────────────────────────────────────

Terminal 2 (Nueva terminal en laptop):

    cd /home/smenaq/Documents/UNSA/robotica/proyecto-final
    ./run_inference.sh

O manualmente:

    python3 client_inference.py --host 172.22.234.49 --episodes 10

═══════════════════════════════════════════════════════════════════
🎯 ¿QUÉ VAS A VER?
═══════════════════════════════════════════════════════════════════

En la ventana de Pygame:
• Robot (azul) moviéndose inteligentemente
• Objetivo (verde)
• Obstáculos (negro)
• Información en pantalla

En la terminal del cliente:
=== Episode 1/10 ===
Step 10: Action=↗, Max Q=0.523
Step 20: Action=→, Max Q=0.687

    ==================================================
    Episode 1 Summary:
      Result: ✓ SUCCESS
      Reward: 45.20
      Steps: 87
      Final distance to goal: 15.2
      Success rate: 1/1 (100.0%)
    ==================================================

Al final:
============================================================
INFERENCE STATISTICS
============================================================
Total episodes: 10
Successful episodes: 8 (80.0%)

    Rewards:
      Average: 38.45
      Best: 52.30
      Worst: -12.40
      Std Dev: 18.23

    Steps:
      Average: 125.3
      Best: 65
      Worst: 250
    ============================================================

═══════════════════════════════════════════════════════════════════
⚙️ OPCIONES DISPONIBLES
═══════════════════════════════════════════════════════════════════

# Cambiar número de episodios

python3 client_inference.py --host 172.22.234.49 --episodes 20

# Cambiar velocidad de visualización

python3 client_inference.py --host 172.22.234.49 --delay 0.1

# Más rápido (menos delay)

python3 client_inference.py --host 172.22.234.49 --delay 0.01

# Sin visualización (máxima velocidad)

python3 client_inference.py --host 172.22.234.49 --episodes 50 --no-render

# Usar modelo específico (en el servidor)

./bin/dqn_inference_server dqn_model_ep5.bin

# Test de conexión

python3 client_inference.py --host 172.22.234.49 --test

═══════════════════════════════════════════════════════════════════
📊 DIFERENCIAS: ENTRENAMIENTO vs INFERENCIA
═══════════════════════════════════════════════════════════════════

┌─────────────────┬───────────────────┬───────────────────┐
│ Aspecto │ Entrenamiento │ Inferencia │
├─────────────────┼───────────────────┼───────────────────┤
│ Puerto │ 5556 │ 5557 │
│ Servidor │ dqn_server_cuda │ dqn_inference_srv │
│ Cliente │ client_cuda.py │ client_inference │
│ Script inicio │ run_training.sh │ run_inference.sh │
│ Exploración │ Sí (epsilon) │ No (greedy) │
│ Aprende │ Sí │ No │
│ Guarda modelo │ Sí │ No │
│ Usa modelo │ Crea nuevo │ Carga existente │
│ Q-values │ No se muestran │ Se muestran │
│ Velocidad │ Variable │ Configurable │
└─────────────────┴───────────────────┴───────────────────┘

═══════════════════════════════════════════════════════════════════
🎬 WORKFLOW COMPLETO
═══════════════════════════════════════════════════════════════════

1. ENTRENAR EL MODELO (una vez)
   ────────────────────────────
   Terminal 1: ssh y ejecutar dqn_server_cuda
   Terminal 2: python3 client_cuda.py --host 172.22.234.49 --episodes 100
   Resultado: dqn_model_final.bin guardado en Jetson

2. USAR EL MODELO (múltiples veces)
   ────────────────────────────────
   Terminal 1: ssh y ejecutar dqn_inference_server dqn_model_final.bin
   Terminal 2: python3 client_inference.py --host 172.22.234.49 --episodes 10
   Resultado: Ver al robot navegar inteligentemente

═══════════════════════════════════════════════════════════════════
📈 INTERPRETAR RESULTADOS
═══════════════════════════════════════════════════════════════════

✓ SUCCESS: El robot llegó al objetivo
• Distance < 30 pixels (1.5 grid cells)
• Reward alto y positivo
• Steps razonables

✗ FAILED: El robot no llegó
• Alcanzó max_steps (500)
• Reward negativo
• Puede mejorar con más entrenamiento

Q-values:
• Valores altos (> 0): Buenas acciones
• Valores bajos (< 0): Malas acciones
• Max Q: Confianza de la mejor acción

Success Rate:
• > 70%: Excelente modelo
• 50-70%: Buen modelo
• < 50%: Necesita más entrenamiento

═══════════════════════════════════════════════════════════════════
🔧 SOLUCIÓN DE PROBLEMAS
═══════════════════════════════════════════════════════════════════

Error: "Connection refused to port 5557"
→ El servidor de inferencia no está corriendo
→ Solución: ssh al Jetson y ejecutar ./start_inference_server.sh

Error: "Model file not found"
→ El modelo no existe en el Jetson
→ Solución: Entrenar primero o verificar:
ssh -i ~/.ssh/id_jetson jetson@172.22.234.49 "ls ~/dqn_cuda_project/\*.bin"

Robot se mueve aleatoriamente
→ El modelo no está bien entrenado
→ Solución: Entrenar más episodios (500-1000)

Success rate muy bajo
→ Modelo necesita más entrenamiento
→ Solución: Entrenar con más episodios o ajustar hiperparámetros

Puerto 5557 ya en uso
→ Otra instancia corriendo
→ Solución:
ssh -i ~/.ssh/id_jetson jetson@172.22.234.49 "pkill dqn_inference"

═══════════════════════════════════════════════════════════════════
💡 TIPS Y MEJORES PRÁCTICAS
═══════════════════════════════════════════════════════════════════

1. VELOCIDAD DE VISUALIZACIÓN:
   • delay 0.05: Bueno para ver y analizar
   • delay 0.01: Más rápido pero visible
   • --no-render: Máxima velocidad (sin visualización)

2. EVALUACIÓN:
   • Usa 10-20 episodios para evaluar
   • Varía posiciones iniciales automáticamente
   • Mide success rate y average reward

3. COMPARACIÓN:
   • Evalúa dqn_model_ep5.bin vs dqn_model_final.bin
   • Compara success rate
   • El modelo final debería ser mejor

4. DEMO:
   • Usa delay 0.05-0.1 para presentaciones
   • 5-10 episodios es suficiente
   • Muestra primero sin modelo vs con modelo

═══════════════════════════════════════════════════════════════════
📞 COMANDOS DE REFERENCIA RÁPIDA
═══════════════════════════════════════════════════════════════════

# Servidor inferencia (en Jetson)

ssh -i ~/.ssh/id_jetson jetson@172.22.234.49 "cd ~/dqn_cuda_project && ./start_inference_server.sh"

# Cliente inferencia (en Laptop)

./run_inference.sh

# Con opciones

python3 client_inference.py --host 172.22.234.49 --episodes 20 --delay 0.05

# Ver modelos disponibles

ssh -i ~/.ssh/id_jetson jetson@172.22.234.49 "ls -lh ~/dqn_cuda_project/\*.bin"

# Descargar modelo a laptop

scp -i ~/.ssh/id_jetson jetson@172.22.234.49:~/dqn_cuda_project/dqn_model_final.bin .

# Matar servidor

ssh -i ~/.ssh/id_jetson jetson@172.22.234.49 "pkill dqn_inference"

═══════════════════════════════════════════════════════════════════
🎯 EJEMPLO DE SESIÓN COMPLETA
═══════════════════════════════════════════════════════════════════

Terminal 1 - Jetson:
$ ssh -i ~/.ssh/id_jetson jetson@172.22.234.49
$ cd ~/dqn_cuda_project
$ ./start_inference_server.sh

[Servidor inicia y carga modelo...]

Terminal 2 - Laptop:
$ cd /home/smenaq/Documents/UNSA/robotica/proyecto-final
$ python3 client_inference.py --host 172.22.234.49 --episodes 10

[Se abre ventana Pygame]
[Robot navega inteligentemente]
[Muestra estadísticas al final]

Success rate: 8/10 (80%)
Average reward: 42.5

═══════════════════════════════════════════════════════════════════
🎉 ¡LISTO!
═══════════════════════════════════════════════════════════════════

Ahora puedes:
✅ Ver al robot usar el modelo entrenado
✅ Evaluar el rendimiento del modelo
✅ Comparar diferentes modelos
✅ Hacer demos del proyecto

Para más información:
• README.md - Documentación completa
• GUIA_USO.md - Guía de entrenamiento
• DEPLOY_STATUS.md - Estado del deployment

═══════════════════════════════════════════════════════════════════
