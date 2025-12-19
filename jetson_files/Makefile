# Makefile para compilar DQN con CUDA C
# Para NVIDIA Jetson Xavier

# Compilador CUDA
NVCC = nvcc

# Flags de compilación
NVCC_FLAGS = -O3 -arch=sm_72 -lcurand
# sm_72 es para Jetson Xavier. Ajustar según tu GPU:
# sm_72: Jetson Xavier
# sm_87: Jetson Orin
# sm_75: RTX 20xx
# sm_86: RTX 30xx

# Directorios
SRC_DIR = .
BUILD_DIR = build
BIN_DIR = bin

# Archivos
TARGET = $(BIN_DIR)/dqn_cuda
SOURCES = dqn_cuda.cu
SERVER_TARGET = $(BIN_DIR)/dqn_server_cuda
SERVER_SOURCES = dqn_server_cuda.cu
INFERENCE_TARGET = $(BIN_DIR)/dqn_inference_server
INFERENCE_SOURCES = dqn_inference_server.cu

# Regla por defecto
all: directories $(TARGET) $(SERVER_TARGET) $(INFERENCE_TARGET)

# Crear directorios
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

# Compilar programa principal DQN
$(TARGET): $(SOURCES)
	@echo "Compilando DQN CUDA..."
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SOURCES)
	@echo "✓ Compilación exitosa: $(TARGET)"

# Compilar servidor
$(SERVER_TARGET): $(SERVER_SOURCES)
	@echo "Compilando servidor CUDA..."
	$(NVCC) $(NVCC_FLAGS) -o $(SERVER_TARGET) $(SERVER_SOURCES)
	@echo "✓ Compilación exitosa: $(SERVER_TARGET)"

# Compilar servidor de inferencia
$(INFERENCE_TARGET): $(INFERENCE_SOURCES)
	@echo "Compilando servidor de inferencia..."
	$(NVCC) $(NVCC_FLAGS) -o $(INFERENCE_TARGET) $(INFERENCE_SOURCES)
	@echo "✓ Compilación exitosa: $(INFERENCE_TARGET)"

# Limpiar archivos compilados
clean:
	@echo "Limpiando..."
	rm -rf $(BUILD_DIR) $(BIN_DIR)
	@echo "✓ Limpieza completada"

# Ejecutar programa principal
run: $(TARGET)
	@echo "Ejecutando DQN CUDA..."
	./$(TARGET)

# Ejecutar servidor
run-server: $(SERVER_TARGET)
	@echo "Ejecutando servidor CUDA..."
	./$(SERVER_TARGET)

# Ejecutar servidor de inferencia
run-inference: $(INFERENCE_TARGET)
	@echo "Ejecutando servidor de inferencia..."
	./$(INFERENCE_TARGET) dqn_model_final.bin

# Verificar instalación CUDA
check-cuda:
	@echo "=== Verificando CUDA ==="
	@which nvcc || echo "✗ nvcc no encontrado"
	@nvcc --version || echo "✗ Error al obtener versión de nvcc"
	@nvidia-smi || echo "✗ nvidia-smi no disponible (puede ser normal en Jetson)"
	@echo ""

# Información del dispositivo
device-info:
	@echo "=== Información del dispositivo CUDA ==="
	@$(NVCC) -o $(BUILD_DIR)/device_info device_query.cu 2>/dev/null || echo "Compile device_query.cu primero"
	@$(BUILD_DIR)/device_info 2>/dev/null || echo "No se pudo obtener información"

# Ayuda
help:
	@echo "Makefile para DQN con CUDA C"
	@echo ""
	@echo "Uso:"
	@echo "  make              - Compilar todos los programas"
	@echo "  make clean        - Limpiar archivos compilados"
	@echo "  make run          - Compilar y ejecutar programa principal"
	@echo "  make run-server   - Compilar y ejecutar servidor"
	@echo "  make check-cuda   - Verificar instalación de CUDA"
	@echo "  make device-info  - Mostrar información del dispositivo"
	@echo "  make help         - Mostrar esta ayuda"
	@echo ""
	@echo "Variables:"
	@echo "  NVCC_FLAGS        - Flags para nvcc (actual: $(NVCC_FLAGS))"
	@echo ""

.PHONY: all clean run run-server check-cuda device-info help directories
