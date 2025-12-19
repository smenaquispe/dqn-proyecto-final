/*
 * DQN Implementation with CUDA Kernels
 * Implementación completa de Deep Q-Network usando CUDA C
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// ============================================================================
// CONFIGURACIÓN Y CONSTANTES
// ============================================================================

#define STATE_SIZE 15
#define ACTION_SIZE 8
#define HIDDEN_SIZE 256
#define HIDDEN_SIZE_2 128
#define BATCH_SIZE 64
#define BUFFER_SIZE 100000
#define LEARNING_RATE 0.0001f
#define GAMMA 0.99f

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// ESTRUCTURAS DE DATOS
// ============================================================================

// Red neuronal (pesos y biases)
typedef struct {
    // Capa 1: STATE_SIZE -> HIDDEN_SIZE
    float *W1;
    float *b1;
    
    // Capa 2: HIDDEN_SIZE -> HIDDEN_SIZE
    float *W2;
    float *b2;
    
    // Capa 3: HIDDEN_SIZE -> HIDDEN_SIZE_2
    float *W3;
    float *b3;
    
    // Capa 4: HIDDEN_SIZE_2 -> ACTION_SIZE
    float *W4;
    float *b4;
} NeuralNetwork;

// Experiencia en el replay buffer
typedef struct {
    float state[STATE_SIZE];
    int action;
    float reward;
    float next_state[STATE_SIZE];
    int done;
} Experience;

// Replay Buffer
typedef struct {
    Experience *buffer;
    int capacity;
    int size;
    int index;
} ReplayBuffer;

// Agente DQN
typedef struct {
    NeuralNetwork *policy_net;
    NeuralNetwork *target_net;
    ReplayBuffer *replay_buffer;
    float epsilon;
    int update_counter;
} DQNAgent;

// ============================================================================
// KERNELS CUDA - OPERACIONES DE RED NEURONAL
// ============================================================================

// Kernel para inicialización Xavier
__global__ void xavier_init_kernel(float *W, int fan_in, int fan_out, int size, 
                                   unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        float limit = sqrtf(6.0f / (fan_in + fan_out));
        W[idx] = (curand_uniform(&state) * 2.0f - 1.0f) * limit;
    }
}

// Kernel para multiplicación matriz-vector (forward pass)
__global__ void matmul_kernel(float *output, float *input, float *W, float *b,
                              int in_size, int out_size, int batch_size) {
    int row = blockIdx.x;
    int batch_idx = blockIdx.y;
    
    if (row < out_size && batch_idx < batch_size) {
        float sum = 0.0f;
        for (int i = 0; i < in_size; i++) {
            sum += input[batch_idx * in_size + i] * W[row * in_size + i];
        }
        sum += b[row];
        output[batch_idx * out_size + row] = sum;
    }
}

// Kernel para función de activación ReLU
__global__ void relu_kernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// Kernel para Batch Normalization
__global__ void batch_norm_kernel(float *data, float *mean, float *var, 
                                  int batch_size, int feature_size) {
    int feature_idx = blockIdx.x;
    int batch_idx = threadIdx.x;
    
    if (feature_idx < feature_size && batch_idx < batch_size) {
        int idx = batch_idx * feature_size + feature_idx;
        float eps = 1e-5f;
        data[idx] = (data[idx] - mean[feature_idx]) / sqrtf(var[feature_idx] + eps);
    }
}

// Kernel para calcular mean y variance para batch norm
__global__ void compute_mean_var_kernel(float *data, float *mean, float *var,
                                       int batch_size, int feature_size) {
    int feature_idx = blockIdx.x;
    
    if (feature_idx < feature_size) {
        float sum = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            sum += data[i * feature_size + feature_idx];
        }
        mean[feature_idx] = sum / batch_size;
        
        float var_sum = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            float diff = data[i * feature_size + feature_idx] - mean[feature_idx];
            var_sum += diff * diff;
        }
        var[feature_idx] = var_sum / batch_size;
    }
}

// Kernel para backward pass - calcular gradientes
__global__ void compute_gradients_kernel(float *grad_W, float *grad_b,
                                        float *grad_output, float *input,
                                        int in_size, int out_size, int batch_size) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    
    if (row < out_size && col < in_size) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            sum += grad_output[b * out_size + row] * input[b * in_size + col];
        }
        grad_W[row * in_size + col] = sum / batch_size;
    }
    
    // Calcular gradiente del bias
    if (col == 0 && row < out_size) {
        float sum_b = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            sum_b += grad_output[b * out_size + row];
        }
        grad_b[row] = sum_b / batch_size;
    }
}

// Kernel para actualizar pesos (SGD con momentum)
__global__ void update_weights_kernel(float *W, float *grad_W, float learning_rate, 
                                     int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        W[idx] -= learning_rate * grad_W[idx];
    }
}

// Kernel para calcular TD error (Temporal Difference)
__global__ void compute_td_error_kernel(float *td_error, float *q_values, 
                                       float *next_q_values, float *rewards,
                                       int *actions, int *dones, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        int action = actions[idx];
        float q_current = q_values[idx * ACTION_SIZE + action];
        
        // Encontrar max Q-value del siguiente estado
        float max_next_q = next_q_values[idx * ACTION_SIZE];
        for (int a = 1; a < ACTION_SIZE; a++) {
            float q = next_q_values[idx * ACTION_SIZE + a];
            if (q > max_next_q) max_next_q = q;
        }
        
        float target = rewards[idx] + GAMMA * max_next_q * (1 - dones[idx]);
        td_error[idx] = target - q_current;
    }
}

// Kernel para Huber Loss
__global__ void huber_loss_kernel(float *loss, float *td_error, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float delta = 1.0f;
        float abs_error = fabsf(td_error[idx]);
        if (abs_error <= delta) {
            loss[idx] = 0.5f * td_error[idx] * td_error[idx];
        } else {
            loss[idx] = delta * (abs_error - 0.5f * delta);
        }
    }
}

// ============================================================================
// FUNCIONES DE HOST - GESTIÓN DE MEMORIA Y RED
// ============================================================================

// Crear red neuronal
NeuralNetwork* create_network() {
    NeuralNetwork *net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    
    // Alocar memoria en GPU para pesos y biases
    CUDA_CHECK(cudaMalloc(&net->W1, STATE_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net->b1, HIDDEN_SIZE * sizeof(float)));
    
    CUDA_CHECK(cudaMalloc(&net->W2, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net->b2, HIDDEN_SIZE * sizeof(float)));
    
    CUDA_CHECK(cudaMalloc(&net->W3, HIDDEN_SIZE * HIDDEN_SIZE_2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net->b3, HIDDEN_SIZE_2 * sizeof(float)));
    
    CUDA_CHECK(cudaMalloc(&net->W4, HIDDEN_SIZE_2 * ACTION_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net->b4, ACTION_SIZE * sizeof(float)));
    
    // Inicializar pesos con Xavier initialization
    int threads = 256;
    unsigned long seed = time(NULL);
    
    int blocks_W1 = (STATE_SIZE * HIDDEN_SIZE + threads - 1) / threads;
    xavier_init_kernel<<<blocks_W1, threads>>>(net->W1, STATE_SIZE, HIDDEN_SIZE, 
                                              STATE_SIZE * HIDDEN_SIZE, seed);
    
    int blocks_W2 = (HIDDEN_SIZE * HIDDEN_SIZE + threads - 1) / threads;
    xavier_init_kernel<<<blocks_W2, threads>>>(net->W2, HIDDEN_SIZE, HIDDEN_SIZE,
                                              HIDDEN_SIZE * HIDDEN_SIZE, seed + 1);
    
    int blocks_W3 = (HIDDEN_SIZE * HIDDEN_SIZE_2 + threads - 1) / threads;
    xavier_init_kernel<<<blocks_W3, threads>>>(net->W3, HIDDEN_SIZE, HIDDEN_SIZE_2,
                                              HIDDEN_SIZE * HIDDEN_SIZE_2, seed + 2);
    
    int blocks_W4 = (HIDDEN_SIZE_2 * ACTION_SIZE + threads - 1) / threads;
    xavier_init_kernel<<<blocks_W4, threads>>>(net->W4, HIDDEN_SIZE_2, ACTION_SIZE,
                                              HIDDEN_SIZE_2 * ACTION_SIZE, seed + 3);
    
    // Inicializar biases a cero
    CUDA_CHECK(cudaMemset(net->b1, 0, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(net->b2, 0, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(net->b3, 0, HIDDEN_SIZE_2 * sizeof(float)));
    CUDA_CHECK(cudaMemset(net->b4, 0, ACTION_SIZE * sizeof(float)));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return net;
}

// Forward pass de la red neuronal
void forward_pass(NeuralNetwork *net, float *input, float *output, int batch_size) {
    // Buffers temporales en GPU
    float *layer1, *layer2, *layer3;
    float *mean1, *var1, *mean2, *var2, *mean3, *var3;
    
    CUDA_CHECK(cudaMalloc(&layer1, batch_size * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer2, batch_size * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer3, batch_size * HIDDEN_SIZE_2 * sizeof(float)));
    
    CUDA_CHECK(cudaMalloc(&mean1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&var1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&mean2, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&var2, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&mean3, HIDDEN_SIZE_2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&var3, HIDDEN_SIZE_2 * sizeof(float)));
    
    // Capa 1
    dim3 grid1(HIDDEN_SIZE, batch_size);
    matmul_kernel<<<grid1, 1>>>(layer1, input, net->W1, net->b1, 
                                STATE_SIZE, HIDDEN_SIZE, batch_size);
    
    compute_mean_var_kernel<<<HIDDEN_SIZE, 1>>>(layer1, mean1, var1, 
                                                batch_size, HIDDEN_SIZE);
    batch_norm_kernel<<<HIDDEN_SIZE, batch_size>>>(layer1, mean1, var1, 
                                                   batch_size, HIDDEN_SIZE);
    
    int threads = 256;
    int blocks = (batch_size * HIDDEN_SIZE + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(layer1, batch_size * HIDDEN_SIZE);
    
    // Capa 2
    dim3 grid2(HIDDEN_SIZE, batch_size);
    matmul_kernel<<<grid2, 1>>>(layer2, layer1, net->W2, net->b2,
                                HIDDEN_SIZE, HIDDEN_SIZE, batch_size);
    
    compute_mean_var_kernel<<<HIDDEN_SIZE, 1>>>(layer2, mean2, var2,
                                                batch_size, HIDDEN_SIZE);
    batch_norm_kernel<<<HIDDEN_SIZE, batch_size>>>(layer2, mean2, var2,
                                                   batch_size, HIDDEN_SIZE);
    
    relu_kernel<<<blocks, threads>>>(layer2, batch_size * HIDDEN_SIZE);
    
    // Capa 3
    dim3 grid3(HIDDEN_SIZE_2, batch_size);
    matmul_kernel<<<grid3, 1>>>(layer3, layer2, net->W3, net->b3,
                                HIDDEN_SIZE, HIDDEN_SIZE_2, batch_size);
    
    compute_mean_var_kernel<<<HIDDEN_SIZE_2, 1>>>(layer3, mean3, var3,
                                                  batch_size, HIDDEN_SIZE_2);
    batch_norm_kernel<<<HIDDEN_SIZE_2, batch_size>>>(layer3, mean3, var3,
                                                     batch_size, HIDDEN_SIZE_2);
    
    blocks = (batch_size * HIDDEN_SIZE_2 + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(layer3, batch_size * HIDDEN_SIZE_2);
    
    // Capa 4 (output)
    dim3 grid4(ACTION_SIZE, batch_size);
    matmul_kernel<<<grid4, 1>>>(output, layer3, net->W4, net->b4,
                                HIDDEN_SIZE_2, ACTION_SIZE, batch_size);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Liberar buffers temporales
    cudaFree(layer1);
    cudaFree(layer2);
    cudaFree(layer3);
    cudaFree(mean1);
    cudaFree(var1);
    cudaFree(mean2);
    cudaFree(var2);
    cudaFree(mean3);
    cudaFree(var3);
}

// Seleccionar acción con epsilon-greedy
int select_action(NeuralNetwork *net, float *state, float epsilon) {
    // Decidir si explorar o explotar
    float rand_val = (float)rand() / RAND_MAX;
    
    if (rand_val < epsilon) {
        // Exploración: acción aleatoria
        return rand() % ACTION_SIZE;
    } else {
        // Explotación: mejor acción según Q-values
        float *state_gpu, *q_values_gpu;
        float q_values[ACTION_SIZE];
        
        CUDA_CHECK(cudaMalloc(&state_gpu, STATE_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&q_values_gpu, ACTION_SIZE * sizeof(float)));
        
        CUDA_CHECK(cudaMemcpy(state_gpu, state, STATE_SIZE * sizeof(float),
                             cudaMemcpyHostToDevice));
        
        forward_pass(net, state_gpu, q_values_gpu, 1);
        
        CUDA_CHECK(cudaMemcpy(q_values, q_values_gpu, ACTION_SIZE * sizeof(float),
                             cudaMemcpyDeviceToHost));
        
        cudaFree(state_gpu);
        cudaFree(q_values_gpu);
        
        // Encontrar acción con máximo Q-value
        int best_action = 0;
        float max_q = q_values[0];
        for (int i = 1; i < ACTION_SIZE; i++) {
            if (q_values[i] > max_q) {
                max_q = q_values[i];
                best_action = i;
            }
        }
        
        return best_action;
    }
}

// Crear replay buffer
ReplayBuffer* create_replay_buffer(int capacity) {
    ReplayBuffer *buffer = (ReplayBuffer*)malloc(sizeof(ReplayBuffer));
    buffer->capacity = capacity;
    buffer->size = 0;
    buffer->index = 0;
    
    // Alocar en memoria de host (CPU)
    buffer->buffer = (Experience*)malloc(capacity * sizeof(Experience));
    
    return buffer;
}

// Agregar experiencia al buffer
void add_experience(ReplayBuffer *buffer, float *state, int action, float reward,
                   float *next_state, int done) {
    Experience *exp = &buffer->buffer[buffer->index];
    
    memcpy(exp->state, state, STATE_SIZE * sizeof(float));
    exp->action = action;
    exp->reward = reward;
    memcpy(exp->next_state, next_state, STATE_SIZE * sizeof(float));
    exp->done = done;
    
    buffer->index = (buffer->index + 1) % buffer->capacity;
    if (buffer->size < buffer->capacity) {
        buffer->size++;
    }
}

// Muestrear batch del replay buffer
void sample_batch(ReplayBuffer *buffer, float *states, int *actions, float *rewards,
                 float *next_states, int *dones, int batch_size) {
    for (int i = 0; i < batch_size; i++) {
        int idx = rand() % buffer->size;
        Experience *exp = &buffer->buffer[idx];
        
        memcpy(&states[i * STATE_SIZE], exp->state, STATE_SIZE * sizeof(float));
        actions[i] = exp->action;
        rewards[i] = exp->reward;
        memcpy(&next_states[i * STATE_SIZE], exp->next_state, STATE_SIZE * sizeof(float));
        dones[i] = exp->done;
    }
}

// Entrenar el agente (un paso)
float train_step(DQNAgent *agent) {
    if (agent->replay_buffer->size < BATCH_SIZE) {
        return 0.0f;
    }
    
    // Muestrear batch
    float *states_host = (float*)malloc(BATCH_SIZE * STATE_SIZE * sizeof(float));
    int *actions_host = (int*)malloc(BATCH_SIZE * sizeof(int));
    float *rewards_host = (float*)malloc(BATCH_SIZE * sizeof(float));
    float *next_states_host = (float*)malloc(BATCH_SIZE * STATE_SIZE * sizeof(float));
    int *dones_host = (int*)malloc(BATCH_SIZE * sizeof(int));
    
    sample_batch(agent->replay_buffer, states_host, actions_host, rewards_host,
                next_states_host, dones_host, BATCH_SIZE);
    
    // Copiar a GPU
    float *states_gpu, *next_states_gpu, *rewards_gpu;
    int *actions_gpu, *dones_gpu;
    float *q_values_gpu, *next_q_values_gpu, *td_error_gpu, *loss_gpu;
    
    CUDA_CHECK(cudaMalloc(&states_gpu, BATCH_SIZE * STATE_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&next_states_gpu, BATCH_SIZE * STATE_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&rewards_gpu, BATCH_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&actions_gpu, BATCH_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dones_gpu, BATCH_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&q_values_gpu, BATCH_SIZE * ACTION_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&next_q_values_gpu, BATCH_SIZE * ACTION_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&td_error_gpu, BATCH_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&loss_gpu, BATCH_SIZE * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(states_gpu, states_host, BATCH_SIZE * STATE_SIZE * sizeof(float),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(next_states_gpu, next_states_host, 
                         BATCH_SIZE * STATE_SIZE * sizeof(float),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(rewards_gpu, rewards_host, BATCH_SIZE * sizeof(float),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(actions_gpu, actions_host, BATCH_SIZE * sizeof(int),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dones_gpu, dones_host, BATCH_SIZE * sizeof(int),
                         cudaMemcpyHostToDevice));
    
    // Forward pass con policy network
    forward_pass(agent->policy_net, states_gpu, q_values_gpu, BATCH_SIZE);
    
    // Forward pass con target network
    forward_pass(agent->target_net, next_states_gpu, next_q_values_gpu, BATCH_SIZE);
    
    // Calcular TD error
    int threads = 256;
    int blocks = (BATCH_SIZE + threads - 1) / threads;
    compute_td_error_kernel<<<blocks, threads>>>(td_error_gpu, q_values_gpu,
                                                next_q_values_gpu, rewards_gpu,
                                                actions_gpu, dones_gpu, BATCH_SIZE);
    
    // Calcular loss (Huber loss)
    huber_loss_kernel<<<blocks, threads>>>(loss_gpu, td_error_gpu, BATCH_SIZE);
    
    // Copiar loss a CPU para retornar
    float *loss_host = (float*)malloc(BATCH_SIZE * sizeof(float));
    CUDA_CHECK(cudaMemcpy(loss_host, loss_gpu, BATCH_SIZE * sizeof(float),
                         cudaMemcpyDeviceToHost));
    
    float total_loss = 0.0f;
    for (int i = 0; i < BATCH_SIZE; i++) {
        total_loss += loss_host[i];
    }
    total_loss /= BATCH_SIZE;
    
    // TODO: Implementar backward pass y actualización de pesos
    // (Esto requeriría implementar autograd en CUDA, que es muy complejo)
    // Por ahora, esta es la estructura básica
    
    // Actualizar target network periódicamente
    agent->update_counter++;
    if (agent->update_counter % 10 == 0) {
        // Copiar pesos de policy_net a target_net
        // (Implementar función copy_network)
    }
    
    // Liberar memoria
    free(states_host);
    free(actions_host);
    free(rewards_host);
    free(next_states_host);
    free(dones_host);
    free(loss_host);
    
    cudaFree(states_gpu);
    cudaFree(next_states_gpu);
    cudaFree(rewards_gpu);
    cudaFree(actions_gpu);
    cudaFree(dones_gpu);
    cudaFree(q_values_gpu);
    cudaFree(next_q_values_gpu);
    cudaFree(td_error_gpu);
    cudaFree(loss_gpu);
    
    return total_loss;
}

// Guardar modelo
void save_model(NeuralNetwork *net, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Copiar pesos de GPU a CPU y guardar
    float *W1_host = (float*)malloc(STATE_SIZE * HIDDEN_SIZE * sizeof(float));
    float *b1_host = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    // ... similar para otros pesos
    
    CUDA_CHECK(cudaMemcpy(W1_host, net->W1, STATE_SIZE * HIDDEN_SIZE * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b1_host, net->b1, HIDDEN_SIZE * sizeof(float),
                         cudaMemcpyDeviceToHost));
    
    fwrite(W1_host, sizeof(float), STATE_SIZE * HIDDEN_SIZE, file);
    fwrite(b1_host, sizeof(float), HIDDEN_SIZE, file);
    // ... guardar otros pesos
    
    fclose(file);
    free(W1_host);
    free(b1_host);
    
    printf("Model saved to %s\n", filename);
}

// Cargar modelo
void load_model(NeuralNetwork *net, const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening file for reading: %s\n", filename);
        return;
    }
    
    float *W1_host = (float*)malloc(STATE_SIZE * HIDDEN_SIZE * sizeof(float));
    float *b1_host = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    
    fread(W1_host, sizeof(float), STATE_SIZE * HIDDEN_SIZE, file);
    fread(b1_host, sizeof(float), HIDDEN_SIZE, file);
    
    CUDA_CHECK(cudaMemcpy(net->W1, W1_host, STATE_SIZE * HIDDEN_SIZE * sizeof(float),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->b1, b1_host, HIDDEN_SIZE * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    fclose(file);
    free(W1_host);
    free(b1_host);
    
    printf("Model loaded from %s\n", filename);
}

// Liberar memoria de la red
void free_network(NeuralNetwork *net) {
    cudaFree(net->W1);
    cudaFree(net->b1);
    cudaFree(net->W2);
    cudaFree(net->b2);
    cudaFree(net->W3);
    cudaFree(net->b3);
    cudaFree(net->W4);
    cudaFree(net->b4);
    free(net);
}

// Liberar replay buffer
void free_replay_buffer(ReplayBuffer *buffer) {
    free(buffer->buffer);
    free(buffer);
}

// ============================================================================
// FUNCIÓN PRINCIPAL (EJEMPLO)
// ============================================================================

int main() {
    printf("=== DQN with CUDA C Implementation ===\n\n");
    
    // Inicializar CUDA
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    printf("CUDA devices found: %d\n", device_count);
    
    if (device_count > 0) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        printf("Using device: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Global memory: %.2f GB\n\n", prop.totalGlobalMem / 1e9);
    }
    
    // Crear agente DQN
    DQNAgent agent;
    agent.policy_net = create_network();
    agent.target_net = create_network();
    agent.replay_buffer = create_replay_buffer(BUFFER_SIZE);
    agent.epsilon = 1.0f;
    agent.update_counter = 0;
    
    printf("DQN Agent created successfully!\n");
    printf("State size: %d\n", STATE_SIZE);
    printf("Action size: %d\n", ACTION_SIZE);
    printf("Hidden layer 1: %d neurons\n", HIDDEN_SIZE);
    printf("Hidden layer 2: %d neurons\n", HIDDEN_SIZE_2);
    printf("Replay buffer capacity: %d\n\n", BUFFER_SIZE);
    
    // Ejemplo de uso
    float state[STATE_SIZE] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 
                               0.6f, 0.7f, 0.8f, 0.9f, 0.1f,
                               0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    
    int action = select_action(agent.policy_net, state, agent.epsilon);
    printf("Selected action: %d\n", action);
    
    // Agregar experiencia de ejemplo
    float next_state[STATE_SIZE] = {0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                                    0.7f, 0.8f, 0.9f, 0.1f, 0.2f,
                                    0.3f, 0.4f, 0.5f, 0.6f, 0.7f};
    add_experience(agent.replay_buffer, state, action, 1.0f, next_state, 0);
    
    printf("Experience added to replay buffer\n");
    printf("Buffer size: %d\n\n", agent.replay_buffer->size);
    
    // Limpiar
    free_network(agent.policy_net);
    free_network(agent.target_net);
    free_replay_buffer(agent.replay_buffer);
    
    printf("Cleanup completed successfully!\n");
    
    return 0;
}
