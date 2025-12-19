/*
 * Servidor DQN con CUDA C
 * Recibe estados del cliente Python, procesa con CUDA, envía acciones
 */

#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define PORT 5556
#define STATE_SIZE 15
#define ACTION_SIZE 8
#define BUFFER_SIZE 4096

// Incluir las definiciones de la red neuronal del archivo principal
// (En un proyecto real, esto estaría en headers compartidos)

typedef struct {
  float *W1, *b1;
  float *W2, *b2;
  float *W3, *b3;
  float *W4, *b4;
  // Added for training
  float *d_W1, *d_b1;
  float *d_W2, *d_b2;
  float *d_W3, *d_b3;
  float *d_W4, *d_b4;
} NeuralNetwork;

// Replay Buffer Structures
typedef struct {
  float state[STATE_SIZE];
  int action;
  float reward;
  float next_state[STATE_SIZE];
  int done;
} Experience;

typedef struct {
  Experience *buffer;
  int capacity;
  int size;
  int index;
} ReplayBuffer;

ReplayBuffer *create_replay_buffer(int capacity) {
  ReplayBuffer *buffer = (ReplayBuffer *)malloc(sizeof(ReplayBuffer));
  buffer->capacity = capacity;
  buffer->size = 0;
  buffer->index = 0;
  buffer->buffer = (Experience *)malloc(capacity * sizeof(Experience));
  return buffer;
}

void add_experience(ReplayBuffer *buffer, float *state, int action,
                    float reward, float *next_state, int done) {
  Experience *exp = &buffer->buffer[buffer->index];
  memcpy(exp->state, state, STATE_SIZE * sizeof(float));
  exp->action = action;
  exp->reward = reward;
  memcpy(exp->next_state, next_state, STATE_SIZE * sizeof(float));
  exp->done = done;
  buffer->index = (buffer->index + 1) % buffer->capacity;
  if (buffer->size < buffer->capacity)
    buffer->size++;
}

// Kernel simple para forward pass (versión simplificada)
__global__ void simple_forward_kernel(float *output, float *input, float *W,
                                      float *b, int in_size, int out_size) {
  int idx = blockIdx.x;
  if (idx < out_size) {
    float sum = 0.0f;
    for (int i = 0; i < in_size; i++) {
      sum += input[i] * W[idx * in_size + i];
    }
    output[idx] = fmaxf(0.0f, sum + b[idx]); // ReLU
  }
}

__global__ void final_layer_kernel(float *output, float *input, float *W,
                                   float *b, int in_size, int out_size) {
  int idx = blockIdx.x;
  if (idx < out_size) {
    float sum = 0.0f;
    for (int i = 0; i < in_size; i++) {
      sum += input[i] * W[idx * in_size + i];
    }
    output[idx] = sum + b[idx]; // Sin activación para Q-values
  }
}
// Kernels for Backpropagation

__global__ void relu_backward_kernel(float *d_input, float *d_output,
                                     float *input, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    d_input[idx] = (input[idx] > 0.0f) ? d_output[idx] : 0.0f;
  }
}

__global__ void update_weights_kernel(float *W, float *d_W, float lr,
                                      int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    W[idx] -= lr * d_W[idx];
  }
}

__global__ void update_bias_kernel(float *b, float *d_b, float lr, int size) {
  int idx = blockIdx.x;
  if (idx < size) {
    b[idx] -= lr * d_b[idx];
  }
}

// Transpose kernel for d_W calculation (Batch=1 simplification)
__global__ void compute_gradients_kernel(float *d_W, float *d_b, float *d_out,
                                         float *input, int in_size,
                                         int out_size) {
  int row = blockIdx.x;  // out_size
  int col = threadIdx.x; // in_size

  if (row < out_size && col < in_size) {
    // d_W[row][col] = d_out[row] * input[col]
    d_W[row * in_size + col] = d_out[row] * input[col];
  }

  if (col == 0 && row < out_size) {
    d_b[row] = d_out[row];
  }
}

// Backprop through Linear layer (calculate d_Input)
__global__ void backward_linear_kernel(float *d_input, float *d_output,
                                       float *W, int in_size, int out_size) {
  int idx = blockIdx.x; // in_size
  if (idx < in_size) {
    float sum = 0.0f;
    for (int i = 0; i < out_size; i++) {
      sum += d_output[i] * W[i * in_size + idx];
    }
    d_input[idx] = sum;
  }
}

// Crear red neuronal simplificada
NeuralNetwork *create_simple_network() {
  NeuralNetwork *net = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));

#define HIDDEN_SIZE 128
#define HIDDEN_SIZE_2 64

  cudaMalloc(&net->W1, STATE_SIZE * HIDDEN_SIZE * sizeof(float));
  cudaMalloc(&net->b1, HIDDEN_SIZE * sizeof(float));
  cudaMalloc(&net->W2, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
  cudaMalloc(&net->b2, HIDDEN_SIZE * sizeof(float));
  cudaMalloc(&net->W3, HIDDEN_SIZE * HIDDEN_SIZE_2 * sizeof(float));
  cudaMalloc(&net->b3, HIDDEN_SIZE_2 * sizeof(float));
  cudaMalloc(&net->W4, HIDDEN_SIZE_2 * ACTION_SIZE * sizeof(float));
  cudaMalloc(&net->b4, ACTION_SIZE * sizeof(float));

  // GRADIENTS Allocation
  cudaMalloc(&net->d_W1, STATE_SIZE * HIDDEN_SIZE * sizeof(float));
  cudaMalloc(&net->d_b1, HIDDEN_SIZE * sizeof(float));
  cudaMalloc(&net->d_W2, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
  cudaMalloc(&net->d_b2, HIDDEN_SIZE * sizeof(float));
  cudaMalloc(&net->d_W3, HIDDEN_SIZE * HIDDEN_SIZE_2 * sizeof(float));
  cudaMalloc(&net->d_b3, HIDDEN_SIZE_2 * sizeof(float));
  cudaMalloc(&net->d_W4, HIDDEN_SIZE_2 * ACTION_SIZE * sizeof(float));
  cudaMalloc(&net->d_b4, ACTION_SIZE * sizeof(float));

  srand(time(NULL));

  // Xavier initialization for Layer 1: STATE_SIZE -> HIDDEN_SIZE
  int W1_size = STATE_SIZE * HIDDEN_SIZE;
  float *W1_host = (float *)malloc(W1_size * sizeof(float));
  float limit1 = sqrtf(6.0f / (STATE_SIZE + HIDDEN_SIZE));
  for (int i = 0; i < W1_size; i++) {
    W1_host[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * limit1;
  }
  cudaMemcpy(net->W1, W1_host, W1_size * sizeof(float), cudaMemcpyHostToDevice);
  free(W1_host);

  float *b1_host = (float *)malloc(HIDDEN_SIZE * sizeof(float));
  for (int i = 0; i < HIDDEN_SIZE; i++)
    b1_host[i] = 0.0f;
  cudaMemcpy(net->b1, b1_host, HIDDEN_SIZE * sizeof(float),
             cudaMemcpyHostToDevice);
  free(b1_host);

  // Xavier initialization for Layer 2: HIDDEN_SIZE -> HIDDEN_SIZE
  int W2_size = HIDDEN_SIZE * HIDDEN_SIZE;
  float *W2_host = (float *)malloc(W2_size * sizeof(float));
  float limit2 = sqrtf(6.0f / (HIDDEN_SIZE + HIDDEN_SIZE));
  for (int i = 0; i < W2_size; i++) {
    W2_host[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * limit2;
  }
  cudaMemcpy(net->W2, W2_host, W2_size * sizeof(float), cudaMemcpyHostToDevice);
  free(W2_host);

  float *b2_host = (float *)malloc(HIDDEN_SIZE * sizeof(float));
  for (int i = 0; i < HIDDEN_SIZE; i++)
    b2_host[i] = 0.0f;
  cudaMemcpy(net->b2, b2_host, HIDDEN_SIZE * sizeof(float),
             cudaMemcpyHostToDevice);
  free(b2_host);

  // Xavier initialization for Layer 3: HIDDEN_SIZE -> HIDDEN_SIZE_2
  int W3_size = HIDDEN_SIZE * HIDDEN_SIZE_2;
  float *W3_host = (float *)malloc(W3_size * sizeof(float));
  float limit3 = sqrtf(6.0f / (HIDDEN_SIZE + HIDDEN_SIZE_2));
  for (int i = 0; i < W3_size; i++) {
    W3_host[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * limit3;
  }
  cudaMemcpy(net->W3, W3_host, W3_size * sizeof(float), cudaMemcpyHostToDevice);
  free(W3_host);

  float *b3_host = (float *)malloc(HIDDEN_SIZE_2 * sizeof(float));
  for (int i = 0; i < HIDDEN_SIZE_2; i++)
    b3_host[i] = 0.0f;
  cudaMemcpy(net->b3, b3_host, HIDDEN_SIZE_2 * sizeof(float),
             cudaMemcpyHostToDevice);
  free(b3_host);

  // Xavier initialization for Layer 4: HIDDEN_SIZE_2 -> ACTION_SIZE
  int W4_size = HIDDEN_SIZE_2 * ACTION_SIZE;
  float *W4_host = (float *)malloc(W4_size * sizeof(float));
  float limit4 = sqrtf(6.0f / (HIDDEN_SIZE_2 + ACTION_SIZE));
  for (int i = 0; i < W4_size; i++) {
    W4_host[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * limit4;
  }
  cudaMemcpy(net->W4, W4_host, W4_size * sizeof(float), cudaMemcpyHostToDevice);
  free(W4_host);

  float *b4_host = (float *)malloc(ACTION_SIZE * sizeof(float));
  for (int i = 0; i < ACTION_SIZE; i++)
    b4_host[i] = 0.0f;
  cudaMemcpy(net->b4, b4_host, ACTION_SIZE * sizeof(float),
             cudaMemcpyHostToDevice);
  free(b4_host);

  printf("✓ Network initialized with Xavier weights\n");

  return net;
}

// Forward pass simplificado
void simple_forward(NeuralNetwork *net, float *state, float *q_values) {
  float *state_gpu, *layer1_gpu, *layer2_gpu, *layer3_gpu, *output_gpu;

#define HIDDEN_SIZE 128
#define HIDDEN_SIZE_2 64

  cudaMalloc(&state_gpu, STATE_SIZE * sizeof(float));
  cudaMalloc(&layer1_gpu, HIDDEN_SIZE * sizeof(float));
  cudaMalloc(&layer2_gpu, HIDDEN_SIZE * sizeof(float));
  cudaMalloc(&layer3_gpu, HIDDEN_SIZE_2 * sizeof(float));
  cudaMalloc(&output_gpu, ACTION_SIZE * sizeof(float));

  cudaMemcpy(state_gpu, state, STATE_SIZE * sizeof(float),
             cudaMemcpyHostToDevice);

  // Capa 1
  simple_forward_kernel<<<HIDDEN_SIZE, 1>>>(layer1_gpu, state_gpu, net->W1,
                                            net->b1, STATE_SIZE, HIDDEN_SIZE);

  // Capa 2
  simple_forward_kernel<<<HIDDEN_SIZE, 1>>>(layer2_gpu, layer1_gpu, net->W2,
                                            net->b2, HIDDEN_SIZE, HIDDEN_SIZE);

  // Capa 3
  simple_forward_kernel<<<HIDDEN_SIZE_2, 1>>>(
      layer3_gpu, layer2_gpu, net->W3, net->b3, HIDDEN_SIZE, HIDDEN_SIZE_2);

  // Capa output
  final_layer_kernel<<<ACTION_SIZE, 1>>>(output_gpu, layer3_gpu, net->W4,
                                         net->b4, HIDDEN_SIZE_2, ACTION_SIZE);

  cudaDeviceSynchronize();

  cudaMemcpy(q_values, output_gpu, ACTION_SIZE * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(state_gpu);
  cudaFree(layer1_gpu);
  cudaFree(layer2_gpu);
  cudaFree(layer3_gpu);
  cudaFree(output_gpu);
}

// Seleccionar mejor acción
int get_best_action(float *q_values) {
  int best = 0;
  float max_q = q_values[0];
  for (int i = 1; i < ACTION_SIZE; i++) {
    if (q_values[i] > max_q) {
      max_q = q_values[i];
      best = i;
    }
  }
  return best;
}

// Guardar pesos de la red
int save_network(NeuralNetwork *net, const char *filename) {
  printf("\nSaving model to %s...\n", filename);

  FILE *f = fopen(filename, "wb");
  if (!f) {
    fprintf(stderr, "Error opening file for writing: %s\n", filename);
    return -1;
  }

  // Tamaños de las capas
  int sizes[] = {STATE_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE_2,
                 ACTION_SIZE};
  fwrite(sizes, sizeof(int), 5, f);

  // Alocar memoria temporal para copiar de GPU
  float *temp;

  // Capa 1
  int W1_size = STATE_SIZE * HIDDEN_SIZE;
  temp = (float *)malloc(W1_size * sizeof(float));
  cudaMemcpy(temp, net->W1, W1_size * sizeof(float), cudaMemcpyDeviceToHost);
  fwrite(temp, sizeof(float), W1_size, f);
  free(temp);

  temp = (float *)malloc(HIDDEN_SIZE * sizeof(float));
  cudaMemcpy(temp, net->b1, HIDDEN_SIZE * sizeof(float),
             cudaMemcpyDeviceToHost);
  fwrite(temp, sizeof(float), HIDDEN_SIZE, f);
  free(temp);

  // Capa 2
  int W2_size = HIDDEN_SIZE * HIDDEN_SIZE;
  temp = (float *)malloc(W2_size * sizeof(float));
  cudaMemcpy(temp, net->W2, W2_size * sizeof(float), cudaMemcpyDeviceToHost);
  fwrite(temp, sizeof(float), W2_size, f);
  free(temp);

  temp = (float *)malloc(HIDDEN_SIZE * sizeof(float));
  cudaMemcpy(temp, net->b2, HIDDEN_SIZE * sizeof(float),
             cudaMemcpyDeviceToHost);
  fwrite(temp, sizeof(float), HIDDEN_SIZE, f);
  free(temp);

  // Capa 3
  int W3_size = HIDDEN_SIZE * HIDDEN_SIZE_2;
  temp = (float *)malloc(W3_size * sizeof(float));
  cudaMemcpy(temp, net->W3, W3_size * sizeof(float), cudaMemcpyDeviceToHost);
  fwrite(temp, sizeof(float), W3_size, f);
  free(temp);

  temp = (float *)malloc(HIDDEN_SIZE_2 * sizeof(float));
  cudaMemcpy(temp, net->b3, HIDDEN_SIZE_2 * sizeof(float),
             cudaMemcpyDeviceToHost);
  fwrite(temp, sizeof(float), HIDDEN_SIZE_2, f);
  free(temp);

  // Capa 4
  int W4_size = HIDDEN_SIZE_2 * ACTION_SIZE;
  temp = (float *)malloc(W4_size * sizeof(float));
  cudaMemcpy(temp, net->W4, W4_size * sizeof(float), cudaMemcpyDeviceToHost);
  fwrite(temp, sizeof(float), W4_size, f);
  free(temp);

  temp = (float *)malloc(ACTION_SIZE * sizeof(float));
  cudaMemcpy(temp, net->b4, ACTION_SIZE * sizeof(float),
             cudaMemcpyDeviceToHost);
  fwrite(temp, sizeof(float), ACTION_SIZE, f);
  free(temp);

  fclose(f);
  printf("✓ Model saved successfully\n");
  return 0;
}

// Cargar pesos de la red
int load_network(NeuralNetwork *net, const char *filename) {
  printf("\nLoading model from %s...\n", filename);

  FILE *f = fopen(filename, "rb");
  if (!f) {
    fprintf(stderr, "Error opening file for reading: %s\n", filename);
    return -1;
  }

  // Verificar tamaños
  int sizes[5];
  fread(sizes, sizeof(int), 5, f);

  if (sizes[0] != STATE_SIZE || sizes[4] != ACTION_SIZE) {
    fprintf(stderr, "Model size mismatch!\n");
    fclose(f);
    return -1;
  }

  // Alocar memoria temporal
  float *temp;

  // Capa 1
  int W1_size = STATE_SIZE * HIDDEN_SIZE;
  temp = (float *)malloc(W1_size * sizeof(float));
  fread(temp, sizeof(float), W1_size, f);
  cudaMemcpy(net->W1, temp, W1_size * sizeof(float), cudaMemcpyHostToDevice);
  free(temp);

  temp = (float *)malloc(HIDDEN_SIZE * sizeof(float));
  fread(temp, sizeof(float), HIDDEN_SIZE, f);
  cudaMemcpy(net->b1, temp, HIDDEN_SIZE * sizeof(float),
             cudaMemcpyHostToDevice);
  free(temp);

  // Capa 2
  int W2_size = HIDDEN_SIZE * HIDDEN_SIZE;
  temp = (float *)malloc(W2_size * sizeof(float));
  fread(temp, sizeof(float), W2_size, f);
  cudaMemcpy(net->W2, temp, W2_size * sizeof(float), cudaMemcpyHostToDevice);
  free(temp);

  temp = (float *)malloc(HIDDEN_SIZE * sizeof(float));
  fread(temp, sizeof(float), HIDDEN_SIZE, f);
  cudaMemcpy(net->b2, temp, HIDDEN_SIZE * sizeof(float),
             cudaMemcpyHostToDevice);
  free(temp);

  // Capa 3
  int W3_size = HIDDEN_SIZE * HIDDEN_SIZE_2;
  temp = (float *)malloc(W3_size * sizeof(float));
  fread(temp, sizeof(float), W3_size, f);
  cudaMemcpy(net->W3, temp, W3_size * sizeof(float), cudaMemcpyHostToDevice);
  free(temp);

  temp = (float *)malloc(HIDDEN_SIZE_2 * sizeof(float));
  fread(temp, sizeof(float), HIDDEN_SIZE_2, f);
  cudaMemcpy(net->b3, temp, HIDDEN_SIZE_2 * sizeof(float),
             cudaMemcpyHostToDevice);
  free(temp);

  // Capa 4
  int W4_size = HIDDEN_SIZE_2 * ACTION_SIZE;
  temp = (float *)malloc(W4_size * sizeof(float));
  fread(temp, sizeof(float), W4_size, f);
  cudaMemcpy(net->W4, temp, W4_size * sizeof(float), cudaMemcpyHostToDevice);
  free(temp);

  temp = (float *)malloc(ACTION_SIZE * sizeof(float));
  fread(temp, sizeof(float), ACTION_SIZE, f);
  cudaMemcpy(net->b4, temp, ACTION_SIZE * sizeof(float),
             cudaMemcpyHostToDevice);
  free(temp);

  fclose(f);
  printf("✓ Model loaded successfully\n");
  return 0;
}

// Parsear estado del cliente
void parse_state(char *buffer, float *state) {
  char *token = strtok(buffer, ",");
  int idx = 0;
  while (token != NULL && idx < STATE_SIZE) {
    state[idx++] = atof(token);
    token = strtok(NULL, ",");
  }
}

// Formatear respuesta
void format_response(char *buffer, int action, float epsilon) {
  sprintf(buffer, "%d,%.4f", action, epsilon);
}

// Backward Pass Implementation
void backward_pass(NeuralNetwork *net, float *state, float *d_output,
                   float *buffer) {
  // Allocations
  float *state_gpu, *layer1_gpu, *layer2_gpu, *layer3_gpu;
  float *d_layer1, *d_layer2, *d_layer3;

  cudaMalloc(&state_gpu, STATE_SIZE * sizeof(float));
  cudaMalloc(&layer1_gpu, HIDDEN_SIZE * sizeof(float));
  cudaMalloc(&layer2_gpu, HIDDEN_SIZE * sizeof(float));
  cudaMalloc(&layer3_gpu, HIDDEN_SIZE_2 * sizeof(float));

  cudaMalloc(&d_layer1, HIDDEN_SIZE * sizeof(float));
  cudaMalloc(&d_layer2, HIDDEN_SIZE * sizeof(float));
  cudaMalloc(&d_layer3, HIDDEN_SIZE_2 * sizeof(float));

  // 1. Re-compute Forward Pass to get activations
  cudaMemcpy(state_gpu, state, STATE_SIZE * sizeof(float),
             cudaMemcpyHostToDevice);

  simple_forward_kernel<<<HIDDEN_SIZE, 1>>>(layer1_gpu, state_gpu, net->W1,
                                            net->b1, STATE_SIZE, HIDDEN_SIZE);
  simple_forward_kernel<<<HIDDEN_SIZE, 1>>>(layer2_gpu, layer1_gpu, net->W2,
                                            net->b2, HIDDEN_SIZE, HIDDEN_SIZE);
  simple_forward_kernel<<<HIDDEN_SIZE_2, 1>>>(
      layer3_gpu, layer2_gpu, net->W3, net->b3, HIDDEN_SIZE, HIDDEN_SIZE_2);

  // 2. Backward Pass

  // Layer 4 (Output -> Layer 3)
  compute_gradients_kernel<<<ACTION_SIZE, HIDDEN_SIZE_2>>>(
      net->d_W4, net->d_b4, d_output, layer3_gpu, HIDDEN_SIZE_2, ACTION_SIZE);
  backward_linear_kernel<<<HIDDEN_SIZE_2, 1>>>(d_layer3, d_output, net->W4,
                                               HIDDEN_SIZE_2, ACTION_SIZE);
  relu_backward_kernel<<<HIDDEN_SIZE_2, 256>>>(d_layer3, d_layer3, layer3_gpu,
                                               HIDDEN_SIZE_2);

  // Layer 3 (Layer 3 -> Layer 2)
  compute_gradients_kernel<<<HIDDEN_SIZE_2, HIDDEN_SIZE>>>(
      net->d_W3, net->d_b3, d_layer3, layer2_gpu, HIDDEN_SIZE, HIDDEN_SIZE_2);
  backward_linear_kernel<<<HIDDEN_SIZE, 1>>>(d_layer2, d_layer3, net->W3,
                                             HIDDEN_SIZE, HIDDEN_SIZE_2);
  relu_backward_kernel<<<HIDDEN_SIZE, 256>>>(d_layer2, d_layer2, layer2_gpu,
                                             HIDDEN_SIZE);

  // Layer 2 (Layer 2 -> Layer 1)
  compute_gradients_kernel<<<HIDDEN_SIZE, HIDDEN_SIZE>>>(
      net->d_W2, net->d_b2, d_layer2, layer1_gpu, HIDDEN_SIZE, HIDDEN_SIZE);
  backward_linear_kernel<<<HIDDEN_SIZE, 1>>>(d_layer1, d_layer2, net->W2,
                                             HIDDEN_SIZE, HIDDEN_SIZE);
  relu_backward_kernel<<<HIDDEN_SIZE, 256>>>(d_layer1, d_layer1, layer1_gpu,
                                             HIDDEN_SIZE);

  // Layer 1 (Layer 1 -> Input)
  compute_gradients_kernel<<<HIDDEN_SIZE, STATE_SIZE>>>(
      net->d_W1, net->d_b1, d_layer1, state_gpu, STATE_SIZE, HIDDEN_SIZE);

  // 3. Update Weights (SGD)
  float lr = 0.001f; // Learning Rate
  update_weights_kernel<<<256, 256>>>(net->W1, net->d_W1, lr,
                                      STATE_SIZE * HIDDEN_SIZE);
  update_bias_kernel<<<256, 1>>>(net->b1, net->d_b1, lr, HIDDEN_SIZE);

  update_weights_kernel<<<256, 256>>>(net->W2, net->d_W2, lr,
                                      HIDDEN_SIZE * HIDDEN_SIZE);
  update_bias_kernel<<<256, 1>>>(net->b2, net->d_b2, lr, HIDDEN_SIZE);

  update_weights_kernel<<<256, 256>>>(net->W3, net->d_W3, lr,
                                      HIDDEN_SIZE * HIDDEN_SIZE_2);
  update_bias_kernel<<<256, 1>>>(net->b3, net->d_b3, lr, HIDDEN_SIZE_2);

  update_weights_kernel<<<256, 256>>>(net->W4, net->d_W4, lr,
                                      HIDDEN_SIZE_2 * ACTION_SIZE);
  update_bias_kernel<<<256, 1>>>(net->b4, net->d_b4, lr, ACTION_SIZE);

  cudaDeviceSynchronize();

  // Free vars
  cudaFree(state_gpu);
  cudaFree(layer1_gpu);
  cudaFree(layer2_gpu);
  cudaFree(layer3_gpu);
  cudaFree(d_layer1);
  cudaFree(d_layer2);
  cudaFree(d_layer3);
}

void train_step(NeuralNetwork *net, ReplayBuffer *buffer) {
  if (buffer->size < 64)
    return;

  int batch_size = 4; // Mini batch for speed
  float gamma = 0.99f;

  for (int b = 0; b < batch_size; b++) {
    int idx = rand() % buffer->size;
    Experience *exp = &buffer->buffer[idx];

    float current_q[ACTION_SIZE];
    simple_forward(net, exp->state, current_q);

    float next_q[ACTION_SIZE];
    simple_forward(net, exp->next_state, next_q);

    float max_next_q = -99999.0f;
    for (int i = 0; i < ACTION_SIZE; i++)
      if (next_q[i] > max_next_q)
        max_next_q = next_q[i];

    float target = exp->reward + (exp->done ? 0.0f : gamma * max_next_q);

    float d_output_host[ACTION_SIZE];
    for (int i = 0; i < ACTION_SIZE; i++)
      d_output_host[i] = 0.0f;

    d_output_host[exp->action] = (current_q[exp->action] - target);

    float *d_output_gpu;
    cudaMalloc(&d_output_gpu, ACTION_SIZE * sizeof(float));
    cudaMemcpy(d_output_gpu, d_output_host, ACTION_SIZE * sizeof(float),
               cudaMemcpyHostToDevice);

    backward_pass(net, exp->state, d_output_gpu, NULL);

    cudaFree(d_output_gpu);
  }
}

// Parsear feedback: FEEDBACK:state|action|reward|next_state|done
void parse_feedback(char *buffer, ReplayBuffer *replay_buffer) {
  // Ejemplo string: 0.1,0.2...|3|0.5|0.2,0.3...|0
  char *token = strtok(buffer, "|");
  if (!token)
    return;

  // 1. State
  float state[STATE_SIZE];
  char *state_str = token;
  char *val = strtok(state_str, ",");
  int i = 0;
  while (val && i < STATE_SIZE) {
    state[i++] = atof(val);
    val = strtok(NULL, ",");
  }

  // 2. Action
  token = strtok(NULL, "|");
  if (!token)
    return;
  int action = atoi(token);

  // 3. Reward
  token = strtok(NULL, "|");
  if (!token)
    return;
  float reward = atof(token);

  // 4. Next State
  token = strtok(NULL, "|");
  if (!token)
    return;
  float next_state[STATE_SIZE];

  char *next_state_str = token;
  char *cursor = next_state_str;
  for (int j = 0; j < STATE_SIZE; j++) {
    next_state[j] = strtof(cursor, &cursor);
    if (*cursor == ',')
      cursor++;
  }

  // 5. Done
  token = strtok(NULL, "|");
  if (!token)
    return;
  int done = atoi(token);

  // Agregar al buffer
  add_experience(replay_buffer, state, action, reward, next_state, done);
}

int main(int argc, char *argv[]) {
  printf("==============================================\n");
  printf("   DQN Server with CUDA C\n");
  printf("   Jetson Xavier Implementation\n");
  printf("==============================================\n\n");

  // Verificar CUDA
  int device_count;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    fprintf(stderr, "No CUDA devices found!\n");
    return 1;
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("CUDA Device: %s\n", prop.name);
  printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
  printf("Global Memory: %.2f GB\n\n", prop.totalGlobalMem / 1e9);

  // Crear red neuronal
  printf("Initializing DQN network...\n");
  NeuralNetwork *net = create_simple_network();
  // Inicializar Replay Buffer
  ReplayBuffer *replay_buffer = create_replay_buffer(100000);
  printf("✓ Network and Replay Buffer created\n\n");

  // Configurar socket
  int server_fd, client_fd;
  struct sockaddr_in address;
  int opt = 1;
  int addrlen = sizeof(address);

  // Crear socket
  if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
    perror("Socket creation failed");
    exit(EXIT_FAILURE);
  }

  // Configurar opciones del socket
  if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
    perror("setsockopt failed");
    exit(EXIT_FAILURE);
  }

  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(PORT);

  // Bind
  if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
    perror("Bind failed");
    exit(EXIT_FAILURE);
  }

  // Listen
  if (listen(server_fd, 3) < 0) {
    perror("Listen failed");
    exit(EXIT_FAILURE);
  }

  printf("Server listening on port %d\n", PORT);
  printf("Waiting for client connection...\n\n");

  // Aceptar conexión
  if ((client_fd = accept(server_fd, (struct sockaddr *)&address,
                          (socklen_t *)&addrlen)) < 0) {
    perror("Accept failed");
    exit(EXIT_FAILURE);
  }

  printf("✓ Client connected from %s\n\n", inet_ntoa(address.sin_addr));
  printf("Ready to process requests...\n");
  printf("Press Ctrl+C to stop\n\n");

  // Variables de estado
  float epsilon = 1.0f;
  int episode = 0;
  int total_requests = 0;
  int train_count = 0;

  // Loop principal
  char buffer[BUFFER_SIZE];
  while (1) {
    memset(buffer, 0, BUFFER_SIZE);

    // Recibir datos del cliente
    int valread = read(client_fd, buffer, BUFFER_SIZE);
    if (valread <= 0) {
      printf("Client disconnected\n");
      break;
    }

    total_requests++;

    // Parsear comando
    if (strncmp(buffer, "STATE:", 6) == 0) {
      // Parsear estado
      float state[STATE_SIZE];
      parse_state(buffer + 6, state);

      // Forward pass en GPU
      float q_values[ACTION_SIZE];
      simple_forward(net, state, q_values);

      // Seleccionar acción (con epsilon-greedy simple)
      int action;
      if ((float)rand() / RAND_MAX < epsilon) {
        action = rand() % ACTION_SIZE;
      } else {
        action = get_best_action(q_values);
      }

      // Enviar respuesta
      memset(buffer, 0, BUFFER_SIZE);
      format_response(buffer, action, epsilon);
      send(client_fd, buffer, strlen(buffer), 0);

      if (total_requests % 100 == 0) {
        printf("Processed %d requests (epsilon: %.4f)\n", total_requests,
               epsilon);
      }
    } else if (strncmp(buffer, "FEEDBACK:", 9) == 0) {
      parse_feedback(buffer + 9, replay_buffer);

      // Trigger training every few steps
      if (replay_buffer->size > 64 && total_requests % 4 == 0) {
        train_step(net, replay_buffer);
        train_count++;
      }

      strcpy(buffer, "OK");
      send(client_fd, buffer, strlen(buffer), 0);
    } else if (strncmp(buffer, "EPISODE_END", 11) == 0) {
      episode++;
      epsilon = fmaxf(0.01f, epsilon * 0.995f);
      printf("\n=== Episode %d completed ===\n", episode);
      printf("Replay buffer size: %d\n", replay_buffer->size);
      printf("Training steps this session: %d\n", train_count);
      printf("New epsilon: %.4f\n\n", epsilon);

      // Respuesta simple
      strcpy(buffer, "OK");
      send(client_fd, buffer, strlen(buffer), 0);
    } else if (strncmp(buffer, "SAVE", 4) == 0) {
      printf("\n=== Save request received ===\n");

      // Guardar modelo
      char model_file[256];
      sprintf(model_file, "dqn_model_ep%d.bin", episode);

      if (save_network(net, model_file) == 0) {
        printf("Model saved after %d episodes\n", episode);
        strcpy(buffer, "SAVED");
      } else {
        fprintf(stderr, "Failed to save model\n");
        strcpy(buffer, "ERROR");
      }
      send(client_fd, buffer, strlen(buffer), 0);
    } else if (strncmp(buffer, "QUIT", 4) == 0) {
      printf("Quit request received\n");
      strcpy(buffer, "BYE");
      send(client_fd, buffer, strlen(buffer), 0);
      break;
    } else {
      printf("Unknown command: %s\n", buffer);
      strcpy(buffer, "ERROR");
      send(client_fd, buffer, strlen(buffer), 0);
    }
  }

  // Cerrar conexiones
  close(client_fd);
  close(server_fd);

  // Guardar modelo final automáticamente
  printf("\n=== Saving final model ===\n");
  save_network(net, "models/dqn_model_final.bin");

  // Liberar red neuronal
  cudaFree(net->W1);
  cudaFree(net->b1);
  cudaFree(net->W2);
  cudaFree(net->b2);
  cudaFree(net->W3);
  cudaFree(net->b3);
  cudaFree(net->W4);
  cudaFree(net->b4);
  free(net);

  printf("\n========================================\n");
  printf("Server stopped\n");
  printf("========================================\n");
  printf("Total requests processed: %d\n", total_requests);
  printf("Episodes completed: %d\n", episode);
  printf("Final model saved as: models/dqn_model_final.bin\n");
  printf("========================================\n");

  return 0;
}
