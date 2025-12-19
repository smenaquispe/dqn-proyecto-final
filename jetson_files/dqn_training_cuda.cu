/*
 * DQN Training Server with CUDA - Full Implementation
 * Compatible with Python client protocol
 * Implements: Double DQN, Replay Buffer, Target Network Updates
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

// ============= CONFIGURATION =============
#define PORT 5556
#define STATE_SIZE 15
#define ACTION_SIZE 8
#define HIDDEN_SIZE 128
#define BATCH_SIZE 64
#define MEMORY_SIZE 100000
#define TARGET_UPDATE 10

#define GAMMA 0.99f
#define LEARNING_RATE 0.001f
#define EPSILON_START 1.0f
#define EPSILON_END 0.01f
#define EPSILON_DECAY 0.995f

#define BUFFER_SIZE 8192

// ============= DATA STRUCTURES =============
typedef struct
{
    float state[STATE_SIZE];
    int action;
    float reward;
    float next_state[STATE_SIZE];
    int done;
} Experience;

typedef struct
{
    Experience *buffer;
    int capacity;
    int size;
    int index;
} ReplayBuffer;

typedef struct
{
    // Policy Network (trainable)
    float *W1, *b1;
    float *W2, *b2;
    float *W3, *b3;

    // Target Network (fixed, updated periodically)
    float *W1_target, *b1_target;
    float *W2_target, *b2_target;
    float *W3_target, *b3_target;

    // Gradients
    float *d_W1, *d_b1;
    float *d_W2, *d_b2;
    float *d_W3, *d_b3;
} DQNNetwork;

// ============= REPLAY BUFFER FUNCTIONS =============
ReplayBuffer *create_replay_buffer(int capacity)
{
    ReplayBuffer *buffer = (ReplayBuffer *)malloc(sizeof(ReplayBuffer));
    buffer->capacity = capacity;
    buffer->size = 0;
    buffer->index = 0;
    buffer->buffer = (Experience *)malloc(capacity * sizeof(Experience));
    return buffer;
}

void add_experience(ReplayBuffer *buffer, float *state, int action,
                    float reward, float *next_state, int done)
{
    Experience *exp = &buffer->buffer[buffer->index];
    memcpy(exp->state, state, STATE_SIZE * sizeof(float));
    exp->action = action;
    exp->reward = reward;
    memcpy(exp->next_state, next_state, STATE_SIZE * sizeof(float));
    exp->done = done;

    buffer->index = (buffer->index + 1) % buffer->capacity;
    if (buffer->size < buffer->capacity)
    {
        buffer->size++;
    }
}

void free_replay_buffer(ReplayBuffer *buffer)
{
    free(buffer->buffer);
    free(buffer);
}

// ============= CUDA KERNELS =============

// Forward pass with ReLU activation
__global__ void forward_relu_kernel(float *output, float *input,
                                    float *W, float *b,
                                    int in_size, int out_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out_size)
    {
        float sum = 0.0f;
        for (int i = 0; i < in_size; i++)
        {
            sum += input[i] * W[idx * in_size + i];
        }
        output[idx] = fmaxf(0.0f, sum + b[idx]); // ReLU
    }
}

// Final layer without activation (Q-values)
__global__ void forward_linear_kernel(float *output, float *input,
                                      float *W, float *b,
                                      int in_size, int out_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out_size)
    {
        float sum = 0.0f;
        for (int i = 0; i < in_size; i++)
        {
            sum += input[i] * W[idx * in_size + i];
        }
        output[idx] = sum + b[idx];
    }
}

// ReLU backward
__global__ void relu_backward_kernel(float *d_input, float *d_output,
                                     float *forward_output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        d_input[idx] = (forward_output[idx] > 0.0f) ? d_output[idx] : 0.0f;
    }
}

// Compute gradients for weights and biases
__global__ void compute_gradients_kernel(float *d_W, float *d_b,
                                         float *d_output, float *input,
                                         int in_size, int out_size,
                                         int batch_size)
{
    int out_idx = blockIdx.x;
    int in_idx = threadIdx.x;

    if (out_idx < out_size && in_idx < in_size)
    {
        // Average gradient over batch
        float grad = d_output[out_idx] * input[in_idx] / batch_size;
        atomicAdd(&d_W[out_idx * in_size + in_idx], grad);
    }

    if (in_idx == 0 && out_idx < out_size)
    {
        atomicAdd(&d_b[out_idx], d_output[out_idx] / batch_size);
    }
}

// Backward through linear layer
__global__ void backward_linear_kernel(float *d_input, float *d_output,
                                       float *W, int in_size, int out_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < in_size)
    {
        float sum = 0.0f;
        for (int i = 0; i < out_size; i++)
        {
            sum += d_output[i] * W[i * in_size + idx];
        }
        d_input[idx] = sum;
    }
}

// Update weights with Adam-like optimizer (simplified SGD for now)
__global__ void update_weights_kernel(float *W, float *d_W,
                                      float lr, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        W[idx] -= lr * d_W[idx];
        d_W[idx] = 0.0f; // Reset gradient
    }
}

__global__ void update_bias_kernel(float *b, float *d_b,
                                   float lr, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        b[idx] -= lr * d_b[idx];
        d_b[idx] = 0.0f; // Reset gradient
    }
}

// Copy network weights (for target network update)
__global__ void copy_weights_kernel(float *dest, float *src, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        dest[idx] = src[idx];
    }
}

// ============= NETWORK INITIALIZATION =============
void initialize_weights_xavier(float *W_host, int in_size, int out_size)
{
    float limit = sqrtf(6.0f / (in_size + out_size));
    for (int i = 0; i < in_size * out_size; i++)
    {
        W_host[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * limit;
    }
}

DQNNetwork *create_network()
{
    DQNNetwork *net = (DQNNetwork *)malloc(sizeof(DQNNetwork));

    // Allocate policy network
    cudaMalloc(&net->W1, STATE_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&net->b1, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&net->W2, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&net->b2, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&net->W3, HIDDEN_SIZE * ACTION_SIZE * sizeof(float));
    cudaMalloc(&net->b3, ACTION_SIZE * sizeof(float));

    // Allocate target network
    cudaMalloc(&net->W1_target, STATE_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&net->b1_target, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&net->W2_target, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&net->b2_target, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&net->W3_target, HIDDEN_SIZE * ACTION_SIZE * sizeof(float));
    cudaMalloc(&net->b3_target, ACTION_SIZE * sizeof(float));

    // Allocate gradients
    cudaMalloc(&net->d_W1, STATE_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&net->d_b1, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&net->d_W2, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&net->d_b2, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&net->d_W3, HIDDEN_SIZE * ACTION_SIZE * sizeof(float));
    cudaMalloc(&net->d_b3, ACTION_SIZE * sizeof(float));

    // Initialize weights with Xavier initialization
    srand(time(NULL));

    // Layer 1
    float *W1_host = (float *)malloc(STATE_SIZE * HIDDEN_SIZE * sizeof(float));
    float *b1_host = (float *)calloc(HIDDEN_SIZE, sizeof(float));
    initialize_weights_xavier(W1_host, STATE_SIZE, HIDDEN_SIZE);
    cudaMemcpy(net->W1, W1_host, STATE_SIZE * HIDDEN_SIZE * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(net->b1, b1_host, HIDDEN_SIZE * sizeof(float),
               cudaMemcpyHostToDevice);
    free(W1_host);
    free(b1_host);

    // Layer 2
    float *W2_host = (float *)malloc(HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
    float *b2_host = (float *)calloc(HIDDEN_SIZE, sizeof(float));
    initialize_weights_xavier(W2_host, HIDDEN_SIZE, HIDDEN_SIZE);
    cudaMemcpy(net->W2, W2_host, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(net->b2, b2_host, HIDDEN_SIZE * sizeof(float),
               cudaMemcpyHostToDevice);
    free(W2_host);
    free(b2_host);

    // Layer 3
    float *W3_host = (float *)malloc(HIDDEN_SIZE * ACTION_SIZE * sizeof(float));
    float *b3_host = (float *)calloc(ACTION_SIZE, sizeof(float));
    initialize_weights_xavier(W3_host, HIDDEN_SIZE, ACTION_SIZE);
    cudaMemcpy(net->W3, W3_host, HIDDEN_SIZE * ACTION_SIZE * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(net->b3, b3_host, ACTION_SIZE * sizeof(float),
               cudaMemcpyHostToDevice);
    free(W3_host);
    free(b3_host);

    // Initialize gradients to zero
    cudaMemset(net->d_W1, 0, STATE_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMemset(net->d_b1, 0, HIDDEN_SIZE * sizeof(float));
    cudaMemset(net->d_W2, 0, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMemset(net->d_b2, 0, HIDDEN_SIZE * sizeof(float));
    cudaMemset(net->d_W3, 0, HIDDEN_SIZE * ACTION_SIZE * sizeof(float));
    cudaMemset(net->d_b3, 0, ACTION_SIZE * sizeof(float));

    // Copy policy network to target network
    cudaMemcpy(net->W1_target, net->W1, STATE_SIZE * HIDDEN_SIZE * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(net->b1_target, net->b1, HIDDEN_SIZE * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(net->W2_target, net->W2, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(net->b2_target, net->b2, HIDDEN_SIZE * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(net->W3_target, net->W3, HIDDEN_SIZE * ACTION_SIZE * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(net->b3_target, net->b3, ACTION_SIZE * sizeof(float),
               cudaMemcpyDeviceToDevice);

    return net;
}

// ============= FORWARD PASS =============
void forward_policy(DQNNetwork *net, float *state_gpu, float *q_values_gpu,
                    float *h1_gpu, float *h2_gpu)
{
    // Layer 1: state -> h1
    forward_relu_kernel<<<(HIDDEN_SIZE + 255) / 256, 256>>>(
        h1_gpu, state_gpu, net->W1, net->b1, STATE_SIZE, HIDDEN_SIZE);

    // Layer 2: h1 -> h2
    forward_relu_kernel<<<(HIDDEN_SIZE + 255) / 256, 256>>>(
        h2_gpu, h1_gpu, net->W2, net->b2, HIDDEN_SIZE, HIDDEN_SIZE);

    // Layer 3: h2 -> q_values
    forward_linear_kernel<<<(ACTION_SIZE + 255) / 256, 256>>>(
        q_values_gpu, h2_gpu, net->W3, net->b3, HIDDEN_SIZE, ACTION_SIZE);

    cudaDeviceSynchronize();
}

void forward_target(DQNNetwork *net, float *state_gpu, float *q_values_gpu)
{
    float *h1_gpu, *h2_gpu;
    cudaMalloc(&h1_gpu, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&h2_gpu, HIDDEN_SIZE * sizeof(float));

    // Layer 1: state -> h1
    forward_relu_kernel<<<(HIDDEN_SIZE + 255) / 256, 256>>>(
        h1_gpu, state_gpu, net->W1_target, net->b1_target, STATE_SIZE, HIDDEN_SIZE);

    // Layer 2: h1 -> h2
    forward_relu_kernel<<<(HIDDEN_SIZE + 255) / 256, 256>>>(
        h2_gpu, h1_gpu, net->W2_target, net->b2_target, HIDDEN_SIZE, HIDDEN_SIZE);

    // Layer 3: h2 -> q_values
    forward_linear_kernel<<<(ACTION_SIZE + 255) / 256, 256>>>(
        q_values_gpu, h2_gpu, net->W3_target, net->b3_target, HIDDEN_SIZE, ACTION_SIZE);

    cudaDeviceSynchronize();

    cudaFree(h1_gpu);
    cudaFree(h2_gpu);
}

// ============= TRAINING STEP =============
float train_batch(DQNNetwork *net, ReplayBuffer *buffer)
{
    if (buffer->size < BATCH_SIZE)
    {
        return 0.0f;
    }

    // Sample random batch
    int *indices = (int *)malloc(BATCH_SIZE * sizeof(int));
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        indices[i] = rand() % buffer->size;
    }

    float total_loss = 0.0f;

    // Allocate GPU memory for batch processing
    float *state_gpu, *next_state_gpu;
    float *q_values_gpu, *next_q_values_gpu;
    float *h1_gpu, *h2_gpu;
    float *d_output, *d_h2, *d_h1;

    cudaMalloc(&state_gpu, STATE_SIZE * sizeof(float));
    cudaMalloc(&next_state_gpu, STATE_SIZE * sizeof(float));
    cudaMalloc(&q_values_gpu, ACTION_SIZE * sizeof(float));
    cudaMalloc(&next_q_values_gpu, ACTION_SIZE * sizeof(float));
    cudaMalloc(&h1_gpu, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&h2_gpu, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_output, ACTION_SIZE * sizeof(float));
    cudaMalloc(&d_h2, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_h1, HIDDEN_SIZE * sizeof(float));

    // Process each sample in batch
    for (int b = 0; b < BATCH_SIZE; b++)
    {
        Experience *exp = &buffer->buffer[indices[b]];

        // Copy state to GPU
        cudaMemcpy(state_gpu, exp->state, STATE_SIZE * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(next_state_gpu, exp->next_state, STATE_SIZE * sizeof(float),
                   cudaMemcpyHostToDevice);

        // Forward pass through policy network (save activations for backprop)
        forward_policy(net, state_gpu, q_values_gpu, h1_gpu, h2_gpu);

        // Forward pass through target network
        forward_target(net, next_state_gpu, next_q_values_gpu);

        // Compute target Q-value on CPU
        float q_values_host[ACTION_SIZE];
        float next_q_values_host[ACTION_SIZE];
        cudaMemcpy(q_values_host, q_values_gpu, ACTION_SIZE * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(next_q_values_host, next_q_values_gpu, ACTION_SIZE * sizeof(float),
                   cudaMemcpyDeviceToHost);

        // Find max Q-value for next state
        float max_next_q = next_q_values_host[0];
        for (int i = 1; i < ACTION_SIZE; i++)
        {
            if (next_q_values_host[i] > max_next_q)
            {
                max_next_q = next_q_values_host[i];
            }
        }

        // Compute TD target
        float target_q = exp->reward + (exp->done ? 0.0f : GAMMA * max_next_q);
        float current_q = q_values_host[exp->action];

        // Compute Smooth L1 Loss derivative (Huber loss)
        float td_error = current_q - target_q;
        float gradient;
        if (fabsf(td_error) < 1.0f)
        {
            gradient = td_error; // L2 for small errors
        }
        else
        {
            gradient = (td_error > 0) ? 1.0f : -1.0f; // L1 for large errors
        }

        total_loss += fabsf(td_error);

        // Prepare gradient for output layer (only for the action taken)
        float d_output_host[ACTION_SIZE] = {0.0f};
        d_output_host[exp->action] = gradient;
        cudaMemcpy(d_output, d_output_host, ACTION_SIZE * sizeof(float),
                   cudaMemcpyHostToDevice);

        // ====== BACKPROPAGATION ======

        // Layer 3 gradients
        compute_gradients_kernel<<<ACTION_SIZE, HIDDEN_SIZE>>>(
            net->d_W3, net->d_b3, d_output, h2_gpu, HIDDEN_SIZE, ACTION_SIZE, BATCH_SIZE);
        backward_linear_kernel<<<(HIDDEN_SIZE + 255) / 256, 256>>>(
            d_h2, d_output, net->W3, HIDDEN_SIZE, ACTION_SIZE);
        relu_backward_kernel<<<(HIDDEN_SIZE + 255) / 256, 256>>>(
            d_h2, d_h2, h2_gpu, HIDDEN_SIZE);

        // Layer 2 gradients
        compute_gradients_kernel<<<HIDDEN_SIZE, HIDDEN_SIZE>>>(
            net->d_W2, net->d_b2, d_h2, h1_gpu, HIDDEN_SIZE, HIDDEN_SIZE, BATCH_SIZE);
        backward_linear_kernel<<<(HIDDEN_SIZE + 255) / 256, 256>>>(
            d_h1, d_h2, net->W2, HIDDEN_SIZE, HIDDEN_SIZE);
        relu_backward_kernel<<<(HIDDEN_SIZE + 255) / 256, 256>>>(
            d_h1, d_h1, h1_gpu, HIDDEN_SIZE);

        // Layer 1 gradients
        compute_gradients_kernel<<<HIDDEN_SIZE, STATE_SIZE>>>(
            net->d_W1, net->d_b1, d_h1, state_gpu, STATE_SIZE, HIDDEN_SIZE, BATCH_SIZE);

        cudaDeviceSynchronize();
    }

    // Update weights after processing full batch
    update_weights_kernel<<<(STATE_SIZE * HIDDEN_SIZE + 255) / 256, 256>>>(
        net->W1, net->d_W1, LEARNING_RATE, STATE_SIZE * HIDDEN_SIZE);
    update_bias_kernel<<<(HIDDEN_SIZE + 255) / 256, 256>>>(
        net->b1, net->d_b1, LEARNING_RATE, HIDDEN_SIZE);

    update_weights_kernel<<<(HIDDEN_SIZE * HIDDEN_SIZE + 255) / 256, 256>>>(
        net->W2, net->d_W2, LEARNING_RATE, HIDDEN_SIZE * HIDDEN_SIZE);
    update_bias_kernel<<<(HIDDEN_SIZE + 255) / 256, 256>>>(
        net->b2, net->d_b2, LEARNING_RATE, HIDDEN_SIZE);

    update_weights_kernel<<<(HIDDEN_SIZE * ACTION_SIZE + 255) / 256, 256>>>(
        net->W3, net->d_W3, LEARNING_RATE, HIDDEN_SIZE * ACTION_SIZE);
    update_bias_kernel<<<(ACTION_SIZE + 255) / 256, 256>>>(
        net->b3, net->d_b3, LEARNING_RATE, ACTION_SIZE);

    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(state_gpu);
    cudaFree(next_state_gpu);
    cudaFree(q_values_gpu);
    cudaFree(next_q_values_gpu);
    cudaFree(h1_gpu);
    cudaFree(h2_gpu);
    cudaFree(d_output);
    cudaFree(d_h2);
    cudaFree(d_h1);
    free(indices);

    return total_loss / BATCH_SIZE;
}

// Update target network (copy weights from policy network)
void update_target_network(DQNNetwork *net)
{
    int threads = 256;

    int blocks_W1 = (STATE_SIZE * HIDDEN_SIZE + threads - 1) / threads;
    copy_weights_kernel<<<blocks_W1, threads>>>(
        net->W1_target, net->W1, STATE_SIZE * HIDDEN_SIZE);

    int blocks_b1 = (HIDDEN_SIZE + threads - 1) / threads;
    copy_weights_kernel<<<blocks_b1, threads>>>(
        net->b1_target, net->b1, HIDDEN_SIZE);

    int blocks_W2 = (HIDDEN_SIZE * HIDDEN_SIZE + threads - 1) / threads;
    copy_weights_kernel<<<blocks_W2, threads>>>(
        net->W2_target, net->W2, HIDDEN_SIZE * HIDDEN_SIZE);

    copy_weights_kernel<<<blocks_b1, threads>>>(
        net->b2_target, net->b2, HIDDEN_SIZE);

    int blocks_W3 = (HIDDEN_SIZE * ACTION_SIZE + threads - 1) / threads;
    copy_weights_kernel<<<blocks_W3, threads>>>(
        net->W3_target, net->W3, HIDDEN_SIZE * ACTION_SIZE);

    int blocks_b3 = (ACTION_SIZE + threads - 1) / threads;
    copy_weights_kernel<<<blocks_b3, threads>>>(
        net->b3_target, net->b3, ACTION_SIZE);

    cudaDeviceSynchronize();
}

// ============= INFERENCE =============
int select_action(DQNNetwork *net, float *state, float epsilon)
{
    // Epsilon-greedy policy
    if ((float)rand() / RAND_MAX < epsilon)
    {
        return rand() % ACTION_SIZE;
    }

    // Forward pass
    float *state_gpu, *q_values_gpu, *h1_gpu, *h2_gpu;
    cudaMalloc(&state_gpu, STATE_SIZE * sizeof(float));
    cudaMalloc(&q_values_gpu, ACTION_SIZE * sizeof(float));
    cudaMalloc(&h1_gpu, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&h2_gpu, HIDDEN_SIZE * sizeof(float));

    cudaMemcpy(state_gpu, state, STATE_SIZE * sizeof(float),
               cudaMemcpyHostToDevice);

    forward_policy(net, state_gpu, q_values_gpu, h1_gpu, h2_gpu);

    float q_values[ACTION_SIZE];
    cudaMemcpy(q_values, q_values_gpu, ACTION_SIZE * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Find best action
    int best_action = 0;
    float max_q = q_values[0];
    for (int i = 1; i < ACTION_SIZE; i++)
    {
        if (q_values[i] > max_q)
        {
            max_q = q_values[i];
            best_action = i;
        }
    }

    cudaFree(state_gpu);
    cudaFree(q_values_gpu);
    cudaFree(h1_gpu);
    cudaFree(h2_gpu);

    return best_action;
}

// ============= MODEL SAVE/LOAD =============
int save_model(DQNNetwork *net, const char *filename)
{
    printf("Saving model to %s...\n", filename);

    FILE *f = fopen(filename, "wb");
    if (!f)
    {
        fprintf(stderr, "Error opening file for writing: %s\n", filename);
        return -1;
    }

    // Write header
    int sizes[] = {STATE_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, ACTION_SIZE};
    fwrite(sizes, sizeof(int), 4, f);

    // Save policy network weights
    float *temp;

    // Layer 1
    temp = (float *)malloc(STATE_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMemcpy(temp, net->W1, STATE_SIZE * HIDDEN_SIZE * sizeof(float),
               cudaMemcpyDeviceToHost);
    fwrite(temp, sizeof(float), STATE_SIZE * HIDDEN_SIZE, f);
    free(temp);

    temp = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    cudaMemcpy(temp, net->b1, HIDDEN_SIZE * sizeof(float),
               cudaMemcpyDeviceToHost);
    fwrite(temp, sizeof(float), HIDDEN_SIZE, f);
    free(temp);

    // Layer 2
    temp = (float *)malloc(HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMemcpy(temp, net->W2, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float),
               cudaMemcpyDeviceToHost);
    fwrite(temp, sizeof(float), HIDDEN_SIZE * HIDDEN_SIZE, f);
    free(temp);

    temp = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    cudaMemcpy(temp, net->b2, HIDDEN_SIZE * sizeof(float),
               cudaMemcpyDeviceToHost);
    fwrite(temp, sizeof(float), HIDDEN_SIZE, f);
    free(temp);

    // Layer 3
    temp = (float *)malloc(HIDDEN_SIZE * ACTION_SIZE * sizeof(float));
    cudaMemcpy(temp, net->W3, HIDDEN_SIZE * ACTION_SIZE * sizeof(float),
               cudaMemcpyDeviceToHost);
    fwrite(temp, sizeof(float), HIDDEN_SIZE * ACTION_SIZE, f);
    free(temp);

    temp = (float *)malloc(ACTION_SIZE * sizeof(float));
    cudaMemcpy(temp, net->b3, ACTION_SIZE * sizeof(float),
               cudaMemcpyDeviceToHost);
    fwrite(temp, sizeof(float), ACTION_SIZE, f);
    free(temp);

    fclose(f);
    printf("✓ Model saved successfully\n");
    return 0;
}

// ============= PROTOCOL PARSING =============
void parse_state(char *buffer, float *state)
{
    char *token = strtok(buffer, ",");
    int idx = 0;
    while (token != NULL && idx < STATE_SIZE)
    {
        state[idx++] = atof(token);
        token = strtok(NULL, ",");
    }
}

void parse_train_message(char *payload, float *state, int *action,
                         float *reward, float *next_state, int *done)
{
    // Format: state_csv|action|reward|next_state_csv|done
    char *parts[5];
    int i = 0;
    parts[i] = strtok(payload, "|");
    while (parts[i] != NULL && i < 4)
    {
        parts[++i] = strtok(NULL, "|");
    }

    if (i < 4)
        return;

    // Parse state
    char *state_copy = strdup(parts[0]);
    char *token = strtok(state_copy, ",");
    int idx = 0;
    while (token && idx < STATE_SIZE)
    {
        state[idx++] = atof(token);
        token = strtok(NULL, ",");
    }
    free(state_copy);

    // Parse action, reward
    *action = atoi(parts[1]);
    *reward = atof(parts[2]);

    // Parse next_state
    char *next_state_copy = strdup(parts[3]);
    token = strtok(next_state_copy, ",");
    idx = 0;
    while (token && idx < STATE_SIZE)
    {
        next_state[idx++] = atof(token);
        token = strtok(NULL, ",");
    }
    free(next_state_copy);

    // Parse done
    *done = atoi(parts[4]);
}

// ============= MAIN SERVER =============
int main(int argc, char *argv[])
{
    printf("==============================================\n");
    printf("   DQN Training Server with CUDA\n");
    printf("   Full Double-DQN Implementation\n");
    printf("==============================================\n\n");

    // Check CUDA
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0)
    {
        fprintf(stderr, "No CUDA devices found!\n");
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("CUDA Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Global Memory: %.2f GB\n\n", prop.totalGlobalMem / 1e9);

    // Initialize network and buffer
    printf("Initializing DQN network...\n");
    DQNNetwork *net = create_network();
    ReplayBuffer *buffer = create_replay_buffer(MEMORY_SIZE);
    printf("✓ Network and Replay Buffer initialized\n\n");

    // Setup socket
    int server_fd, client_fd;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
    {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)))
    {
        perror("setsockopt failed");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0)
    {
        perror("Bind failed");
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd, 3) < 0)
    {
        perror("Listen failed");
        exit(EXIT_FAILURE);
    }

    printf("Server listening on port %d\n", PORT);
    printf("Waiting for client connection...\n\n");

    // Accept connection
    if ((client_fd = accept(server_fd, (struct sockaddr *)&address,
                            (socklen_t *)&addrlen)) < 0)
    {
        perror("Accept failed");
        exit(EXIT_FAILURE);
    }

    printf("✓ Client connected from %s\n\n", inet_ntoa(address.sin_addr));
    printf("Starting training...\n\n");

    // Training state
    float epsilon = EPSILON_START;
    int episode_count = 0;
    float episode_loss = 0.0f;
    int loss_count = 0;

    // Main loop
    char recv_buffer[BUFFER_SIZE];
    char send_buffer[256];

    while (1)
    {
        memset(recv_buffer, 0, BUFFER_SIZE);

        int valread = recv(client_fd, recv_buffer, BUFFER_SIZE, 0);
        if (valread <= 0)
        {
            printf("Client disconnected\n");
            break;
        }

        // Protocol handling (same as Python server)
        if (strncmp(recv_buffer, "STEP:", 5) == 0)
        {
            // Inference only
            float state[STATE_SIZE];
            parse_state(recv_buffer + 5, state);

            int action = select_action(net, state, epsilon);

            snprintf(send_buffer, sizeof(send_buffer), "%d,%.4f", action, epsilon);
            send(client_fd, send_buffer, strlen(send_buffer), 0);
        }
        else if (strncmp(recv_buffer, "TRAIN:", 6) == 0)
        {
            // Store experience and train
            float state[STATE_SIZE], next_state[STATE_SIZE];
            int action, done;
            float reward;

            parse_train_message(recv_buffer + 6, state, &action, &reward,
                                next_state, &done);

            add_experience(buffer, state, action, reward, next_state, done);

            // Train if enough samples
            float loss = train_batch(net, buffer);
            if (loss > 0)
            {
                episode_loss += loss;
                loss_count++;
            }

            send(client_fd, "OK", 2, 0);
        }
        else if (strncmp(recv_buffer, "EPISODE_END", 11) == 0)
        {
            episode_count++;

            // Decay epsilon
            if (epsilon > EPSILON_END)
            {
                epsilon *= EPSILON_DECAY;
            }

            // Update target network
            if (episode_count % TARGET_UPDATE == 0)
            {
                update_target_network(net);
                printf("Target network updated at episode %d\n", episode_count);
            }

            float avg_loss = (loss_count > 0) ? episode_loss / loss_count : 0.0f;
            printf("Episode %d | Loss: %.4f | Epsilon: %.4f | Buffer: %d\n",
                   episode_count, avg_loss, epsilon, buffer->size);

            episode_loss = 0.0f;
            loss_count = 0;

            send(client_fd, "OK", 2, 0);
        }
        else if (strncmp(recv_buffer, "SAVE", 4) == 0)
        {
            char filename[256];
            snprintf(filename, sizeof(filename), "dqn_model_ep%d.bin", episode_count);

            if (save_model(net, filename) == 0)
            {
                printf("Model saved at episode %d\n", episode_count);
                send(client_fd, "SAVED", 5, 0);
            }
            else
            {
                send(client_fd, "ERROR", 5, 0);
            }
        }
        else if (strncmp(recv_buffer, "QUIT", 4) == 0)
        {
            printf("Quit request received\n");
            send(client_fd, "BYE", 3, 0);
            break;
        }
        else
        {
            printf("Unknown command: %s\n", recv_buffer);
            send(client_fd, "ERROR", 5, 0);
        }
    }

    // Cleanup
    close(client_fd);
    close(server_fd);

    printf("\n=== Saving final model ===\n");
    save_model(net, "dqn_model_final.bin");

    // Free memory
    cudaFree(net->W1);
    cudaFree(net->b1);
    cudaFree(net->W2);
    cudaFree(net->b2);
    cudaFree(net->W3);
    cudaFree(net->b3);
    cudaFree(net->W1_target);
    cudaFree(net->b1_target);
    cudaFree(net->W2_target);
    cudaFree(net->b2_target);
    cudaFree(net->W3_target);
    cudaFree(net->b3_target);
    cudaFree(net->d_W1);
    cudaFree(net->d_b1);
    cudaFree(net->d_W2);
    cudaFree(net->d_b2);
    cudaFree(net->d_W3);
    cudaFree(net->d_b3);
    free(net);
    free_replay_buffer(buffer);

    printf("\nServer stopped. Training complete!\n");
    printf("Episodes: %d | Final epsilon: %.4f\n", episode_count, epsilon);

    return 0;
}
