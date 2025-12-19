/*
 * Servidor de Inferencia DQN con CUDA C
 * Carga modelo entrenado y realiza solo inferencia (sin entrenamiento)
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <time.h>
#include <math.h>

#define PORT 5557 // Puerto diferente para inferencia
#define STATE_SIZE 15
#define ACTION_SIZE 8
#define BUFFER_SIZE 4096
#define HIDDEN_SIZE 128

// Red neuronal (pesos y biases) - 3 capas
typedef struct
{
    float *W1, *b1;
    float *W2, *b2;
    float *W3, *b3;
} NeuralNetwork;

// Kernel simple para forward pass
__global__ void simple_forward_kernel(float *output, float *input, float *W, float *b,
                                      int in_size, int out_size)
{
    int idx = blockIdx.x;
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

__global__ void final_layer_kernel(float *output, float *input, float *W, float *b,
                                   int in_size, int out_size)
{
    int idx = blockIdx.x;
    if (idx < out_size)
    {
        float sum = 0.0f;
        for (int i = 0; i < in_size; i++)
        {
            sum += input[i] * W[idx * in_size + i];
        }
        output[idx] = sum + b[idx]; // Sin activación para Q-values
    }
}

// Crear red neuronal (ahora con 3 capas)
NeuralNetwork *create_network()
{
    NeuralNetwork *net = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));

    cudaMalloc(&net->W1, STATE_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&net->b1, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&net->W2, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&net->b2, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&net->W3, HIDDEN_SIZE * ACTION_SIZE * sizeof(float));
    cudaMalloc(&net->b3, ACTION_SIZE * sizeof(float));

    return net;
}

// Cargar pesos de la red desde archivo
int load_network(NeuralNetwork *net, const char *filename)
{
    printf("\n=== Loading model ===\n");
    printf("File: %s\n", filename);

    FILE *f = fopen(filename, "rb");
    if (!f)
    {
        fprintf(stderr, "✗ Error opening file: %s\n", filename);
        return -1;
    }

    // Verificar tamaños (ahora son 4 valores: input -> hidden1 -> hidden2 -> output)
    int sizes[4];
    fread(sizes, sizeof(int), 4, f);

    printf("Model architecture: %d -> %d -> %d -> %d\n",
           sizes[0], sizes[1], sizes[2], sizes[3]);

    if (sizes[0] != STATE_SIZE || sizes[3] != ACTION_SIZE)
    {
        fprintf(stderr, "✗ Model size mismatch!\n");
        fprintf(stderr, "  Expected: %d -> ? -> ? -> %d\n", STATE_SIZE, ACTION_SIZE);
        fprintf(stderr, "  Got: %d -> ? -> ? -> %d\n", sizes[0], sizes[3]);
        fclose(f);
        return -1;
    }

    // Alocar memoria temporal
    float *temp;

    // Capa 1: STATE_SIZE -> HIDDEN_SIZE
    int W1_size = STATE_SIZE * HIDDEN_SIZE;
    temp = (float *)malloc(W1_size * sizeof(float));
    fread(temp, sizeof(float), W1_size, f);
    cudaMemcpy(net->W1, temp, W1_size * sizeof(float), cudaMemcpyHostToDevice);
    free(temp);

    temp = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    fread(temp, sizeof(float), HIDDEN_SIZE, f);
    cudaMemcpy(net->b1, temp, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    free(temp);

    // Capa 2: HIDDEN_SIZE -> HIDDEN_SIZE
    int W2_size = HIDDEN_SIZE * HIDDEN_SIZE;
    temp = (float *)malloc(W2_size * sizeof(float));
    fread(temp, sizeof(float), W2_size, f);
    cudaMemcpy(net->W2, temp, W2_size * sizeof(float), cudaMemcpyHostToDevice);
    free(temp);

    temp = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    fread(temp, sizeof(float), HIDDEN_SIZE, f);
    cudaMemcpy(net->b2, temp, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    free(temp);

    // Capa 3: HIDDEN_SIZE -> ACTION_SIZE
    int W3_size = HIDDEN_SIZE * ACTION_SIZE;
    temp = (float *)malloc(W3_size * sizeof(float));
    fread(temp, sizeof(float), W3_size, f);
    cudaMemcpy(net->W3, temp, W3_size * sizeof(float), cudaMemcpyHostToDevice);
    free(temp);

    temp = (float *)malloc(ACTION_SIZE * sizeof(float));
    fread(temp, sizeof(float), ACTION_SIZE, f);
    cudaMemcpy(net->b3, temp, ACTION_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    free(temp);

    fclose(f);
    printf("✓ Model loaded successfully\n");
    return 0;
}

// Forward pass (ahora con 3 capas)
void forward_pass(NeuralNetwork *net, float *state, float *q_values)
{
    float *state_gpu, *layer1_gpu, *layer2_gpu, *output_gpu;

    cudaMalloc(&state_gpu, STATE_SIZE * sizeof(float));
    cudaMalloc(&layer1_gpu, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&layer2_gpu, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&output_gpu, ACTION_SIZE * sizeof(float));

    cudaMemcpy(state_gpu, state, STATE_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Capa 1: STATE_SIZE -> HIDDEN_SIZE
    simple_forward_kernel<<<HIDDEN_SIZE, 1>>>(layer1_gpu, state_gpu, net->W1, net->b1,
                                              STATE_SIZE, HIDDEN_SIZE);

    // Capa 2: HIDDEN_SIZE -> HIDDEN_SIZE
    simple_forward_kernel<<<HIDDEN_SIZE, 1>>>(layer2_gpu, layer1_gpu, net->W2, net->b2,
                                              HIDDEN_SIZE, HIDDEN_SIZE);

    // Capa 3: HIDDEN_SIZE -> ACTION_SIZE
    final_layer_kernel<<<ACTION_SIZE, 1>>>(output_gpu, layer2_gpu, net->W3, net->b3,
                                           HIDDEN_SIZE, ACTION_SIZE);

    cudaDeviceSynchronize();

    cudaMemcpy(q_values, output_gpu, ACTION_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(state_gpu);
    cudaFree(layer1_gpu);
    cudaFree(layer2_gpu);
    cudaFree(output_gpu);
}

// Seleccionar mejor acción
int get_best_action(float *q_values)
{
    int best = 0;
    float max_q = q_values[0];
    for (int i = 1; i < ACTION_SIZE; i++)
    {
        if (q_values[i] > max_q)
        {
            max_q = q_values[i];
            best = i;
        }
    }
    return best;
}

// Parsear estado del cliente
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

// Formatear respuesta con Q-values
void format_response(char *buffer, int action, float *q_values)
{
    int offset = sprintf(buffer, "%d", action);
    for (int i = 0; i < ACTION_SIZE; i++)
    {
        offset += sprintf(buffer + offset, ",%.4f", q_values[i]);
    }
}

int main(int argc, char *argv[])
{
    printf("==============================================\n");
    printf("   DQN Inference Server with CUDA C\n");
    printf("   Jetson Xavier - Inference Only Mode\n");
    printf("==============================================\n\n");

    // Archivo de modelo (por defecto o pasado como argumento)
    const char *model_file = "dqn_model_final.bin";
    if (argc > 1)
    {
        model_file = argv[1];
    }

    // Verificar CUDA
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

    // Crear red neuronal
    printf("Creating neural network...\n");
    NeuralNetwork *net = create_network();
    printf("✓ Network structure created\n");

    // Cargar modelo
    if (load_network(net, model_file) != 0)
    {
        fprintf(stderr, "\nFailed to load model!\n");
        fprintf(stderr, "Make sure the model file exists: %s\n", model_file);
        return 1;
    }

    // Configurar socket
    int server_fd, client_fd;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    // Crear socket
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
    {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    // Configurar opciones del socket
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)))
    {
        perror("setsockopt failed");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    // Bind
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0)
    {
        perror("Bind failed");
        exit(EXIT_FAILURE);
    }

    // Listen
    if (listen(server_fd, 3) < 0)
    {
        perror("Listen failed");
        exit(EXIT_FAILURE);
    }

    printf("\n==============================================\n");
    printf("Inference server listening on port %d\n", PORT);
    printf("Model loaded: %s\n", model_file);
    printf("Waiting for client connection...\n");
    printf("==============================================\n\n");

    // Aceptar conexión
    if ((client_fd = accept(server_fd, (struct sockaddr *)&address,
                            (socklen_t *)&addrlen)) < 0)
    {
        perror("Accept failed");
        exit(EXIT_FAILURE);
    }

    printf("✓ Client connected from %s\n\n", inet_ntoa(address.sin_addr));
    printf("Ready for inference requests...\n");
    printf("Press Ctrl+C to stop\n\n");

    // Variables de estado
    int total_inferences = 0;
    int total_episodes = 0;

    // Loop principal
    char buffer[BUFFER_SIZE];
    while (1)
    {
        memset(buffer, 0, BUFFER_SIZE);

        // Recibir datos del cliente
        int valread = read(client_fd, buffer, BUFFER_SIZE);
        if (valread <= 0)
        {
            printf("Client disconnected\n");
            break;
        }

        // Parsear comando
        if (strncmp(buffer, "STATE:", 6) == 0)
        {
            // Parsear estado
            float state[STATE_SIZE];
            parse_state(buffer + 6, state);

            // Forward pass en GPU (solo inferencia)
            float q_values[ACTION_SIZE];
            forward_pass(net, state, q_values);

            // Seleccionar mejor acción (greedy, sin exploración)
            int action = get_best_action(q_values);

            total_inferences++;

            // Enviar respuesta con acción y Q-values
            memset(buffer, 0, BUFFER_SIZE);
            format_response(buffer, action, q_values);
            send(client_fd, buffer, strlen(buffer), 0);

            if (total_inferences % 50 == 0)
            {
                printf("Inferences processed: %d\n", total_inferences);
            }
        }
        else if (strncmp(buffer, "EPISODE_END", 11) == 0)
        {
            total_episodes++;
            printf("\n=== Episode %d completed ===\n", total_episodes);
            printf("Total inferences: %d\n\n", total_inferences);

            strcpy(buffer, "OK");
            send(client_fd, buffer, strlen(buffer), 0);
        }
        else if (strncmp(buffer, "QUIT", 4) == 0)
        {
            printf("Quit request received\n");
            strcpy(buffer, "BYE");
            send(client_fd, buffer, strlen(buffer), 0);
            break;
        }
        else
        {
            printf("Unknown command: %s\n", buffer);
            strcpy(buffer, "ERROR");
            send(client_fd, buffer, strlen(buffer), 0);
        }
    }

    // Cerrar conexiones
    close(client_fd);
    close(server_fd);

    // Liberar red neuronal
    cudaFree(net->W1);
    cudaFree(net->b1);
    cudaFree(net->W2);
    cudaFree(net->b2);
    cudaFree(net->W3);
    cudaFree(net->b3);
    free(net);

    printf("\n========================================\n");
    printf("Inference server stopped\n");
    printf("========================================\n");
    printf("Total inferences: %d\n", total_inferences);
    printf("Episodes completed: %d\n", total_episodes);
    printf("========================================\n");

    return 0;
}
