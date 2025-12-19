/*
 * Servidor DQN con CUDA C
 * Recibe estados del cliente Python, procesa con CUDA, envía acciones
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

#define PORT 5556
#define STATE_SIZE 15
#define ACTION_SIZE 8
#define BUFFER_SIZE 4096

// Incluir las definiciones de la red neuronal del archivo principal
// (En un proyecto real, esto estaría en headers compartidos)

typedef struct
{
    float *W1, *b1;
    float *W2, *b2;
    float *W3, *b3;
    float *W4, *b4;
} NeuralNetwork;

// Kernel simple para forward pass (versión simplificada)
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

// Crear red neuronal simplificada
NeuralNetwork *create_simple_network()
{
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

    // Inicialización aleatoria simple
    float *W1_host = (float *)malloc(STATE_SIZE * HIDDEN_SIZE * sizeof(float));
    float *b1_host = (float *)malloc(HIDDEN_SIZE * sizeof(float));

    srand(time(NULL));
    float limit = sqrtf(6.0f / (STATE_SIZE + HIDDEN_SIZE));
    for (int i = 0; i < STATE_SIZE * HIDDEN_SIZE; i++)
    {
        W1_host[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * limit;
    }
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        b1_host[i] = 0.0f;
    }

    cudaMemcpy(net->W1, W1_host, STATE_SIZE * HIDDEN_SIZE * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(net->b1, b1_host, HIDDEN_SIZE * sizeof(float),
               cudaMemcpyHostToDevice);

    // Similar para otras capas (simplificado)
    cudaMemset(net->W2, 0, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMemset(net->b2, 0, HIDDEN_SIZE * sizeof(float));
    cudaMemset(net->W3, 0, HIDDEN_SIZE * HIDDEN_SIZE_2 * sizeof(float));
    cudaMemset(net->b3, 0, HIDDEN_SIZE_2 * sizeof(float));
    cudaMemset(net->W4, 0, HIDDEN_SIZE_2 * ACTION_SIZE * sizeof(float));
    cudaMemset(net->b4, 0, ACTION_SIZE * sizeof(float));

    free(W1_host);
    free(b1_host);

    return net;
}

// Forward pass simplificado
void simple_forward(NeuralNetwork *net, float *state, float *q_values)
{
    float *state_gpu, *layer1_gpu, *layer2_gpu, *layer3_gpu, *output_gpu;

#define HIDDEN_SIZE 128
#define HIDDEN_SIZE_2 64

    cudaMalloc(&state_gpu, STATE_SIZE * sizeof(float));
    cudaMalloc(&layer1_gpu, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&layer2_gpu, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&layer3_gpu, HIDDEN_SIZE_2 * sizeof(float));
    cudaMalloc(&output_gpu, ACTION_SIZE * sizeof(float));

    cudaMemcpy(state_gpu, state, STATE_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Capa 1
    simple_forward_kernel<<<HIDDEN_SIZE, 1>>>(layer1_gpu, state_gpu, net->W1, net->b1,
                                              STATE_SIZE, HIDDEN_SIZE);

    // Capa 2
    simple_forward_kernel<<<HIDDEN_SIZE, 1>>>(layer2_gpu, layer1_gpu, net->W2, net->b2,
                                              HIDDEN_SIZE, HIDDEN_SIZE);

    // Capa 3
    simple_forward_kernel<<<HIDDEN_SIZE_2, 1>>>(layer3_gpu, layer2_gpu, net->W3, net->b3,
                                                HIDDEN_SIZE, HIDDEN_SIZE_2);

    // Capa output
    final_layer_kernel<<<ACTION_SIZE, 1>>>(output_gpu, layer3_gpu, net->W4, net->b4,
                                           HIDDEN_SIZE_2, ACTION_SIZE);

    cudaDeviceSynchronize();

    cudaMemcpy(q_values, output_gpu, ACTION_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(state_gpu);
    cudaFree(layer1_gpu);
    cudaFree(layer2_gpu);
    cudaFree(layer3_gpu);
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

// Guardar pesos de la red
int save_network(NeuralNetwork *net, const char *filename)
{
    printf("\nSaving model to %s...\n", filename);

    FILE *f = fopen(filename, "wb");
    if (!f)
    {
        fprintf(stderr, "Error opening file for writing: %s\n", filename);
        return -1;
    }

    // Tamaños de las capas
    int sizes[] = {STATE_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE_2, ACTION_SIZE};
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
    cudaMemcpy(temp, net->b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(temp, sizeof(float), HIDDEN_SIZE, f);
    free(temp);

    // Capa 2
    int W2_size = HIDDEN_SIZE * HIDDEN_SIZE;
    temp = (float *)malloc(W2_size * sizeof(float));
    cudaMemcpy(temp, net->W2, W2_size * sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(temp, sizeof(float), W2_size, f);
    free(temp);

    temp = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    cudaMemcpy(temp, net->b2, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(temp, sizeof(float), HIDDEN_SIZE, f);
    free(temp);

    // Capa 3
    int W3_size = HIDDEN_SIZE * HIDDEN_SIZE_2;
    temp = (float *)malloc(W3_size * sizeof(float));
    cudaMemcpy(temp, net->W3, W3_size * sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(temp, sizeof(float), W3_size, f);
    free(temp);

    temp = (float *)malloc(HIDDEN_SIZE_2 * sizeof(float));
    cudaMemcpy(temp, net->b3, HIDDEN_SIZE_2 * sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(temp, sizeof(float), HIDDEN_SIZE_2, f);
    free(temp);

    // Capa 4
    int W4_size = HIDDEN_SIZE_2 * ACTION_SIZE;
    temp = (float *)malloc(W4_size * sizeof(float));
    cudaMemcpy(temp, net->W4, W4_size * sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(temp, sizeof(float), W4_size, f);
    free(temp);

    temp = (float *)malloc(ACTION_SIZE * sizeof(float));
    cudaMemcpy(temp, net->b4, ACTION_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(temp, sizeof(float), ACTION_SIZE, f);
    free(temp);

    fclose(f);
    printf("✓ Model saved successfully\n");
    return 0;
}

// Cargar pesos de la red
int load_network(NeuralNetwork *net, const char *filename)
{
    printf("\nLoading model from %s...\n", filename);

    FILE *f = fopen(filename, "rb");
    if (!f)
    {
        fprintf(stderr, "Error opening file for reading: %s\n", filename);
        return -1;
    }

    // Verificar tamaños
    int sizes[5];
    fread(sizes, sizeof(int), 5, f);

    if (sizes[0] != STATE_SIZE || sizes[4] != ACTION_SIZE)
    {
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
    cudaMemcpy(net->b1, temp, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    free(temp);

    // Capa 2
    int W2_size = HIDDEN_SIZE * HIDDEN_SIZE;
    temp = (float *)malloc(W2_size * sizeof(float));
    fread(temp, sizeof(float), W2_size, f);
    cudaMemcpy(net->W2, temp, W2_size * sizeof(float), cudaMemcpyHostToDevice);
    free(temp);

    temp = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    fread(temp, sizeof(float), HIDDEN_SIZE, f);
    cudaMemcpy(net->b2, temp, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    free(temp);

    // Capa 3
    int W3_size = HIDDEN_SIZE * HIDDEN_SIZE_2;
    temp = (float *)malloc(W3_size * sizeof(float));
    fread(temp, sizeof(float), W3_size, f);
    cudaMemcpy(net->W3, temp, W3_size * sizeof(float), cudaMemcpyHostToDevice);
    free(temp);

    temp = (float *)malloc(HIDDEN_SIZE_2 * sizeof(float));
    fread(temp, sizeof(float), HIDDEN_SIZE_2, f);
    cudaMemcpy(net->b3, temp, HIDDEN_SIZE_2 * sizeof(float), cudaMemcpyHostToDevice);
    free(temp);

    // Capa 4
    int W4_size = HIDDEN_SIZE_2 * ACTION_SIZE;
    temp = (float *)malloc(W4_size * sizeof(float));
    fread(temp, sizeof(float), W4_size, f);
    cudaMemcpy(net->W4, temp, W4_size * sizeof(float), cudaMemcpyHostToDevice);
    free(temp);

    temp = (float *)malloc(ACTION_SIZE * sizeof(float));
    fread(temp, sizeof(float), ACTION_SIZE, f);
    cudaMemcpy(net->b4, temp, ACTION_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    free(temp);

    fclose(f);
    printf("✓ Model loaded successfully\n");
    return 0;
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

// Formatear respuesta
void format_response(char *buffer, int action, float epsilon)
{
    sprintf(buffer, "%d,%.4f", action, epsilon);
}

int main(int argc, char *argv[])
{
    printf("==============================================\n");
    printf("   DQN Server with CUDA C\n");
    printf("   Jetson Xavier Implementation\n");
    printf("==============================================\n\n");

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
    printf("Initializing DQN network...\n");
    NeuralNetwork *net = create_simple_network();
    printf("✓ Network created\n\n");

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

    printf("Server listening on port %d\n", PORT);
    printf("Waiting for client connection...\n\n");

    // Aceptar conexión
    if ((client_fd = accept(server_fd, (struct sockaddr *)&address,
                            (socklen_t *)&addrlen)) < 0)
    {
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

        total_requests++;

        // Parsear comando
        if (strncmp(buffer, "STATE:", 6) == 0)
        {
            // Parsear estado
            float state[STATE_SIZE];
            parse_state(buffer + 6, state);

            // Forward pass en GPU
            float q_values[ACTION_SIZE];
            simple_forward(net, state, q_values);

            // Seleccionar acción (con epsilon-greedy simple)
            int action;
            if ((float)rand() / RAND_MAX < epsilon)
            {
                action = rand() % ACTION_SIZE;
            }
            else
            {
                action = get_best_action(q_values);
            }

            // Enviar respuesta
            memset(buffer, 0, BUFFER_SIZE);
            format_response(buffer, action, epsilon);
            send(client_fd, buffer, strlen(buffer), 0);

            if (total_requests % 100 == 0)
            {
                printf("Processed %d requests (epsilon: %.4f)\n",
                       total_requests, epsilon);
            }
        }
        else if (strncmp(buffer, "EPISODE_END", 11) == 0)
        {
            episode++;
            epsilon = fmaxf(0.01f, epsilon * 0.995f);
            printf("\n=== Episode %d completed ===\n", episode);
            printf("New epsilon: %.4f\n\n", epsilon);

            // Respuesta simple
            strcpy(buffer, "OK");
            send(client_fd, buffer, strlen(buffer), 0);
        }
        else if (strncmp(buffer, "SAVE", 4) == 0)
        {
            printf("\n=== Save request received ===\n");

            // Guardar modelo
            char model_file[256];
            sprintf(model_file, "dqn_model_ep%d.bin", episode);

            if (save_network(net, model_file) == 0)
            {
                printf("Model saved after %d episodes\n", episode);
                strcpy(buffer, "SAVED");
            }
            else
            {
                fprintf(stderr, "Failed to save model\n");
                strcpy(buffer, "ERROR");
            }
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
