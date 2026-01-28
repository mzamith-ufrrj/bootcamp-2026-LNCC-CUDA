/**
 * @file gol.cu
 * @brief Implementação do Jogo da Vida (Game of Life) utilizando CUDA.
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
   
/**
 * @brief Macro para definir o tamanho do bloco
 */
#define BLOCK_SIZE 32

/**
 * @brief Macro para verificação de erros em chamadas da API CUDA.
 * @param call Chamada da função CUDA a ser verificada.
 */
#define CHECK_ERROR(call) do {                              \
   if( cudaSuccess != call) {                               \
      fprintf(stderr,"CUDA ERROR:%s in file: %s in line: ", \
         cudaGetErrorString(call),  __FILE__, __LINE__);    \
         exit(EXIT_FAILURE);                                \
   } } while (0)

/**
 * @brief Inicializa o estado do gerador de números aleatórios (cuRAND) para cada thread.
 * * @param seed Semente para a inicialização do gerador.
 * @param d_state Ponteiro para o vetor de estados do cuRAND na memória do dispositivo.
 */
__global__ void setup_rand_kernel(const unsigned long long seed,  curandState *d_state){
   const int w = blockDim.x * blockIdx.x + threadIdx.x;
   const int h = blockDim.y * blockIdx.y + threadIdx.y;
   const int k = h * gridDim.x * blockDim.y + w;
   curand_init(seed, k, 0, &d_state[k]);
}

/**
 * @brief Inicializa o tabuleiro do Jogo da Vida com valores aleatórios.
 * * @param d_state Ponteiro para os estados do cuRAND previamente inicializados.
 * @param h_buff0 Ponteiro para o buffer do tabuleiro (grid) na memória do dispositivo.
 * @param percent Probabilidade (0.0 a 1.0) de uma célula iniciar como viva (1).
 */
__global__ void init_gol_kernel(curandState *d_state, int *h_buff0, const float percent){
   const int w = blockDim.x * blockIdx.x + threadIdx.x;
   const int h = blockDim.y * blockIdx.y + threadIdx.y;
   const int k = h * gridDim.x * blockDim.y + w;
   h_buff0[k] =  (curand_uniform(&d_state[k]) < percent);
}


/**
 * @brief Kernel principal que computa a próxima geração do Jogo da Vida.
 * * Utiliza memória global para ler os vizinhos (stencil 3x3) e aplica as regras clássicas:
 * 1. Célula morta com 3 vizinhos vivos torna-se viva.
 * 2. Célula viva com 2 ou 3 vizinhos vivos permanece viva.
 * 3. Caso contrário, a célula morre ou permanece morta.
 * * @param buff1 Buffer de saída (próxima geração).
 * @param buff0 Buffer de entrada (geração atual).
 */
__global__ void GPU_Global_K(int *buff1, int *buff0){
    int i      = blockDim.x * blockIdx.x + threadIdx.x,
        j      = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned char nw = 0, n = 0,  ne = 0,
                  w  = 0, c = 0,  e  = 0,
                  sw = 0, s = 0, se  = 0,
                  sum;

    int height = gridDim.y * blockDim.y;
    int width  = gridDim.x * blockDim.x;

    int alphaJ = (int) (j-1 < 0);
    int minJ   = (alphaJ * (height-1)) + ( (1 - alphaJ) * (j-1) );
    int maxJ   = (j+1) % height;

    int alphaI = (int) (i-1 < 0);
    int minI   = (alphaI * (width-1)) + ( (1 - alphaI) * (i-1) );
    int maxI   = (i+1) % width;


    nw = buff0[minJ * width +  minI];
    n  = buff0[minJ * width  +  i];
    ne = buff0[minJ * width  +  maxI];

    w  = buff0[j    * width  +  minI];
    c  = buff0[j * width +  i];
    e  = buff0[j    * width  +  maxI];

    sw = buff0[maxJ * width  +  minI];
    s  = buff0[maxJ * width  +  i];
    se = buff0[maxJ * width  +  maxI];

    sum = nw + n + ne + w + e + sw + s + se;
    //The GOL rule
    if ((sum == 3) && (c == 0))
        buff1[j  * width  +  i] = 1;
    else if ((sum >= 2) && (sum <= 3) && (c == 1))
        buff1[j  * width  +  i] = 1;
    else
        buff1[j  * width  +  i] = 0;
}
/**
 * @brief Exibe o estado atual do tabuleiro no console (CPU).
 * * @param h_buff Ponteiro para o buffer do tabuleiro na memória do host.
 * @param width Largura do tabuleiro.
 * @param height Altura do tabuleiro.
 */
void print_gol(int *h_buff, const int width, const int height){
    printf("\n");
    for (int j = 0; j < height; j++){
        for (int i = 0; i < width; i++){
            printf(" %d", h_buff[j * width + i]);
        }
        printf("\n");
    }
}

/**
 * @brief Ponto de entrada principal do programa.
 * * Gerencia a alocação de memória no host e device, configura a grade de execução CUDA,
 * executa o loop de gerações e realiza a limpeza dos recursos.
 * * @param argc Contador de argumentos.
 * @param argv Vetor de argumentos (espera: largura, altura, passos).
 * @return int Status de saída do programa.
 */
int main (int argc, char **argv){
    int *d_buff0 = NULL,
        *d_buff1 = NULL,
        *h_buff = NULL,
         width  = atoi(argv[1]),
         height = atoi(argv[2]),
         steps  = atoi(argv[3]);
    curandState       *d_States = NULL;

    printf("\t - Domínio(%d, %d, %d)\n", width, height, steps);
    fflush(stdout);

    CHECK_ERROR(cudaDeviceReset());
    CHECK_ERROR(cudaMalloc((void**)&d_buff0,  width * height * sizeof(int)));
    CHECK_ERROR(cudaMalloc((void**)&d_buff1,  width * height * sizeof(int)));
    CHECK_ERROR(cudaMalloc((void**)&d_States, width * height * sizeof(curandState)));

    h_buff = (int*) malloc( width * height * sizeof(int));

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y, 1);
    
    //Condição inicial
    //Gerando sementes
    printf("\t - Rand\n");
    fflush(stdout);

    setup_rand_kernel<<<numBlocks, threadsPerBlock >>>(time (NULL), d_States);
    CHECK_ERROR(cudaDeviceSynchronize());
    
    //Condição inicial
    printf("\t - init condition\n");
    fflush(stdout);

    init_gol_kernel<<<numBlocks, threadsPerBlock >>>(d_States, d_buff0, 0.25);
    CHECK_ERROR(cudaDeviceSynchronize());

    
    printf("\t - Steps:\n");
    fflush(stdout);

    for (int t = 0; t < steps; t++){
        printf("\t\t B(%d, %d, %d) / T(%d, %d, %d)\n", numBlocks.x, numBlocks.y, numBlocks.z, threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.y);
        fflush(stdout);
        GPU_Global_K<<<numBlocks, threadsPerBlock >>>(d_buff1, d_buff0);
        //GPU_Shared_K<<<numBlocks, threadsPerBlock >>>(d_buff1, d_buff0);
        CHECK_ERROR(cudaDeviceSynchronize());


        int *swap = d_buff0;
        d_buff0 = d_buff1;
        d_buff1 = swap;
        printf(" %d ", t);
        fflush(stdout);

    }
    

    CHECK_ERROR(cudaMemcpy(h_buff, d_buff0,  width * height * sizeof(int), cudaMemcpyDeviceToHost));
    //print_gol(h_buff, width, height);




    CHECK_ERROR(cudaFree(d_buff0));
    CHECK_ERROR(cudaFree(d_buff1));
    free(h_buff);
    printf("\n\t\tFIM");
    return EXIT_SUCCESS;
}
