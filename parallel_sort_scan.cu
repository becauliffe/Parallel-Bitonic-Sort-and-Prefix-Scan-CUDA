/*
*   Author:     Ben McAuliffe
*   Assignment: Use CUDA to perform a bitonic sort and prefix sum
*/


#include <iostream>
#include <random>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <utility>


using namespace std;


const int MAX_BLOCK_SIZE = 1024;

struct coordinate{

public:
    float x;
    float y;
    float total;

    __init__(float x,float y,float total=0){
        this.x = x;
        this.y = y;
        this.total = total;
    }
}


void auto_throw(cudaError_t status) {
    if(status != cudaSuccess) {
        std::string message = "ERROR: '";
        message += cudaGetErrorString(status);
        message +="'\n";
        throw std::runtime_error(message);
    }
}


__global__ void scan(float *data) {
	__shared__ float local[MAX_BLOCK_SIZE];
	int gindex = threadIdx.x; //gpu index
	int index = gindex;
	local[index] = data[gindex]; // store start index
	for (int stride = 1; stride < blockDim.x; stride *= 2) {
    for (int i = index; i < blockDim.x+index; i++) {}//no
		
		__syncthreads();  // cannot be inside the if-block 'cuz everyone has to call it!
		int addend = 0;
		if (stride <= index)
			addend = local[index - stride];

		__syncthreads();
		local[index] += addend;
	}
	data[gindex] = local[index]
}


__global__ void allreduce(float *data) {
	int id = blockDim.x * blockIdx.x + threadIdx.x; // unique id
	// use this to start the loop then iterate by blocksize
	__shared__ float local[MAX_BLOCK_SIZE]; // 10x faster at least than global memory via data[]
    int gindex = threadIdx.x;
	int index = gindex;
	local[index] = data[gindex];
        for (int stride = 1; stride < blockDim.x; stride *= 2) {

		__syncthreads();  // wait for my writing partner to put his value in local before reading it
		int source = (index - stride) % blockDim.x;
		float addend = local[source];
		
		__syncthreads();  // wait for my reading partner to pull her value from local before updating it
        	local[index] += addend;
        }
	data[gindex] = local[index]; 
}


// // Upsweep
// __global__ void allreduce(float *data) {
// 	__shared__ float local[MAX_BLOCK_SIZE]; // 10x faster at least than global memory via data[]
//     int gindex = threadIdx.x;
// 	int index = gindex;
// 	local[index] = data[gindex];
//         for (int stride = 1; stride < blockDim.x; stride *= 2) {

// 		__syncthreads();  // wait for my writing partner to put his value in local before reading it
// 		int source = (index - stride) % blockDim.x;
// 		float addend = local[source];
		
// 		__syncthreads();  // wait for my reading partner to pull her value from local before updating it
//         	local[index] += addend;
//         }
// 	data[gindex] = local[index]; 
// }

// // Downsweep
// __global__ void scan(float *data) {
// 	__shared__ float local[MAX_BLOCK_SIZE];
// 	int gindex = threadIdx.x;
// 	int index = gindex;
// 	local[index] = data[gindex];
// 	for (int stride = 1; stride < blockDim.x; stride *= 2) {
		
// 		__syncthreads();  // cannot be inside the if-block 'cuz everyone has to call it!
// 		int addend = 0;
// 		if (stride <= index)
// 			addend = local[index - stride];

// 		__syncthreads();
// 		local[index] += addend;
// 	}
// 	data[gindex] = local[index];
// }

// __device__ 
void swap(coordinate *data, int a, int b) {
	coordinate temp = data[a];
	data[a] = data[b];
	data[b] = temp;
}

// __global__ 

void bitonic_sort_phase(coordinate *pairs, int j, int k){
    int i = blockDim.x * blockIdx.x + threadIdx.x; // unique id
    int ixj = i ^ j;
    if(i&j == 0){
        // Ascending
        if(pairs[i].x > pairs[ixj].x){
            swap(pairs,i,ixj);
        }
    }
    else{
        // Descending
        if(pairs[i].x < pairs[ixj].x){
            swap(pairs,i,ixj);
        }

    }
}


void bitonic_sort_step(coordinate *pairs){
    int num_blocks;
    int thds_per_block;
    int total_threads = num_blocks *  thds_per_block;
    int n = pairs.size();
    //works if total_threads == n
    // see if n is power of 2
    // get total_threads to closest upper pow of 2 to n
    // pad

    int j, k;
    // Phase
    for (int k = 2; k <= n; k *= 2) {
        // Step
        for (j=k>>1; j>0; j=j>>1) { 
            bitonic_sort_step<<<num_blocks, thds_per_block>>>(pairs, j, k);
            __syncthreads(); //unsure if needed
        }
    }
}


int main(){

    int num_blocks;
    int thds_per_block;

    int total_threads = num_blocks *  thds_per_block;



    string file_name;
    ifstream input(file_name);

    if (!input.is_open()) {
        std::cerr << "Couldn't read file: " << file_name << "\n";
        return 1; 
    }
    string line;
    
    // Get csv file size for the malloc
    int size = 0;
    while(getline(input, line)){
        size++;
    }

    input.clear();
    input.seekg(0);

    coordinate *cpu_array = new coordinate[size];
    coordinate *gpu_array;

    auto_throw(cudaMalloc(&gpu_array,size*sizeof(coordinate)));
    // double check that pairs are allowed to be cudamalloc'd

    int idx = 0;
    while(getline(input,line)){
        istringstream iss(line);
        float x,y;
        iss >> x >> y;
        coordinate coord(x,y);

        cpu_array[idx] = coord;

        idx++;

    }
    input.close();

    auto_throw(cudaMemcpy(cpu_array,gpu_array,size*sizeof(coordinate),cudaMemcpyHostToDevice));

    // end file parsing

    // sort

    bitonic_sort(gpu_array);


    // make kernel to block scan (upsweep)

    // make kernel to full scan (downsweep)



    // free other stuff too
    cudaFree(gpu_array);

    return 0;

}