/*
*   Author:     Ben McAuliffe
*   Assignment: Use CUDA to perform a bitonic sort and prefix sum
*   
*   Confirmed works on sizes up to 2^20
*   input a csv file with x y pairs
*   output a csv file with x, y, scan total, original position (0 based)
*
*   run on Pascal class NVIDIA card with compute capability of 6.1.
*/


#include <iostream>
#include <random>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <utility>
#include <cmath>
#include<limits>
#include <algorithm>


using namespace std;


const size_t MAX_BLOCK_SIZE = 1024;

// The Coordinate class which holds all necessary data
class Coordinate{

    public:
        float x;
        float y;
        float total_of_totals;
        float total;
        size_t row_num;


    Coordinate(float x,float y, size_t row_num = 0){
        this->x = x;
        this->y = y;
        this->total = y;
        this->total_of_totals = 0;
        this->row_num = row_num;
    }
    Coordinate()=default;
};

// Stolen autothrow for cuda errors
void auto_throw(cudaError_t status) {
    if(status != cudaSuccess) {
        std::string message = "ERROR: '";
        message += cudaGetErrorString(status);
        message +="'\n";
        throw std::runtime_error(message);
    }
}

// Start Prefix Scan

// Scan by block
__global__ 
void scan(Coordinate *data) {
    size_t id = blockDim.x * blockIdx.x + threadIdx.x; // unique id.. 1 to 1 with elements
	__shared__ Coordinate local[MAX_BLOCK_SIZE];
	size_t index = threadIdx.x;
	local[index].total = data[id].total; // store start index
	for (size_t stride = 1; stride < blockDim.x; stride *= 2) {
		
		__syncthreads();  // cannot be inside the if-block 'cuz everyone has to call it!
		float addend = 0;
		if (stride <= index)
			addend = local[index - stride].total; 

		__syncthreads();
		local[index].total += addend;
	}
	data[id].total = local[index].total;
}

// Does a scan over the last element in each block
__global__ 
void scan_of_scans(Coordinate *data) {
    size_t id = threadIdx.x; // unique id.. 1 to 1 with elements
	__shared__ Coordinate local[MAX_BLOCK_SIZE]; // double check we dont go past end
	size_t index = threadIdx.x;
	local[index].total_of_totals = data[(id+1)*blockDim.x -1].total; // store start index

    for (size_t stride = 1; stride < blockDim.x; stride *= 2) {
		
		__syncthreads();  // cannot be inside the if-block 'cuz everyone has to call it!
		float addend = 0;
		if (stride <= index)
			addend = local[index - stride].total_of_totals; 

		__syncthreads();
		local[index].total_of_totals += addend;
	}
	data[(id+1)*blockDim.x -1].total_of_totals = local[index].total_of_totals;
}

// Distributes the scan of scans over each block thus finishing the prefix scan
__global__ 
void finalize_prefix(Coordinate *data) {
    size_t id = blockDim.x * blockIdx.x + threadIdx.x; // unique id.. 1 to 1 with elements

    if(blockIdx.x != 0){
        data[id].total += data[blockDim.x * blockIdx.x - 1].total_of_totals;
    }
}

// End Prefix Scan


// Start Bitonic

__device__ 
void swap(Coordinate *data, size_t a, size_t b) {
	Coordinate temp = data[a];
	data[a] = data[b];
	data[b] = temp;
}

__global__ 
void bitonic_sort_step(Coordinate *data, size_t j, size_t k){
    
    size_t i = blockDim.x * blockIdx.x + threadIdx.x; // unique id
    size_t ixj = i ^ j;
    if (ixj > i) {

        if((i & k) == 0){
            // Ascending

            if(data[i].x > data[ixj].x){
                swap(data,i,ixj);
            }
        }
        if((i & k) != 0){
            // Descending
            if(data[i].x < data[ixj].x){
                swap(data,i,ixj);
            }
        }
        __syncthreads();
    }
}

void bitonic_sort_phase(Coordinate *data,size_t n){
    size_t block_size = MAX_BLOCK_SIZE;
    if(n < MAX_BLOCK_SIZE){
        block_size = n;
    }

    size_t num_blocks = n/block_size; // as long as n is greater than 1024

    //works if total_threads == n
    // printf("blocksize: %d\n", block_size);
    // printf("numblocks: %d\n", num_blocks);

    // Phase
    for (size_t k = 2; k <= n; k *= 2) {
        // Step
        for(size_t j = k/2; j > 0; j /= 2) { 
            bitonic_sort_step<<<num_blocks, block_size>>>(data, j, k);
        	cudaDeviceSynchronize();

        }
    }
}

// End bitonic

// Finds next highest power of 2 for a 32b int
size_t next_highest_pow2(size_t value){
    value--;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;

    value++;
    return value;
}

// Printing Functions:
void printArrayY(Coordinate *data, size_t n, string title, size_t m=8) {
	cout << title << ":";
	for (size_t i = 0; i < m; i++)
		cout << " " << data[i].y;
	cout << " ...";
	for (size_t i = n - m; i < n; i++)
		cout << " " << data[i].y;
	cout << endl;
}
void printArrayX(Coordinate *data, size_t n, string title, size_t m=8) {
	cout << title << ":";
	for (size_t i = 0; i < m; i++)
		cout << " " << data[i].x;
	cout << " ...";
	for (size_t i = n - m; i < n; i++)
		cout << " " << data[i].x;
	cout << endl;
}
void printArray_tot(Coordinate *data, size_t n, string title, size_t m=8) {
	cout << title << ":";
	for (size_t i = 0; i < m; i++)
		cout << " " << data[i].total_of_totals;
	cout << " ...";
	for (size_t i = n - m; i < n; i++)
		cout << " " << data[i].total_of_totals;
	cout << endl;
}
void printArray_total(Coordinate *data, size_t n, string title, size_t m=8) {
	cout << title << ":";
	for (size_t i = 0; i < m; i++)
		cout << " " << data[i].total;
	cout << " ...";
	for (size_t i = n - m; i < n; i++)
		cout << " " << data[i].total;
	cout << endl;
}

void create_csv(string file_name, Coordinate *data, size_t file_size){
    ofstream output;

    string file_string = file_name + ".csv";

    output.open(file_string);

    output << "x value, y value, cumulative y value, original row number\n";

    for(size_t i = 0; i < file_size; i++){
        output << data[i].x <<','<< data[i].y <<','<< data[i].total <<','<< data[i].row_num << '\n';
    }

    output.close();

}


int main(int argc, char** argv){

    if(argc != 2){
        std::cerr << "Improper run command: must be in the form of $./p6 filename.csv \n";
        return 1;
    }

    size_t num_blocks;
    string file_name = argv[1];
    //string file_name = "x_y/x_y.csv";
    ifstream input(file_name);

    if (!input.is_open()) {
        std::cerr << "Couldn't read file: " << file_name << "\n";
        return 1; 
    }
    string line;
    
    // Get csv file size for the malloc
    size_t file_size = 0;
    getline(input, line);
    while(getline(input, line)){
        file_size++;
    }
    // Reset file pointer
    input.clear();
    input.seekg(0);


    size_t size = next_highest_pow2(file_size);
    size_t block_size = MAX_BLOCK_SIZE;
    if(size < MAX_BLOCK_SIZE){
        block_size = size;
    }
    Coordinate *coord_array;


    auto_throw(cudaMallocManaged(&coord_array,size*sizeof(Coordinate)));
    // double check that data are allowed to be cudamalloc'd

    size_t idx = 0;
    getline(input, line);
    while(getline(input,line)){
        istringstream iss(line);
        float x,y;
        char comma;
        iss >> x >> comma >> y;
        Coordinate coord(x,y,idx);

        coord_array[idx] = coord;

        idx++;
    }

    for(;idx < size; idx++){
        Coordinate coord(numeric_limits<float>::infinity(),0);

        coord_array[idx] = coord;
    }

    input.close();

    // end file parsing

    // sort



    bitonic_sort_phase(coord_array,size);
	cudaDeviceSynchronize();

    num_blocks = size/block_size;


    // scan by block size
    scan<<<num_blocks, block_size>>>(coord_array);
    cudaDeviceSynchronize();

    // get totals of each block

    scan_of_scans<<<num_blocks, block_size>>>(coord_array);

    cudaDeviceSynchronize();

    // distribute totals
    finalize_prefix<<<num_blocks, block_size>>>(coord_array);
	cudaDeviceSynchronize();

    create_csv("x_y_scan", coord_array, file_size);



    // free other stuff too
    cudaFree(coord_array);

    return 0;

}
