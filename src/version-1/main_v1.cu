/*************************************************************************************************
 * Author:    Jay Sanjay Landge
 * E-mail:  co22btech11004@iith.ac.in
 * Roll No: CO22BTECH11004
 * Language used --> CUDA programming languauge.
 
 * Machine (system) specifications:
    --> OS: Ubuntu 22.04.1 LTS
    --> CPU: Ryzen 5 5600H @4.2 GHz 
    --> GPU: RTX 3050
    --> CPU-RAM: 16 GB
    --> GPU-RAM: 4 GB
    --> Number of Threads: 12
    --> Number of Cores: 6
    
 * main_v1. cu
     - This is the naive approach to solve the ride-sharing dispatch problem using CUDA. 
       The code simulates a scenario where we have a certain number of riders and drivers, 
       each with their own coordinates. The goal is to match each rider with the nearest available driver.

     - If the nearest driver is already booked by another rider, the code will search for the next nearest 
       driver by looping again through all DRIVERS until it finds a free DRIVER or exhausts all DRIVERS.

    - TIME COMPLEXITY: O(M*N) where M is the number of riders and N is the number of drivers.


 * ISSUES:
    - High Time Complexity: 
      The algorithm has a time complexity of O(M*N) which can be very inefficient for large values of M and N.

 *************************************************************************************************/



#include <stdio.h>
#include <cuda_runtime.h>
#include <limits.h>
#include <sys/time.h>

#define NUM_DRIVERS 1000000
#define NUM_RIDERS  1000000
#define BLOCK_SIZE  256

// Helper macro for error checking
#define CHECK(call)                                                                   \
    {                                                                                 \
        cudaError_t err = call;                                                       \
        if (err != cudaSuccess)                                                       \
        {                                                                             \
            printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1);                                                                  \
        }                                                                             \
    }

 //------------------------------------------------------------------------------
 // Device function: try to book a driver using atomic OR on the bitmask.
 // Returns true if the driver was successfully booked (bit was 0).
 //------------------------------------------------------------------------------
__device__ void booking_block(int *driver_id, int *driver_info, bool *got_driver){
    int index = (*driver_id) >> 5;
    int bit_pos = (*driver_id) & 31;

    int old_value = atomicOr(&driver_info[index], 1U << bit_pos);
    int status = old_value & (1<<bit_pos);

    if(status==0){
        *got_driver=true;
    }else{
        *got_driver=false;
    }
}

__global__ void ride_sharing_kernel(
        int *rider_x, int *rider_y,    // rider coordinates
        int *driver_x, int *driver_y,    // driver coordinates
        int *driver_info,                // the bitmask
        int *result,
        int *failed_count // riders who couldn't find a driver
    ){
    
    
    int tid = blockIdx.x *blockDim.x + threadIdx.x; //rider id
    if(tid < NUM_RIDERS){
        int my_rider_x = rider_x[tid];
        int my_rider_y = rider_y[tid];

        int my_dirver_id = -1;
        bool booked = false;

        while (!booked)
        {
            int mini_distance = INT_MAX;
            int best_driver_id = -1;

            for(int i=0; i<NUM_DRIVERS; ++i){
                int idx=i>>5, pos=i&31;
                int status = (driver_info[idx]) & (1U<<(pos));
                if(status!=0)
                    continue;
                else{
                    int distance = (my_rider_x - driver_x[i])*(my_rider_x - driver_x[i]) + 
                                   (my_rider_y - driver_y[i])*(my_rider_y - driver_y[i]);
                    if(distance < mini_distance){
                        mini_distance = distance;
                        best_driver_id = i;
                    }
                }
            }

            if(best_driver_id != -1){
                booking_block(&best_driver_id, driver_info, &booked);
                my_dirver_id = best_driver_id;
            }
            else{
                // (No drivers left)
                my_dirver_id = -1;
                booked = true;
                atomicAdd(failed_count, 1);
            }
        }
        result[tid] = my_dirver_id;
    }
}


int main() {
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // 1. Host Memory Allocation
    int *h_rx = (int*)malloc(NUM_RIDERS * sizeof(int));
    int *h_ry = (int*)malloc(NUM_RIDERS * sizeof(int));
    int *h_dx = (int*)malloc(NUM_DRIVERS * sizeof(int));
    int *h_dy = (int*)malloc(NUM_DRIVERS * sizeof(int));
    int *h_results = (int*)malloc(NUM_RIDERS * sizeof(int));
    int *h_failed_count = (int*)malloc(sizeof(int));
    
    // Bitmap size: (1024 + 31) / 32 = 32 integers
    int bitmap_size_ints = (NUM_DRIVERS + 31) / 32;
    int *h_info = (int*)calloc(bitmap_size_ints, sizeof(int)); // Init to 0

    // Initialize random positions
    for(int i=0; i<NUM_RIDERS; i++) { h_rx[i] = rand()%1000; h_ry[i] = rand()%1000; }
    for(int i=0; i<NUM_DRIVERS; i++) { h_dx[i] = rand()%1000; h_dy[i] = rand()%1000; }

    // 2. Device Memory Allocation
    int *d_rx, *d_ry, *d_dx, *d_dy, *d_info, *d_res, *d_failed_riders_count;
    CHECK(cudaMalloc(&d_rx, NUM_RIDERS * sizeof(int)));
    CHECK(cudaMalloc(&d_ry, NUM_RIDERS * sizeof(int)));
    CHECK(cudaMalloc(&d_dx, NUM_DRIVERS * sizeof(int)));
    CHECK(cudaMalloc(&d_dy, NUM_DRIVERS * sizeof(int)));
    CHECK(cudaMalloc(&d_info, bitmap_size_ints * sizeof(int)));
    CHECK(cudaMalloc(&d_res, NUM_RIDERS * sizeof(int)));
    CHECK(cudaMalloc(&d_failed_riders_count, sizeof(int)));

    // set values
    CHECK(cudaMemset(d_failed_riders_count, 0, sizeof(int)));

    // Copy Data
    CHECK(cudaMemcpy(d_rx, h_rx, NUM_RIDERS * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ry, h_ry, NUM_RIDERS * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dx, h_dx, NUM_DRIVERS * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dy, h_dy, NUM_DRIVERS * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_info, 0, bitmap_size_ints * sizeof(int))); // All drivers free

    // 3. Launch Kernel
    int threads = BLOCK_SIZE;
    int blocks = (NUM_RIDERS + threads - 1) / threads;
    
    printf("Launching Kernel with %d Riders and %d Drivers...\n", NUM_RIDERS, NUM_DRIVERS);
    ride_sharing_kernel<<<blocks, threads>>>(d_rx, d_ry, d_dx, d_dy, d_info, d_res, d_failed_riders_count);
    CHECK(cudaDeviceSynchronize());

    // 4. Verify Results
    CHECK(cudaMemcpy(h_results, d_res, NUM_RIDERS * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_failed_count, d_failed_riders_count, sizeof(int), cudaMemcpyDeviceToHost));
    
    int matched_count = 0;
    for(int i=0; i<NUM_RIDERS; i++) {
        if (h_results[i] != -1) matched_count++; // driver found
    }
    
    printf("Simulation Complete!\n");
    printf("Total Riders: %d\n", NUM_RIDERS);
    printf("Total Drivers: %d\n", NUM_DRIVERS);
    printf("Successful Matches (Driver Found): %d\n", matched_count);
    printf("Failed Riders (No Driver Found): %d\n", *h_failed_count);
    
    // Logic Check: Matches should not exceed Drivers
    if (matched_count <= NUM_DRIVERS) printf("SUCCESS: No Double Booking!\n");
    else printf("FAILURE: Oversold drivers!\n");



    // free memory
    free(h_rx);
    free(h_ry);
    free(h_dx);
    free(h_dy);
    free(h_results);
    free(h_info);
    free(h_failed_count);
    cudaFree(d_rx);
    cudaFree(d_ry);
    cudaFree(d_dx);
    cudaFree(d_dy);
    cudaFree(d_info);
    cudaFree(d_res);
    cudaFree(d_failed_riders_count);

    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    printf("Execution Time: %.4f seconds\n", elapsed);

    return 0;
}