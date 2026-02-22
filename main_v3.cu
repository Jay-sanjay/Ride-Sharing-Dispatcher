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

 * main_v3.cu
     - In this code the TODO left from v2 has been fixed.
     - Using Search Radius Expansion.
     - In this i start searching for drivers in the rider's current cell.
     - If no driver is found we expand out search to the neighboring cells,
     - this is done layer-by-layer until we find a driver or exhaust the entire grid.
     - If NUM_DRIVERS > NUM_RIDERS: The code is guranteed to find a driver to each rider.


 * ISSUES:
     - Memeory-Overhead:
    -However, there are memory overheads associated with maintaining the grid and the driver lists.
    - (TODO: Think if you can use bits like structure instead of grid & the driver list)

 *************************************************************************************************/


#include <stdio.h>
#include <cuda_runtime.h>
#include <limits.h>
#include <sys/time.h>

#define NUM_DRIVERS 1000000
#define NUM_RIDERS  1000000
#define BLOCK_SIZE 256

#define MAP_SIZE 1000                   // The world is 1000x1000 units
#define CELL_SIZE 10                    // Each cell is 10x10 units
#define GRID_DIM (MAP_SIZE / CELL_SIZE) // 100x100 grid
#define NUM_CELLS (GRID_DIM * GRID_DIM) // 10,000 total cells
#define MAX_CAPACITY 1024               // Max 1024 drivers per cell (prevent overflow)
// #define MAX_RADIUS 60

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

__device__ void booking_block(int *driver_id, int *driver_info, bool *got_driver)
{
    int index = (*driver_id) >> 5;
    int bit_pos = (*driver_id) & 31;

    int old_value = atomicOr(&driver_info[index], 1U << bit_pos);
    int status = old_value & (1 << bit_pos);

    if (status == 0)
    {
        *got_driver = true;
    }
    else
    {
        *got_driver = false;
    }
}

 //------------------------------------------------------------------------------
 // Kernel 2: Main matching with radius expansion (bounded).
 // Riders that succeed get their driver ID in result; others get -1.
 //------------------------------------------------------------------------------
__global__ void ride_sharing_kernel(
    int *rider_x, int *rider_y,   // rider coordinates
    int *driver_x, int *driver_y, // driver coordinates
    int *driver_info,             // the bitmask
    int *result,
    int *grid_count,         // tells number of drivers per cell
    int *grid_driver,        // the list of drivers
    int *failed_riders_count // count of riders who failed to get a ride
)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_RIDERS)
    {
        int my_x = rider_x[tid];
        int my_y = rider_y[tid];
        int my_rider_id = -1;
        bool booked = false;

        int cell_x = my_x / CELL_SIZE;
        int cell_y = my_y / CELL_SIZE;

        if (cell_x < 0)
            cell_x = 0;
        if (cell_x >= GRID_DIM)
            cell_x = GRID_DIM - 1;
        if (cell_y < 0)
            cell_y = 0;
        if (cell_y >= GRID_DIM)
            cell_y = GRID_DIM - 1;

        int search_radius = 0;
        int MAX_RADIUS = GRID_DIM;

        while (!booked && search_radius <= MAX_RADIUS)
        {
            int mini_dist = INT_MAX;
            int best_driver = -1;

            for (int dx = -search_radius; dx <= search_radius; ++dx)
            {
                for (int dy = -search_radius; dy <= search_radius; ++dy)
                {
                    int target_cell_x = cell_x + dx;
                    int target_cell_y = cell_y + dy;

                    if (target_cell_x < 0 || target_cell_x >= GRID_DIM || target_cell_y < 0 || target_cell_y >= GRID_DIM)
                        continue; // Skip out-of-bounds cells

                    int target_cell_id = target_cell_y * GRID_DIM + target_cell_x;
                    int count = grid_count[target_cell_id];
                    if (count > MAX_CAPACITY)
                        count = MAX_CAPACITY;

                    for (int k = 0; k < count; ++k)
                    { /*this loop runs for MAX_CAPACITY*/
                        int driver_id = grid_driver[target_cell_id * MAX_CAPACITY + k];

                        int idx = driver_id >> 5;
                        int pos = driver_id & 31;
                        int status = (driver_info[idx]) & (1U << pos);

                        if (status != 0)
                            continue;
                        else
                        {
                            int dx = my_x - driver_x[driver_id];
                            int dy = my_y - driver_y[driver_id];
                            int dist = dx * dx + dy * dy;
                            if (dist < mini_dist)
                            {
                                mini_dist = dist;
                                best_driver = driver_id;
                            }
                        }
                    }
                }
            }
            if (best_driver != -1)
            {
                booking_block(&best_driver, driver_info, &booked);
                if (booked)
                {
                    my_rider_id = best_driver;
                }
            }
            else
            {
                search_radius++; //(increase search radius)
            }
        }

        if (!booked)
        {
            my_rider_id = -1;
            atomicAdd(failed_riders_count, 1);
        }

        result[tid] = my_rider_id;
    }
}

 //------------------------------------------------------------------------------
 // Kernel 1: Spatial grid partitioning – store driver IDs per cell.
 //------------------------------------------------------------------------------
__global__ void spatial_grid_partitioning_kernel(
    int *driver_x, int *driver_y,
    int *grid_count,
    int *grid_driver)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_DRIVERS)
    {
        int my_driver_x = driver_x[tid];
        int my_driver_y = driver_y[tid];

        int cell_x = my_driver_x / CELL_SIZE;
        int cell_y = my_driver_y / CELL_SIZE;

        if (cell_x < GRID_DIM && cell_y < GRID_DIM)
        {
            int cell_id = cell_y * GRID_DIM + cell_x;
            int slot = atomicAdd(&grid_count[cell_id], 1);
            if (slot < MAX_CAPACITY)
            {
                int gloabal_index = cell_id * MAX_CAPACITY + slot;
                grid_driver[gloabal_index] = tid;
            }
        }
    }
}

int main()
{
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // 1. Host Memory Allocation
    int *h_rx = (int *)malloc(NUM_RIDERS * sizeof(int));
    int *h_ry = (int *)malloc(NUM_RIDERS * sizeof(int));
    int *h_dx = (int *)malloc(NUM_DRIVERS * sizeof(int));
    int *h_dy = (int *)malloc(NUM_DRIVERS * sizeof(int));
    int *h_results = (int *)malloc(NUM_RIDERS * sizeof(int));
    int *h_failed_count = (int *)malloc(sizeof(int));

    // Bitmap size: (1024 + 31) / 32 = 32 integers
    int bitmap_size_ints = (NUM_DRIVERS + 31) / 32;
    int *h_info = (int *)calloc(bitmap_size_ints, sizeof(int)); // Init to 0

    srand(42); // For reproducibility
    // Initialize random positions
    for (int i = 0; i < NUM_RIDERS; i++)
    {
        h_rx[i] = rand() % MAP_SIZE;
        h_ry[i] = rand() % MAP_SIZE;
    }
    for (int i = 0; i < NUM_DRIVERS; i++)
    {
        h_dx[i] = rand() % MAP_SIZE;
        h_dy[i] = rand() % MAP_SIZE;
    }

    // 2. Device Memory Allocation
    int *d_rx, *d_ry, *d_dx, *d_dy, *d_info, *d_res; 
    int *d_grid_count, *d_grid_driver;
    int *d_failed_riders_count;

    CHECK(cudaMalloc(&d_rx, NUM_RIDERS * sizeof(int)));
    CHECK(cudaMalloc(&d_ry, NUM_RIDERS * sizeof(int)));
    CHECK(cudaMalloc(&d_dx, NUM_DRIVERS * sizeof(int)));
    CHECK(cudaMalloc(&d_dy, NUM_DRIVERS * sizeof(int)));
    CHECK(cudaMalloc(&d_info, bitmap_size_ints * sizeof(int)));
    CHECK(cudaMalloc(&d_res, NUM_RIDERS * sizeof(int)));
    CHECK(cudaMalloc(&d_grid_count, NUM_CELLS * sizeof(int)));
    CHECK(cudaMalloc(&d_grid_driver, NUM_CELLS * MAX_CAPACITY * sizeof(int)));
    CHECK(cudaMalloc(&d_failed_riders_count, sizeof(int)));

    // Initialize grid count to 0
    CHECK(cudaMemset(d_grid_count, 0, NUM_CELLS * sizeof(int)));
    CHECK(cudaMemset(d_failed_riders_count, 0, sizeof(int)));

    // Copy Data
    CHECK(cudaMemcpy(d_rx, h_rx, NUM_RIDERS * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ry, h_ry, NUM_RIDERS * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dx, h_dx, NUM_DRIVERS * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dy, h_dy, NUM_DRIVERS * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_info, 0, bitmap_size_ints * sizeof(int))); // All drivers free

    // 3. Launch Kernel
    int threads = BLOCK_SIZE;
    int driver_blocks = (NUM_DRIVERS + threads - 1) / threads;
    int rider_blocks = (NUM_RIDERS + threads - 1) / threads;

    // Launch:1 (building spa)
    printf("Doing Spatial Grid Partitioning Kernel...\n");
    spatial_grid_partitioning_kernel<<<driver_blocks, threads>>>(d_dx, d_dy, d_grid_count, d_grid_driver);
    CHECK(cudaDeviceSynchronize());

    // Launch:2
    printf("Launching Ride Sharing Kernel...\n");
    ride_sharing_kernel<<<rider_blocks, threads>>>(d_rx, d_ry, d_dx, d_dy, d_info, d_res, d_grid_count, d_grid_driver, d_failed_riders_count);
    CHECK(cudaDeviceSynchronize());

    // 4. Verify Results
    CHECK(cudaMemcpy(h_results, d_res, NUM_RIDERS * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_failed_count, d_failed_riders_count, sizeof(int), cudaMemcpyDeviceToHost));

    int matched_count = 0;
    for (int i = 0; i < NUM_RIDERS; i++)
    {
        if (h_results[i] != -1)
            matched_count++; // driver found
    }

    printf("Simulation Complete!\n");
    printf("Total Riders: %d\n", NUM_RIDERS);
    printf("Total Drivers: %d\n", NUM_DRIVERS);
    printf("Successful Matches (Driver Found): %d\n", matched_count);
    printf("Failed Riders (No Driver Found): %d\n", *h_failed_count);
    printf("Percentage of Riders Matched: %.2f%%\n", (matched_count / (float)NUM_RIDERS) * 100.0);

    if (matched_count <= NUM_DRIVERS)
        printf("SUCCESS: No Double Booking!\n");
    else
        printf("FAILURE: Oversold drivers!\n");
    printf("Maximum Radius Searched: %d\n", GRID_DIM);

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
    cudaFree(d_grid_count);
    cudaFree(d_grid_driver);
    cudaFree(d_failed_riders_count);

    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    printf("Execution Time: %.4f seconds\n", elapsed);

    return 0;
}