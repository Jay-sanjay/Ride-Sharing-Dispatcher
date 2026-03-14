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
    
 * main_v5.cu – Ultra‑optimized ride‑sharing dispatch with:
 *   - Driver coordinates stored directly in grid cells (avoids random access)

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
 
 #define MAP_SIZE 1000
 #define CELL_SIZE 10
 #define GRID_DIM (MAP_SIZE / CELL_SIZE) // 100
 #define NUM_CELLS (GRID_DIM * GRID_DIM) // 10,000
 #define MAX_CAPACITY 512
 #define MAX_RADIUS 5 // search 2 cells out (5×5 = 25 cells)
 
 #define CHECK(call)                                                        \
     {                                                                      \
         cudaError_t err = call;                                            \
         if (err != cudaSuccess)                                            \
         {                                                                  \
             printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), \
                    __LINE__);                                              \
             exit(1);                                                       \
         }                                                                  \
     }
 
 //------------------------------------------------------------------------------
 // Device function: try to book a driver using atomic OR on the bitmask.
 // Returns true if the driver was successfully booked (bit was 0).
 //------------------------------------------------------------------------------
 __device__ bool book_driver(int driver_id, int *driver_info)
 {
     int idx = driver_id >> 5;
     int bit = driver_id & 31;
     int mask = 1U << bit;
     int old = atomicOr(&driver_info[idx], mask);
     return (old & mask) == 0;
 }
 
 //------------------------------------------------------------------------------
 // Kernel 1: Spatial grid partitioning – store driver ID and coordinates per cell.
 //------------------------------------------------------------------------------
 __global__ void spatial_grid_partitioning_kernel(
     const int *driver_x, const int *driver_y,
     int *grid_count,     // [NUM_CELLS]
     int *grid_driver_id, // [NUM_CELLS * MAX_CAPACITY]
     int *grid_driver_x,  // [NUM_CELLS * MAX_CAPACITY]
     int *grid_driver_y)  // [NUM_CELLS * MAX_CAPACITY]
 {
     int tid = blockIdx.x * blockDim.x + threadIdx.x;
     if (tid >= NUM_DRIVERS)
         return;
 
     int x = driver_x[tid];
     int y = driver_y[tid];
 
     int cell_x = x / CELL_SIZE;
     int cell_y = y / CELL_SIZE;
     if (cell_x >= GRID_DIM)
         cell_x = GRID_DIM - 1;
     if (cell_y >= GRID_DIM)
         cell_y = GRID_DIM - 1;
 
     int cell_id = cell_y * GRID_DIM + cell_x;
     int slot = atomicAdd(&grid_count[cell_id], 1);
     if (slot < MAX_CAPACITY)
     {
         int base = cell_id * MAX_CAPACITY;
         grid_driver_id[base + slot] = tid;
         grid_driver_x[base + slot] = x;
         grid_driver_y[base + slot] = y;
     }
 }
 
 //------------------------------------------------------------------------------
 // Kernel 2: Main matching with radius expansion (bounded by MAX_RADIUS).
 // Uses coordinates stored directly in the grid – no random access to driver arrays.
 //------------------------------------------------------------------------------
 __global__ void ride_sharing_kernel(
     const int *rider_x, const int *rider_y,
     int *driver_info,
     int *result,
     const int *grid_count,
     const int *grid_driver_id,
     const int *grid_driver_x,
     const int *grid_driver_y)
 {
     int tid = blockIdx.x * blockDim.x + threadIdx.x;
     if (tid >= NUM_RIDERS)
         return;
 
     int rx = rider_x[tid];
     int ry = rider_y[tid];
 
     // Home cell
     int home_cx = rx / CELL_SIZE;
     int home_cy = ry / CELL_SIZE;
     if (home_cx < 0)
         home_cx = 0;
     if (home_cx >= GRID_DIM)
         home_cx = GRID_DIM - 1;
     if (home_cy < 0)
         home_cy = 0;
     if (home_cy >= GRID_DIM)
         home_cy = GRID_DIM - 1;
 
     int best_driver = -1;
     int best_dist = INT_MAX;
     bool booked = false;
 
     // Expand radius layer by layer (Chebyshev distance)
     for (int r = 0; r <= MAX_RADIUS && !booked; ++r)
     {
         // Scan only cells at distance exactly r (perimeter)
         for (int dx = -r; dx <= r; ++dx)
         {
             for (int dy = -r; dy <= r; ++dy)
             {
                 if (abs(dx) != r && abs(dy) != r)
                     continue; // skip inner cells
 
                 int cx = home_cx + dx;
                 int cy = home_cy + dy;
                 if (cx < 0 || cx >= GRID_DIM || cy < 0 || cy >= GRID_DIM)
                     continue;
 
                 int cell_id = cy * GRID_DIM + cx;
                 int count = grid_count[cell_id];
                 if (count > MAX_CAPACITY)
                     count = MAX_CAPACITY;
 
                 int base = cell_id * MAX_CAPACITY;
                 for (int k = 0; k < count; ++k)
                 {
                     int driver_id = grid_driver_id[base + k];
 
                     // Quick skip if already booked
                     int idx = driver_id >> 5;
                     int bit = driver_id & 31;
                     if (driver_info[idx] & (1U << bit))
                         continue;
 
                     int dx_coord = rx - grid_driver_x[base + k];
                     int dy_coord = ry - grid_driver_y[base + k];
                     int dist = dx_coord * dx_coord + dy_coord * dy_coord;
 
                     if (dist < best_dist)
                     {
                         best_dist = dist;
                         best_driver = driver_id;
                     }
                 }
             }
         }
 
         // After scanning this layer, try to book the best found so far
         if (best_driver != -1)
         {
             if (book_driver(best_driver, driver_info))
             {
                 result[tid] = best_driver;
                 booked = true;
             }
             else
             {
                 // Booking failed (someone else took it). Reset and continue scanning.
                 best_driver = -1;
                 best_dist = INT_MAX;
                 // Continue to next radius (failed driver now marked booked)
             }
         }
     }
 
     if (!booked)
     {
         result[tid] = -1;
     }
 }
 
 //------------------------------------------------------------------------------
 // Kernel 3: Fallback global search for riders who failed in the grid search.
 // Processes only the riders whose indices are given in 'failed_indices'.
 //------------------------------------------------------------------------------
 __global__ void fallback_global_kernel(
     const int *rider_x, const int *rider_y,
     const int *driver_x, const int *driver_y,
     int *driver_info,
     int *result,
     const int *failed_indices,
     int num_failed)
 {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx >= num_failed)
         return;
 
     int rider_id = failed_indices[idx];
     int rx = rider_x[rider_id];
     int ry = rider_y[rider_id];
 
     int best_driver = -1;
     int best_dist = INT_MAX;
     bool booked = false;
 
     while (!booked)
     {
         best_driver = -1;
         best_dist = INT_MAX;
 
         // Linear scan over all drivers
         for (int d = 0; d < NUM_DRIVERS; ++d)
         {
             int idx_bit = d >> 5;
             int bit = d & 31;
             if (driver_info[idx_bit] & (1U << bit))
                 continue; // booked
 
             int dx = rx - driver_x[d];
             int dy = ry - driver_y[d];
             int dist = dx * dx + dy * dy;
             if (dist < best_dist)
             {
                 best_dist = dist;
                 best_driver = d;
             }
         }
 
         if (best_driver == -1)
         {
             // No drivers left at all
             result[rider_id] = -1;
             return;
         }
 
         if (book_driver(best_driver, driver_info))
         {
             result[rider_id] = best_driver;
             booked = true;
         }
         // else loop again (failed driver now marked booked)
     }
 }
 
 //------------------------------------------------------------------------------
 // Main
 //------------------------------------------------------------------------------
 int main()
 {
     struct timeval start, end;
     gettimeofday(&start, NULL);
 
     // Host allocations
     int *h_rx = (int *)malloc(NUM_RIDERS * sizeof(int));
     int *h_ry = (int *)malloc(NUM_RIDERS * sizeof(int));
     int *h_dx = (int *)malloc(NUM_DRIVERS * sizeof(int));
     int *h_dy = (int *)malloc(NUM_DRIVERS * sizeof(int));
     int *h_results = (int *)malloc(NUM_RIDERS * sizeof(int));
 
     int bitmap_size_ints = (NUM_DRIVERS + 31) / 32;
     int *h_info = (int *)calloc(bitmap_size_ints, sizeof(int));
 
     srand(42);
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
 
     // Device allocations
     int *d_rx, *d_ry, *d_dx, *d_dy, *d_info, *d_res;
     int *d_grid_count, *d_grid_driver_id, *d_grid_driver_x, *d_grid_driver_y;
     int *d_failed_indices;
 
     CHECK(cudaMalloc(&d_rx, NUM_RIDERS * sizeof(int)));
     CHECK(cudaMalloc(&d_ry, NUM_RIDERS * sizeof(int)));
     CHECK(cudaMalloc(&d_dx, NUM_DRIVERS * sizeof(int)));
     CHECK(cudaMalloc(&d_dy, NUM_DRIVERS * sizeof(int)));
     CHECK(cudaMalloc(&d_info, bitmap_size_ints * sizeof(int)));
     CHECK(cudaMalloc(&d_res, NUM_RIDERS * sizeof(int)));
     CHECK(cudaMalloc(&d_grid_count, NUM_CELLS * sizeof(int)));
     CHECK(cudaMalloc(&d_grid_driver_id, NUM_CELLS * MAX_CAPACITY * sizeof(int)));
     CHECK(cudaMalloc(&d_grid_driver_x, NUM_CELLS * MAX_CAPACITY * sizeof(int)));
     CHECK(cudaMalloc(&d_grid_driver_y, NUM_CELLS * MAX_CAPACITY * sizeof(int)));
     CHECK(cudaMalloc(&d_failed_indices, NUM_RIDERS * sizeof(int)));
 
     // Copy data to device
     CHECK(cudaMemcpy(d_rx, h_rx, NUM_RIDERS * sizeof(int), cudaMemcpyHostToDevice));
     CHECK(cudaMemcpy(d_ry, h_ry, NUM_RIDERS * sizeof(int), cudaMemcpyHostToDevice));
     CHECK(cudaMemcpy(d_dx, h_dx, NUM_DRIVERS * sizeof(int), cudaMemcpyHostToDevice));
     CHECK(cudaMemcpy(d_dy, h_dy, NUM_DRIVERS * sizeof(int), cudaMemcpyHostToDevice));
     CHECK(cudaMemset(d_info, 0, bitmap_size_ints * sizeof(int)));
     CHECK(cudaMemset(d_grid_count, 0, NUM_CELLS * sizeof(int)));
 
     int threads = BLOCK_SIZE;
     int driver_blocks = (NUM_DRIVERS + threads - 1) / threads;
     int rider_blocks = (NUM_RIDERS + threads - 1) / threads;
 
     // Step 1: Build spatial grid with coordinates
     printf("Building spatial grid (storing coordinates directly)...\n");
     spatial_grid_partitioning_kernel<<<driver_blocks, threads>>>(d_dx, d_dy, d_grid_count, d_grid_driver_id, d_grid_driver_x, d_grid_driver_y);
     CHECK(cudaDeviceSynchronize());
 
     // Step 2: Main matching with radius expansion (MAX_RADIUS=2)
     printf("Running radius‑limited matching (MAX_RADIUS=%d)...\n", MAX_RADIUS);
     ride_sharing_kernel<<<rider_blocks, threads>>>(d_rx, d_ry, d_info, d_res, d_grid_count, d_grid_driver_id, d_grid_driver_x, d_grid_driver_y);
     CHECK(cudaDeviceSynchronize());
 
     // Copy results back to host
     CHECK(cudaMemcpy(h_results, d_res, NUM_RIDERS * sizeof(int), cudaMemcpyDeviceToHost));
 
     // Count matches and compact failed riders
     int matched = 0;
     for (int i = 0; i < NUM_RIDERS; ++i)
     {
         if (h_results[i] != -1)
             matched++;
     }
     double match_percent = 100.0 * matched / NUM_RIDERS;
     printf("After radius search: %.6f %% matched, %.6f %% failed\n",match_percent, 100.0 - match_percent);
 
     // Step 3: Fallback global search for failed riders
     int failed_count = NUM_RIDERS - matched;
     if (failed_count > 0)
     {
         printf("Launching fallback global search for %d riders...\n", failed_count);
         int *h_failed = (int *)malloc(failed_count * sizeof(int));
         int idx = 0;
         for (int i = 0; i < NUM_RIDERS; ++i)
         {
             if (h_results[i] == -1)
                 h_failed[idx++] = i;
         }
         CHECK(cudaMemcpy(d_failed_indices, h_failed, failed_count * sizeof(int),cudaMemcpyHostToDevice));
         free(h_failed);
 
         int fallback_blocks = (failed_count + threads - 1) / threads;
         fallback_global_kernel<<<fallback_blocks, threads>>>(d_rx, d_ry, d_dx, d_dy, d_info, d_res, d_failed_indices, failed_count);
         CHECK(cudaDeviceSynchronize());
 
         // Copy updated results for all riders (simpler than partial update)
         CHECK(cudaMemcpy(h_results, d_res, NUM_RIDERS * sizeof(int), cudaMemcpyDeviceToHost));
     }
     cudaFree(d_rx);
 
     // Final statistics
     matched = 0;
     for (int i = 0; i < NUM_RIDERS; ++i)
     {
         if (h_results[i] != -1)
             matched++;
     }
     printf("Final: %d matched, %d failed\n", matched, NUM_RIDERS - matched);
     printf("SUCCESS: %s\n", matched <= NUM_DRIVERS ? "No double booking" : "Oversold drivers");
 
     // Cleanup
     free(h_rx);
     free(h_ry);
     free(h_dx);
     free(h_dy);
     free(h_results);
     free(h_info);
     cudaFree(d_rx);
     cudaFree(d_ry);
     cudaFree(d_dx);
     cudaFree(d_dy);
     cudaFree(d_info);
     cudaFree(d_res);
     cudaFree(d_grid_count);
     cudaFree(d_grid_driver_id);
     cudaFree(d_grid_driver_x);
     cudaFree(d_grid_driver_y);
     cudaFree(d_failed_indices);
 
     gettimeofday(&end, NULL);
     double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
     printf("Total execution time: %.4f seconds\n", elapsed);
 
     return 0;
 }