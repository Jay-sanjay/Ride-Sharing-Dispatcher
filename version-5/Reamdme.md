## v5 – Coordinates Stored in Grid
- **Improvement**: Instead of storing only driver IDs in the grid, now store **ID, x, and y** together in three parallel arrays.  
  Distance calculations read directly from these arrays, **eliminating two random global memory accesses** per candidate.
- **Complexity**: Same as v4, but with much better memory coalescing.
- **Matching**: 100%.
- **Runtime**: **~15 seconds** – **15× faster than v1** and a 4× improvement over v4.

## How to Compile and Run
```bash
nvcc main_v5.cu -o ride_matcher
./ride_matcher
```