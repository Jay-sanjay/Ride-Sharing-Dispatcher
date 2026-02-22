## v2 – Spatial Grid (Single Cell)
- **Approach**: Drivers are binned into a grid; each rider searches **only its own cell**. Drivers exceeding `MAX_CAPACITY` are dropped.
- **Complexity**: **O(M·MAX_CAPACITY)** – much faster, but many riders fail if their cell is empty or full.
- **Matching**: Only **~51%** of riders matched (due to capacity limits and no neighbor search).
- **Runtime**: ~1.95 seconds – huge speedup, but poor accuracy.

## How to Compile and Run
```bash
nvcc main_v2.cu -o ride_matcher
./ride_matcher
```