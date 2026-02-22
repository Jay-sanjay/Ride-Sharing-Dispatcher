## v3 – Radius Expansion (Unbounded)
- **Approach**: Search expands layer‑by‑layer from the rider’s cell to the entire grid. This fixes the matching issue but **rescans all previous layers at each step**.
- **Complexity**: **O(M·R³)** with `R = GRID_DIM = 100` – actually **worse than v1** for large R.
- **Matching**: 100%, but runtime skyrockets (~8000 seconds)-->Extremely Slow.

## How to Compile and Run
```bash
nvcc main_v3.cu -o ride_matcher
./ride_matcher
```