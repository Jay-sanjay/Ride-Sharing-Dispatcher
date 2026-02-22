## v1 – Naive Global Search
- **Approach**: Each rider thread scans **all drivers** to find the nearest free driver. If booking fails, the scan repeats.
- **Complexity**: **O(M·N)** – catastrophic for 10M agents.
- **Matching**: Always finds a driver if one exists (but very slow).
- **Runtime**: ~224 seconds for 10M riders/drivers.

## How to Compile and Run
```bash
nvcc main_v1.cu -o ride_matcher
./ride_matcher
```