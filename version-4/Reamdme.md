## v4 – Bounded Radius + Fallback + Perimeter Scanning
- **Key fixes**:
  - **Bounded radius** (`MAX_RADIUS = 5`) – caps search to at most 121 cells.
  - **Perimeter‑only scanning** – at each radius, only new cells on the square’s edge are examined (no rescanning).
  - **Fallback global search** – the tiny fraction of riders still unmatched (e.g., 0.4%) finish with a full scan.
  - **Booking retry reset** – after a failed atomic booking, the thread resets its best candidate and continues.
- **Complexity**: **O(M·C)** where C ≈ 124 000 driver checks – independent of N.
- **Matching**: 100%.
- **Runtime**: ~59 seconds – 3.8× faster than v1.


## How to Compile and Run
```bash
nvcc main_v4.cu -o ride_matcher
./ride_matcher
```