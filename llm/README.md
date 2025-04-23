
# 🔧 MLIR Optimization and ASIC Flow

## 🧩 Part 1: MLIR Preprocessing and Loop Transformations

This stage prepares the MLIR for further processing and explores loop-level optimizations like **tiling**, **permutation**, or **unrolling**. Depending on the flags provided, different passes are selectively applied.

### 🔹 Execution

```bash
python3 ./llm/main.py --read_mlir=./models/AlexNet_10.mlir --conv2d --tile --select_layer=3 --part1 --trial=1
```

### 🔹 What it Does
1. **Reads** the input MLIR model via `--read_mlir`.
2. Applies **loop-level transformations**:
   - `--tile`
   - `--permute`
   - `--unroll`
3. Targets specific layers using flags:
   - `--select_layer`
   - `--start_layer` / `--end_layer` (optional)
4. Generates intermediate MLIR files:
   - `output/04a<config>.mlir` → After initial SODA processing
   - `output/04b<config>.mlir` → After basic MLIR transformations

### 🔹 Important: LLM Optimization Step

- **After Part 1**, the file `output/04b<config>.mlir` is generated.
- This file must be **passed through an LLM** for further optimization (semantic transformation, canonicalization, etc.).
- Save the **optimized output back into the same file**: `output/04b<config>.mlir`.

---

## ⚙️ Part 2: MLIR to ASIC Synthesis Flow

After LLM optimization, Part 2 picks up from the modified MLIR and carries out the rest of the flow—transforming MLIR to LLVM IR and finally generating an ASIC implementation.

### 🔹 Execution

```bash
python3 ./llm/main.py --read_mlir=./models/AlexNet_10.mlir --conv2d --tile --select_layer=3 --part2 --trial=1
```

### 🔹 What it Does
1. Starts from the **LLM-optimized** `output/04b<config>.mlir`.
2. Applies final MLIR passes and translates to LLVM IR.
3. Generates:
   - `output/05<config>.ll` → LLVM IR
   - Runs **Bambu** for high-level synthesis (`output/bambu-<config>.log`)
   - Runs **OpenROAD** for backend layout (`output/openroad-<config>.log`)

---

## ✅ Summary

| Step       | Description                             | Output Files                             |
|------------|-----------------------------------------|------------------------------------------|
| **Part 1** | Preprocess + Loop Optimization          | `04a*.mlir`, `04b*.mlir`   |
| **LLM**    | Optimize `04b*.mlir` using an LLM       | Overwrite `04b*.mlir` with optimized IR |
| **Part 2** | MLIR to LLVM IR to ASIC Implementation  | `05*.ll`, `bambu-*.log`, `openroad-*.log`|

---
