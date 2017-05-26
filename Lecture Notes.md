# INTRO TO PARALLEL PROGRAMMING

## Lesson 1 - The GPU Programming Model

### Latency vs Throughput
**CPUs focus on improving latency, whereas GPUs focus on improving throughput:**
![Latency vs Throughput](Images/latency-vs-throughput.png)

### CUDA Program Diagram
![CUDA Program Diagram](Images/cuda-program-diagram.png)

![A Typical GPU Program](Images/typical-gpu-program.png)

![The Big Idea](Images/big-idea.png)

### Example: Squaring the Elements of an Array
![CPU](Images/square-array-on-cpu.png)

![GPU](Images/square-array-on-gpu.png)

![Blocks and Threads](Images/blocks-and-threads.png)

**Our data might be 2 dimensional, such as an image.** <br>
**In this case, we might want to arrange our blocks in 2 dimensions instead of 1:**
![Multiple Dimensions](Images/multiple-dimensions.png)

**We can choose up to 3 dimensional blocks and threads:**
![Kernel Dimensions](Images/kernel-dimensions.png)

**Each thread/blocks knows its index, as well as the size of the block/grid that it resides in:**
![Dimension Variables](Images/dimension-variables.png)

## Lesson 2 - GPU Hardware and Parallel Communication Patterns

![Map Pattern](Images/map-pattern.png)

![Gather Pattern](Images/gather-pattern.png)

![Scatter Pattern](Images/scatter-pattern.png)

![Stencil Pattern](Images/stencil-pattern.png)

![Transpose Pattern](Images/transpose-pattern.png)

![Summary of Patterns](Images/patterns-summary.png)

### GPU Hardware
**Some GPUs have more SMs than the others.**
![Streaming Multiprocessors](Images/streaming-multiprocessors.png)

**All the SMs run in parallel and independently.**
![Allocating Blocks to SMs](Images/allocating-blocks.png)

#### Allocating Blocks to SMs

  * **An SM may run more than one block.**
  * **A block may _not_ run on more than one SM.**

**CUDA guarantees that:**
 - **all threads in a block run on the same SM at the same time.**
 - **all blocks in a kernel finish before any blocks from the next kernel run.**

**CUDA _does not_ guarantee that:**
 - **a block will run at the same time as another block.**
 - **a block will run after another block.**
 - **a block will run on a specific SM.**

| Advantages | Consequences                         |
| :---------:| :-----------------------------------:|
| flexibility| no assumptions on block-SM allocation|
| scalability| no communication between blocks      |

#### GPU Memory Model

![GPU Memory Model](Images/gpu-memory-model.png)
**The shared memory is shared among the threads of a block.**

#### Synchronization

![Barrier](Images/synchronization-barrier.png)

**Example: Suppose we want to shift each element of an array to the left.**
![Barrier Example](Images/barrier-example-1.png)
![Barrier Example](Images/barrier-example-2.png)

**In the code above;**
  - **The first barrier makes sure that the array is initialized correctly.**
  - **The second array makes sure that the value on the right is read correctly by each thread.**
  - **The third barrier makes sure that the array is completely processed before anybody attempts to access it.**

![Shared and Global Thread Synchronization](Images/shared-and-global-sync.png)
**Global synchronization is achieved by using kernels sequentially. <br>
In-block synchronization is achieved by using barriers in a block.**

#### Writing Efficient Programs
![Arithmetic Intensity](Images/arithmetic-intensity.png)

![Minimize Time Spent on Memory](Images/memory-time.png)

![Using Local Memory](Images/using-local-memory.png)

**Using shared memory:**
![Using Shared Memory](Images/using-shared-memory.png)

![Using Global Memory](Images/using-global-memory.png)

![Coalesced Memory Access](Images/coalesced-memory.png)

![Thread Divergence 1](Images/thread-divergence.png)

![Thread Divergence 2](Images/thread-divergence-2.png)

#### Atomic Memory Operations
![Atomic Memory Operations](Images/atomic-memory-operations.png)

**The second function gives right results, but the first one does not:**
![Atomic Add](Images/atomic-add.png)

![Limitations of Atomics](Images/limitations-of-atomics.png)

## Lesson 3 - Fundamental GPU Algorithms
![Step & Work Complexity](Images/step-and-work-complexity.png)
### Reduce Algorithm
![Reduce](Images/reduce.png)
![Reduce Example](Images/reduce-example.png)

#### Serial Reduction
![Serial Reduction](Images/serial-reduction.png)
#### Parallel Reduction
![Parallel Reduction](Images/parallel-reduction.png)

**Parallel Reduction Using Global Memory:**
![Global Memory](Images/parallel-reduction-global.png)

**Parallel Reduction Using Shared Memory:**
![Shared Memory](Images/parallel-reduction-shared.png)
![Shared Memory 2](Images/parallel-reduction-shared-2.png)

**Execute Parallel Reduction Kernels:**
![Execute Kernels](Images/parallel-reduction-execute.png)

### Scan Algorithm
![Scan](Images/scan-algorithm.png)

![Scan Example](Images/scan-example.png)

**Scan Using N Reductions:**
![Reduction Scan](Images/naive-scan.png)
![Scan Complexity](Images/scan-complexity.png)

**Hillis-Steele Scan:**
![Scan Complexity](Images/hillis-steele-scan.png)

**Blelloch Scan:**
![Bleloch Scan](Images/blelloch-scan.png)

**Blelloch Max-Scan Example:**
![Bleloch Scan Example](Images/maxscan-example.png)

**Which Algorithm to Choose:**
![Algorithm Comparison](Images/choosing-the-right-algorithm.png)

### Histogram Algorithm
![Histogram](Images/histogram-algorithm.png)

## Lesson 4 - Fundamental GPU Algorithms cont'd

### Compact
![Compact](Images/compact-primitive.png)

![Compact Algorithm](Images/compact-algorithm.png)

![Compact Steps](Images/compact-steps.png)

#### Multiple Outputs Per Input Element:
![Dynamic Compact](Images/dynamic-compact.png)

### Sparse Matrix Multiplication
#### Segmented Scan:
![Segmented Scan](Images/segmented-scan.png)

#### Sparse Matrices:
![Sparse Matrices](Images/sparse-matrices.png)

#### Multiplication:
![Sparse Matrix Multiplication](Images/sparse-matrix-multiplication.png)
**Note: Exclusive segmented scan can be replaced with segmented reduce which is more efficient.**

### Sort
![Brick Sort](Images/brick-sort.png)

![Radix Sort](Images/radix-sort.png)

## Lesson 5 - Optimizing GPU Programs

![APOD](Images/optimization-apod.png)

![Scaling](Images/scaling.png)

### Memory Bandwidth
![CUDA Device Query](Images/device-query.png)

![Theoretical Peak Bandwidth](Images/peak-bandwidth.png)

#### NVVP Tool
![NVVP Profiler Tool](Images/nvvp-profiler.png)

![NVVP Screen](Images/nvvp-screen.png)

### Math Optimizations
![Math Optimizations](Images/math-optimizations.png)

### Host-GPU Interactions

#### Pinned Host Memory
![Pinned Host Memory](Images/cuda-memcpy-async.png)

#### Streams
![Advantages of Streams](Images/stream-advantages.png)

![Streams](Images/streams.png)

![Using Streams 1](Images/stream-usage.png)
![Using Streams 2](Images/stream-usage-2.png)

**Without Streams:**
![Without Streams](Images/without-streams.png)

**With Streams:**
![With Streams](Images/with-streams.png)
