









A note about the special attributes in the kernel signature:
Specifying where should a GPU thread keep its data is done through the kernel function signature.
looks like a regular C function, with a special __kernel attribute, and its arguments have additional attributes describing in which GPU memory they live.
On the CPU we have automatic vs static variables, while the GPU has 
	__private	Per-SMSP (seen between warp threads), managed fully by the GPU
	__local		Per-SM (seen between all blocks in a SM), managed fully by the GPU
	__global	Device-visible buffers, seen by all SMs, allocated (clCreateBuffer()) and initialized (clEnqueueWriteBuffer()) by the host 
	__constant	Read-only __global


// matrixmul_kernel.cl

__kernel void matrixMul(		// host side arguments:
    __global const float* A,	// d_A  (input matrix A)
    __global const float* B,	// d_B  (input matrix B)
    __global float* C,			// d_C  (output matrix C)
    const int wA,				// wA   (width of A)
    const int wC)				// wC   (width of C)
{
    // Global coordinates of the current work-item. Equivalent to the CPU keeping a 'int i' and 'int j' on the stack
	// Executed on the GPU. 
    const int col = get_global_id(0);
    const int row = get_global_id(1);

    // Width of B can be derived from width of C.
    const int wB = wC;

    float sum = 0.0f;

    // Compute one element C[row][col].
    for (int k = 0; k < wA; ++k)
    {
        sum += A[row * wA + k] *
               B[k * wB + col];
    }

    // Store result.
    C[row * wC + col] = sum;
}







(in clGetDeviceIDs function)


// ------------------------------------------------------------------
// 1. Discover available OpenCL platforms
// ------------------------------------------------------------------

cl_uint platformCount = 0;

// Query how many OpenCL platforms are available.
clGetPlatformIDs(
    0,
    nullptr,
    &platformCount
);

// Allocate storage for all platform IDs.
cl_platform_id* platformIds =
    new cl_platform_id[platformCount];

// Retrieve the platform IDs.
clGetPlatformIDs(
    platformCount,
    platformIds,
    nullptr
);

// Select the first platform.
cl_platform_id platform = platformIds[0];

// ------------------------------------------------------------------
// 2. Discover devices on the selected platform
// ------------------------------------------------------------------

cl_uint deviceCount = 0;

// Query number of devices available on the platform.
clGetDeviceIDs(
    platform,
    CL_DEVICE_TYPE_GPU,
    0,
    nullptr,
    &deviceCount
);

// Retrieve device IDs.
cl_device_id* deviceIds =
    new cl_device_id[deviceCount];

clGetDeviceIDs(
    platform,
    CL_DEVICE_TYPE_GPU,
    deviceCount,
    deviceIds,
    nullptr
);

/*
Note: To make OpenCL run the kernel on the CPU/GPU you can change the constant CL_DEVICE_TYPE_DEFAULT to CL_DEVICE_TYPE_GPU or CL_DEVICE_TYPE_CPU
*/

// Select the first GPU device.
cl_device_id device_id = deviceIds[0];



// ------------------------------------------------------------------
// 3. Create OpenCL context and command queue
// ------------------------------------------------------------------

// opencl maintains a queue of work to be done

// Context represents the OpenCL execution environment.
cl_context context =
    clCreateContext(
        nullptr,
        1,
        &device_id,
        nullptr,
        nullptr,
        nullptr
    );

// Command queue is used to submit memory transfers and kernels.
cl_command_queue commands =
    clCreateCommandQueue(
        context,
        device_id,
        0,
        nullptr
    );

// ------------------------------------------------------------------
// 4. Load and compile kernel source
// ------------------------------------------------------------------

/*
Note: 
OpenCL uses so-called runtime compilation because the program/programmer don't have to necessarily know the details of the device on which the software will be executed.
Unlike general-purpose processors, GPUs and similar accelerators are not built for the general use cases. Rather, code is fine-tuned for specific contexts.
This means your kernel is not compiled along with the C++ source code, rather during execution, when the clBuildProgram() function is called.

This also means your open source code is hardware agnostic.
The kernel compiler, provided by the runtime, gets the information it needs to "just-in-time" compile a fine-tuned kernel (specific to the GPU you have) at runtime.
After the kernel compilation is done, the produced code is dynamically linked to your source program, compiled beforehand.
*/

// Load kernel source code from file.
LoadOpenCLKernel(
    "matrixmul_kernel.cl",
    &KernelSource
);

// Create program object from source code.
cl_program program =
    clCreateProgramWithSource(
        context,
        1,
        &KernelSource,
        nullptr,
        nullptr
    );

// Compile/build the OpenCL program.
clBuildProgram(
    program,
    1,
    &device_id,
    nullptr,
    nullptr,
    nullptr
);

// Create kernel object.
// Assumes kernel function is named "matrixMul".
cl_kernel kernel =
    clCreateKernel(
        program,
        "matrixMul",
        nullptr
    );

// ------------------------------------------------------------------
// 5. Allocate device buffers
// ------------------------------------------------------------------

// allocate buffers in GPU VRAM, where initial values will be written to, and the output produced to and read from by the host
// note: depending on the actual hardware, cl_mem might refer to VRAM, system RAM, unified memory, pinned memory, some device-specific memory pool...
// e.g. discrete GPUs will allocate straight to VRAM, while integrated ones might use RAM

// Device buffer containing matrix A.
cl_mem d_A =
    clCreateBuffer(
        context,
        CL_MEM_READ_ONLY,
        mem_size_A,
        nullptr,
        nullptr
    );

// Device buffer containing matrix B.
cl_mem d_B =
    clCreateBuffer(
        context,
        CL_MEM_READ_ONLY,
        mem_size_B,
        nullptr,
        nullptr
    );

// Device buffer containing result matrix C.
cl_mem d_C =
    clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        mem_size_C,
        nullptr,
        nullptr
    );

// ------------------------------------------------------------------
// 6. Copy input matrices from host to device
// ------------------------------------------------------------------

// Host RAM  --->  GPU VRAM

// Upload matrix A to GPU memory.
clEnqueueWriteBuffer(
    commands,
    d_A,
    CL_TRUE,
    0,
    mem_size_A,
    h_A,
    0,
    nullptr,
    nullptr
);

// Upload matrix B to GPU memory.
clEnqueueWriteBuffer(
    commands,
    d_B,
    CL_TRUE,
    0,
    mem_size_B,
    h_B,
    0,
    nullptr,
    nullptr
);

// ------------------------------------------------------------------
// 7. Set kernel arguments
// ------------------------------------------------------------------

// bind arguments in kernel's template to the device memory we allocated and initialized


// Input matrix A.
clSetKernelArg(
    kernel,
    0,
    sizeof(cl_mem),
    &d_A
);

// Input matrix B.
clSetKernelArg(
    kernel,
    1,
    sizeof(cl_mem),
    &d_B
);

// Output matrix C.
clSetKernelArg(
    kernel,
    2,
    sizeof(cl_mem),
    &d_C
);

// Width of matrix A.
clSetKernelArg(
    kernel,
    3,
    sizeof(int),
    &wA
);

// Width of matrix C.
clSetKernelArg(
    kernel,
    4,
    sizeof(int),
    &wC
);

// ------------------------------------------------------------------
// 8. Configure execution dimensions
// ------------------------------------------------------------------

// One work-group contains 16×16 work-items.
size_t localWorkSize[2] = { 16, 16 };

// Total number of work-items.
// One work-item computes one output element.
size_t globalWorkSize[2] = {
    1024,
    1024
};


/*
The number of times a kernel will execute depends on the size of the global problem, and the number of local groups.

Conceptually, each work-item computes exactly one element of matrix C:
// A (HA × WA) × B (WA × WB) = C (HA × WB)
row = get_global_id(1), col = get_global_id(0)

C[row][col] =
    A[row][0] * B[0][col] +
    A[row][1] * B[1][col] +
    ...
    A[row][WA-1] * B[WA-1][col]


With launch configuration:
size_t globalWorkSize[2] = { 1024, 1024 };
size_t localWorkSize[2]  = { 16, 16 };

OpenCL creates:
1024 × 1024 = 1,048,576 independant work-items (dot products), each with 1024 sequential multiply-add steps

A GPU with 64 Compute Units, each with 64 Processing Elements (e.g. Nvidia gpu with 32 SMs, each with 2 SMSPs, each with 32 INT32 and 32 multi purpose units), has 4096 "arithmetic lanes" in total
Meaning each lane will do around 256 tasks from the work queue
*/



// ------------------------------------------------------------------
// 9. Launch kernel
// ------------------------------------------------------------------



// Execute matrix multiplication kernel on GPU.
clEnqueueNDRangeKernel(
    commands,			// kernel queue
    kernel,				// kernel
    2,                  // problem dimension
    nullptr,            // no global offset
    globalWorkSize,
    localWorkSize,
    0,
    nullptr,
    nullptr
);

// send the queue to the device
clFlush(commands)

// Wait until all queued commands complete.
clFinish(commands);

// ------------------------------------------------------------------
// 10. Copy result matrix back to host
// ------------------------------------------------------------------

// Download matrix C from GPU memory to CPU memory.
clEnqueueReadBuffer(
    commands,
    d_C,
    CL_TRUE,
    0,
    mem_size_C,
    h_C,
    0,
    nullptr,
    nullptr
);

// ------------------------------------------------------------------
// 11. Performance calculation
// ------------------------------------------------------------------

// Approximate number of multiply-add operations.
double dNumOps =
    static_cast<double>(uiWA) *
    static_cast<double>(uiHA) *
    static_cast<double>(uiWB);

// Effective throughput estimate (GB/s).
double trans =
    ((dNumOps * sizeof(float)) / 1e9) /
    cpu_time_used;

// ------------------------------------------------------------------
// 12. Cleanup
// ------------------------------------------------------------------

// Release GPU buffers.
clReleaseMemObject(d_A);
clReleaseMemObject(d_B);
clReleaseMemObject(d_C);

// Release kernel and program objects.
clReleaseKernel(kernel);
clReleaseProgram(program);

// Release command queue and context.
clReleaseCommandQueue(commands);
clReleaseContext(context);

// Release host-side platform/device arrays.
delete[] platformIds;
delete[] deviceIds;

// Release host result memory if dynamically allocated.
free(h_C);