////////////////////////////////////////////////////////////////////////////////

#include <fcntl.h>
#define _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/cl.h>
#include <stdbool.h>




#include <time.h>

//src https://www.es.ele.tue.nl/~mwijtvliet/5KK73/?page=mmopencl


////////////////////////////////////////////////////////////////////////////////
#define WA 128
#define HA 128
#define WB 128

#define HB WA
#define WC WB
#define HC HA
////////////////////////////////////////////////////////////////////////////////


#define MAX_IN_MUL 16

// Standardized MAX, MIN and CLAMP
#define MAX(a, b) ((a > b) ? a : b)
#define MIN(a, b) ((a < b) ? a : b)
#define CLAMP(a, b, c) MIN(MAX(a, b), c)    // double sided clip of input a
#define TOPCLAMP(a, b) (a < b ? a:b)	    // single top side clip of input a


void shrPrint2DArray(float* pfData, int he, int wi)
{
    int i, j;
    for (i = 0; i < he; ++i)
    {
        for (j = 0; j < wi; ++j)
        {
            printf("  %3.3f", pfData[(i*wi)+j]); // prints zeroes
        }
        printf("\n");
    }
}


// Allocates a matrix with random float entries.
void randomMemInit(float* data, int size)
{
    int i;

    for (i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}


void computeCPU(float* C, const float* A, const float* B, unsigned int uiHA, unsigned int uiWA, unsigned int uiWB)
{
    clock_t start, end;
    double cpu_time_used;


    printf("Starting CPU compute\n");
    start = clock();
    for (unsigned int i = 0; i < uiHA; ++i) {
        for (unsigned int j = 0; j < uiWB; ++j) {
            double sum = 0;
            for (unsigned int k = 0; k < uiWA; ++k) {
                double a = A[i * uiWA + k];
                double b = B[k * uiWB + j];
                sum += a * b;
            }
            C[i * uiWB + j] = (float)sum;
        }
    }

    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Host computation took %f seconds to execute \n", cpu_time_used);

    double dNumOps = 2.0 * (double)uiWA * (double)uiHA * (double)uiWB;
    double gflops = 1.0e-9 * dNumOps / cpu_time_used;
    printf("CPU MatrixMul, Throughput = %.4f GFlops/s, Time = %.5f s, Size (Op Num) = %.0f\n", gflops, cpu_time_used, dNumOps);
}


int computeGPU(float* h_C, const float* h_A, const float* h_B, unsigned int uiHA, unsigned int uiWA, unsigned int uiWB) {
    clock_t start, end;
    double cpu_time_used;

    int err;                            // error code returned from api calls

    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel

     // OpenCL device memory for matrices
    cl_mem d_A;
    cl_mem d_B;
    cl_mem d_C;

    // set seed for rand()
    srand(2014);

    unsigned int size_A = uiWA * uiHA;
    unsigned int mem_size_A = sizeof(float) * size_A;

    unsigned int size_B = uiWB * uiWA;
    unsigned int mem_size_B = sizeof(float) * size_B;

    unsigned int size_C = uiWB * uiHA;
    unsigned int mem_size_C = sizeof(float) * size_C;

    printf("\n");

    cl_uint dev_cnt = 0;
    clGetPlatformIDs(0, 0, &dev_cnt);

    cl_platform_id platform_ids[100];
    clGetPlatformIDs(dev_cnt, platform_ids, NULL);

    // Connect to a compute device
    int gpu = 1;
    err = clGetDeviceIDs(platform_ids[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }

    // Create a compute context 
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    // Create a command commands
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    // Create the compute program from the source file
    char* KernelSource;
    long lFileSize;

    //lFileSize = LoadOpenCLKernel("matrixMul.cl", &KernelSource);  //doesn't work because Nvidia's matrixMul does more than just "be a text file"
    lFileSize = LoadOpenCLKernel("matrixmul_kernel.cl", &KernelSource);
    if (lFileSize < 0L) {
        perror("File read failed");
        return 1;
    }

    program = clCreateProgramWithSource(context, 1, (const char**)&KernelSource, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    // Create the compute kernel in the program we wish to run
    //
    kernel = clCreateKernel(program, "matrixMul", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    printf("Starting Host->Device transfer\n");
    start = clock();
    // Create the input and output arrays in device memory for our calculation
    d_C = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_A, NULL, &err);
    d_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_A, h_A, &err); // CL_MEM_COPY_HOST_PTR simply copies the values at a time of creation of the buffer.
    d_B = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_B, h_B, &err);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Host -> Device data transfer took %f seconds to execute \n", cpu_time_used);
    double dNumOps = 2.0 * (double)uiWA * (double)uiHA * (double)uiWB;
    double trans =  ((dNumOps * 4) / 1e9)/cpu_time_used; //MB/s
    printf("Host -> Device data transfer throughput = %.4f MB/s\n\n", trans);


    if (!d_A || !d_B || !d_C)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }

    //Launch OpenCL kernel
    size_t localWorkSize[2], globalWorkSize[2];

    int wA = WA;
    int wC = WC;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&d_C);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&d_A);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&d_B);
    err |= clSetKernelArg(kernel, 3, sizeof(int), (void*)&wA);
    err |= clSetKernelArg(kernel, 4, sizeof(int), (void*)&wC);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    localWorkSize[0] = 16;
    localWorkSize[1] = 16;
    globalWorkSize[0] = 1024;
    globalWorkSize[1] = 1024;

    printf("Starting GPU compute\n");
    start = clock();
    err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    end = clock();

    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Device computation took %f seconds to execute \n", cpu_time_used);

    
    double gflops = 1.0e-9 * dNumOps / cpu_time_used;
    printf("GPU MatrixMul, Throughput = %.4f GFlops/s, Time = %.5f s, Size (Op Num) = %.0f\n\n", gflops, cpu_time_used, dNumOps);


    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to execute kernel! %d\n", err);
        exit(1);
    }

    printf("Starting Device->Host transfer\n");
    start = clock();
    //Retrieve result from device
    err = clEnqueueReadBuffer(commands, d_C, CL_TRUE, 0, mem_size_C, h_C, 0, NULL, NULL);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Device -> Host data transfer took %f seconds to execute \n", cpu_time_used);
	dNumOps = (double)uiWA * (double)uiHA * (double)uiWB;
    trans = ((dNumOps * 4) / 1e9) / cpu_time_used; //MB/s
    printf("Device -> Host data transfer throughput = %.4f MB/s\n\n", trans);


    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    //print out the results
 /*
    printf("\n\nMatrix C (Results)\n");
    int i;
    for(i = 0; i < size_C; i++)
    {
       printf("%f ", h_C[i]);
       if(((i + 1) % WC) == 0)
       printf("\n");
    }
    printf("\n");
 */

    //Shutdown and cleanup
    free(h_A);
    free(h_B);
    free(h_C);

    clReleaseMemObject(d_A);
    clReleaseMemObject(d_C);
    clReleaseMemObject(d_B);

    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return 0;
}


long LoadOpenCLKernel(char const* path, char** buf)
{
    FILE* fp;
    size_t fsz;
    long   off_end;
    int    rc;

    /* Open the file */
    fopen_s(&fp, path, "r");
    if (NULL == fp) {
        return -1L;
    }

    /* Seek to the end of the file */
    rc = fseek(fp, 0L, SEEK_END);
    if (0 != rc) {
        return -1L;
    }

    /* Byte offset to the end of the file (size) */
    if (0 > (off_end = ftell(fp))) {
        return -1L;
    }
    fsz = (size_t)off_end;

    /* Allocate a buffer to hold the whole file */
    *buf = (char*)malloc(fsz + 1);
    if (NULL == *buf) {
        return -1L;
    }

    /* Rewind file pointer to start of file */
    rewind(fp);

    /* Slurp file into buffer */
    if (fsz != fread(*buf, 1, fsz, fp)) {
        free(*buf);
        return -1L;
    }

    /* Close the file */
    if (EOF == fclose(fp)) {
        free(*buf);
        return -1L;
    }


    /* Make sure the buffer is NUL-terminated, just in case */
    (*buf)[fsz] = '\0';

    /* Return the file size */
    return (long)fsz;
}


int main(int argc, char** argv)
{
    printf("Default matrix sizes are: A(%u x %u), B(%u x %u), C(%u x %u)\n",
        WA, HA, WB, HB, WC, HC);


    int inp;
    printf("enter an matrix multiplier in range [1, %d]\n", MAX_IN_MUL);
    while (true)
    {
        scanf_s("%d", &inp);
        if (inp > MAX_IN_MUL || inp < 1)
        {
            printf("Wrong Choice.Enter again \n");
        }
        else break;
    }

    int iSizeMultiple = CLAMP(inp, 1, MAX_IN_MUL);


    int uiWA = WA * iSizeMultiple;
    int uiHA = HA * iSizeMultiple;
    int uiWB = WB * iSizeMultiple;
    int uiHB = HB * iSizeMultiple;
    int uiWC = WC * iSizeMultiple;
    int uiHC = HC * iSizeMultiple;


    printf("New matrix sizes are: A(%u x %u), B(%u x %u), C(%u x %u)\n\n",
        uiWA, uiHA, uiWB, uiHB, uiWC, uiHC);



    //Allocate host memory for matrices A and B
    unsigned int size_A = uiWA * uiHA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*)malloc(mem_size_A);

    unsigned int size_B = uiWB * uiHB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*)malloc(mem_size_B);

    //Initialize host memory
    randomMemInit(h_A, size_A);
    randomMemInit(h_B, size_B);


    //Allocate host memory for the result C
    unsigned int size_C = uiWC * uiHC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* h_C = (float*)malloc(mem_size_C);


    computeCPU(h_C, h_A, h_B, uiHA, uiWA, uiWB);
    computeGPU(h_C, h_A, h_B, uiHA, uiWA, uiWB);

    


    return 0;
}
