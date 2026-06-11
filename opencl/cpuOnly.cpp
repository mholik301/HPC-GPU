

#include <iostream>
#include <chrono>
#include <iostream>
#include <cfloat>
#include <iomanip>

#include <random>


#include <CL/opencl.h>
#include <CL/cl.h> 

#include "exception.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <memory>
#include <iostream>
#include <cassert>


using namespace std;



// Thread block size

#define BLOCK_SIZE 16

#define WA (8 * BLOCK_SIZE) // Matrix A width
#define HA (8 * BLOCK_SIZE) // Matrix A height
#define WB (8 * BLOCK_SIZE) // Matrix B width
#define HB WA  // Matrix B height
#define WC WB  // Matrix C width 
#define HC HA  // Matrix C height



// Standardized MAX, MIN and CLAMP
#define MAX(a, b) ((a > b) ? a : b)
#define MIN(a, b) ((a < b) ? a : b)
#define CLAMP(a, b, c) MIN(MAX(a, b), c)    // double sided clip of input a
#define TOPCLAMP(a, b) (a < b ? a:b)	    // single top side clip of input a


void shrFillArray(float* pfData, int iSize)
{
    int i;
    const float fScale = 1.0f / (float)RAND_MAX;
    for (i = 0; i < iSize; ++i)
    {
        pfData[i] = fScale * rand();
    }
}


void shrPrintArray(float* pfData, int iSize)
{
    int i;
    for (i = 0; i < iSize; ++i)
    {
        printf("%d: %.3f\n", i, pfData[i]);
    }
}


void shrPrint2DArray(float* pfData, int he, int wi)
{
    int i, j;
    std::cout << std::fixed;
    std::cout << std::setprecision(3);
    for (i = 0; i < he; ++i)
    {
        for (j = 0; j < wi; ++j)
        {
            //printf("  %.3f", i, pfData[(i*w)+j]); // prints zeroes
            cout << "   " << pfData[(i * wi) + j];
        }
        printf("\n");
    }
}

// Round Up Division function
size_t shrRoundUp(int group_size, int global_size)
{
    int r = global_size % group_size;
    if (r == 0)
    {
        return global_size;
    }
    else
    {
        return global_size + group_size - r;
    }
}


void computeGold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j) {
            double sum = 0;
            for (unsigned int k = 0; k < wA; ++k) {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }
            C[i * wB + j] = (float)sum;
        }
}




int main(int argc, char** argv)
{
    printf("Default matrix sizes are: A(%u x %u), B(%u x %u), C(%u x %u)\n",
        WA, HA, WB, HB, WC, HC);


    int inp;
    cout << "enter an matrix multiplier in range [1, 10]" << endl;
    while (true)
    {
        cin >> inp;
        if (!cin or inp > 10 or inp < 1)
        {
            cout << "Wrong Choice. Enter again " << endl;
            cin.clear();
            cin.ignore(LLONG_MAX, '\n'); //numeric_limits<streamsize>::max() replaced by <climits>'s LLONG_MAX because of name conflict with max macro
            continue;
        }
        else break;
    }

    int iSizeMultiple = CLAMP(inp, 1, 10);


    int uiWA = WA * iSizeMultiple;
    int uiHA = HA * iSizeMultiple;
    int uiWB = WB * iSizeMultiple;
    int uiHB = HB * iSizeMultiple;
    int uiWC = WC * iSizeMultiple;
    int uiHC = HC * iSizeMultiple;


    printf("New matrix sizes are: A(%u x %u), B(%u x %u), C(%u x %u)\n",
        uiWA, uiHA, uiWB, uiHB, uiWC, uiHC);
    cout << endl;


    char prnt = 'n';
    cout << "Print values? (y/n)" << endl;
    cin >> prnt;
    cout << endl;

    


    // allocate host memory for matrices A and B
    unsigned int size_A = uiWA * uiHA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A_data = (float*)malloc(mem_size_A);
    unsigned int size_B = uiWB * uiHB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B_data = (float*)malloc(mem_size_B);

    // initialize host memory
    srand(2006);
    shrFillArray(h_A_data, size_A);
    shrFillArray(h_B_data, size_B);

    if (prnt == 'y'){
        cout << "Input:" << endl;
        printf("matrix A(%u x %u)\n", uiHA, uiWA);
        shrPrint2DArray(h_A_data, uiHA, uiWA);
        printf("matrix B(%u x %u)\n", uiHB, uiWB);
        shrPrint2DArray(h_B_data, uiHB, uiWB);
        cout << endl;
    }

    // allocate host memory for result
    unsigned int size_C = uiWC * uiHC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* h_C = (float*)malloc(mem_size_C);


    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    cout << "Starting operation\n" << endl;
    auto t1 = high_resolution_clock::now();
    computeGold(h_C, h_A_data, h_B_data, uiHA, uiWA, uiWB);
    auto t2 = high_resolution_clock::now();

    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    duration<double, std::milli> ms_double = t2 - t1;

    double dSeconds = ms_double.count() / 1e3;
    double dNumOps = 2.0 * (double)uiWA * (double)uiHA * (double)uiWB;
    double gflops = 1.0e-9 * dNumOps / dSeconds;
    printf("CPU MatrixMul, Throughput = %.4f GFlops/s, Time = %.5f s, Size (Op Num) = %.0f\n", gflops, dSeconds, dNumOps);

    if (prnt == 'y') {
        cout << "Result (matrix C):" << endl;
        shrPrint2DArray(h_C, uiHC, uiWC);
    }

}