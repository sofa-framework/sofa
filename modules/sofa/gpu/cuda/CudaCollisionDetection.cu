#include "CudaCommon.h"
#include "CudaMath.h"
#include <stdio.h>
#include "cuda.h"

#if defined(__cplusplus) && CUDA_VERSION != 2000
namespace sofa
{
namespace gpu
{
namespace cuda
{
#endif

extern "C"
{
    void CudaCollisionDetection_runTests(unsigned int nbTests, unsigned int maxPoints, const void* tests, void* nresults);
}

struct /*__align__(16)*/ GPUContact
{
    int p1;
    float3 p2;
    float distance;
    float3 normal;
};

struct GPUTest
{
    GPUContact* result;
    const float3* points;
    const float* radius;
    const float* grid;
    matrix3 rotation;
    float3 translation;
    float margin;
    int nbPoints;
    int gridnx, gridny, gridnz;
    float3 gridbbmin, gridbbmax;
    float3 gridp0, gridinvdp;
};

//////////////////////
// GPU-side methods //
//////////////////////

__shared__ GPUTest curTest;

__global__ void CudaCollisionDetection_runTests_kernel(const GPUTest* tests, int* nresults)
{
    if (threadIdx.x == 0)
        curTest = tests[blockIdx.x];


    __syncthreads();

    //! Dynamically allocated shared memory to compact results
    extern  __shared__  int scan[];

    float3 p;
    float distance;
    float3 grad = make_float3(0,0,0);
    //float3 normal;
    int n = 0;
    if (threadIdx.x < curTest.nbPoints)
    {
        p = curTest.points[threadIdx.x];
        p = curTest.rotation * p;
        p += curTest.translation;

        float3 coefs = mul(p-curTest.gridp0, curTest.gridinvdp);
        int x = __float2int_rd(coefs.x);
        int y = __float2int_rd(coefs.y);
        int z = __float2int_rd(coefs.z);
        if ((unsigned)x < curTest.gridnx-1
            && (unsigned)y < curTest.gridny-1
            && (unsigned)z < curTest.gridnz-1)
        {
            int nx = curTest.gridnx;
            int nxny = nx*curTest.gridny;
            coefs.x -= __int2float_rd(x);
            coefs.y -= __int2float_rd(y);
            coefs.z -= __int2float_rd(z);
            const float* gval = curTest.grid + (x+nx*y+nxny*z);
            float d000      = gval[0        ];
            float d100_d000 = gval[1        ] - d000;
            float d010      = gval[  nx     ];
            float d110_d010 = gval[1+nx     ] - d010;
            float d001      = gval[     nxny];
            float d101_d001 = gval[1   +nxny] - d001;
            float d011      = gval[  nx+nxny];
            float d111_d011 = gval[1+nx+nxny] - d011;
            float dx00      = d000 + (d100_d000)*coefs.x;
            float dx10_dx00 = d010 + (d110_d010)*coefs.x - dx00;
            float dx01      = d001 + (d101_d001)*coefs.x;
            float dx11_dx01 = d011 + (d111_d011)*coefs.x - dx01;
            float dy0       = dx00 + (dx10_dx00)*coefs.y;
            float dy1_dy0   = dx01 + (dx11_dx01)*coefs.y - dy0;
            distance = dy0 + (dy1_dy0)*coefs.z;
            float r = 0;
            if (curTest.radius)
            {
                r = curTest.radius[threadIdx.x];
            }
            if (distance < curTest.margin+r)
            {
                n = 1;
                grad.z = dy1_dy0;
                grad.y = (dx10_dx00) + ((dx11_dx01)-(dx10_dx00))*coefs.z;
                dy0     = d100_d000 + (d110_d010 - d100_d000)*coefs.y;
                dy1_dy0 = d101_d001 + (d111_d011 - d101_d001)*coefs.y - dy0;
                grad.x = dy0 + (dy1_dy0)*coefs.z;
                grad *= invnorm(grad);
                //normal = grad;
                p -= grad*distance;
                //distance -= r;
                distance = r;
                //grad = make_float3(0,1,1);
            }
        }
    }

    scan[threadIdx.x] = n;

    for (int i=1; i<curTest.nbPoints; i<<=1)
    {
        __syncthreads();
        if (threadIdx.x>=i)
            scan[threadIdx.x] = scan[threadIdx.x] + scan[threadIdx.x - i];
    }

    if (n)
    {
        GPUContact c;
        c.p1 = threadIdx.x;
        c.p2 = p;
        c.distance = distance;
        c.normal = -grad;
        //c.normal = normal; //make_float3(-grad.x,-grad.y,-grad.z); //-grad;
        curTest.result[scan[threadIdx.x]-1] = c;
        //curTest.result[scan[threadIdx.x]-1].p1 = threadIdx.x;
        //curTest.result[scan[threadIdx.x]-1].p2 = p;
        //curTest.result[scan[threadIdx.x]-1].distance = distance;
        //curTest.result[scan[threadIdx.x]-1].normal = normal;

    }
    if (threadIdx.x == curTest.nbPoints-1)
        nresults[blockIdx.x] = scan[curTest.nbPoints-1];
}

//////////////////////
// CPU-side methods //
//////////////////////

void CudaCollisionDetection_runTests(unsigned int nbTests, unsigned int maxPoints, const void* tests, void* nresults)
{
    printf("sizeof(GPUTest)=%d\nsizeof(GPUContact)=%d\nsizeof(matrix3)=%d\n",sizeof(GPUTest),sizeof(GPUContact),sizeof(matrix3));
    const GPUTest* gputests = (const GPUTest*)tests;
    // round up to 16
    //maxPoints = (maxPoints+15)&-16;
    dim3 threads(maxPoints,1);
    dim3 grid(nbTests,1);
    CudaCollisionDetection_runTests_kernel<<< grid, threads, threads.x*sizeof(int) >>>(gputests, (int*)nresults);

}

#if defined(__cplusplus) && CUDA_VERSION != 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
