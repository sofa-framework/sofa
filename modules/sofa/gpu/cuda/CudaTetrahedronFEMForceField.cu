#include "CudaCommon.h"
#include "CudaMath.h"

#if defined(__cplusplus)
namespace sofa
{
namespace gpu
{
namespace cuda
{
#endif

extern "C"
{
    void TetrahedronFEMForceFieldCuda3f_addForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, const void* velems, void* f, const void* x, const void* v);
    void TetrahedronFEMForceFieldCuda3f_addDForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, const void* velems, void* df, const void* dx);
}

class __align__(16) GPUElement
{
public:
    /// index of the 4 connected vertices
    //Vec<4,int> tetra;
    int ia,ib,ic,id;
    /// material stiffness matrix
    //Mat<6,6,Real> K;
    float gamma_bx2, mu2_bx2;
    /// initial position of the vertices in the local (rotated) coordinate system
    //Vec3f initpos[4];
    float bx,cx;
    float cy,dx,dy,dz;
    /// strain-displacement matrix
    //Mat<12,6,Real> J;
    float Jbx_bx,Jby_bx,Jbz_bx;
    /// unused value to align to 64 bytes
    float dummy;
};

class __align__(16) GPUElementState
{
public:
    /// transposed rotation matrix
    matrix3 Rt;
    /// current internal strain
    float3 S0,S1;
    /// unused value to align to 64 bytes
    float dummy;
};

//////////////////////
// GPU-side methods //
//////////////////////

//#define USE_TEXTURE

#ifdef USE_TEXTURE

texture<float2,1,cudaReadModeElementType> texX;
const void* curX = NULL;

void setX(const void* x)
{
    if (x!=curX)
    {
        cudaBindTexture((size_t*)NULL, texX, x);
        curX = x;
    }
}

__device__ float3 getX(int i)
{
    int i2 = i + (i>>1);
    float2 x1 = tex1Dfetch(texX, i2);
    float2 x2 = tex1Dfetch(texX, i2+1);
    return (i&1)?make_float3(x1.y,x2.x,x2.y) : make_float3(x1.x,x1.y,x2.x);
}

#else

void setX(const void* x)
{
}

#define getX(i) (((const float3*)x)[i])

#endif

__global__ void TetrahedronFEMForceFieldCuda3f_calcForce_kernel(int nbElem, const GPUElement* elems, GPUElementState* state, const float* x)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index = index0+index1;

    GPUElement e = elems[index];

    GPUElementState s;

    if (index < nbElem)
    {
        float3 A = getX(e.ia); //((const float3*)x)[e.ia];
        float3 B = getX(e.ib); //((const float3*)x)[e.ib];
        B -= A;

        // Compute R
        float bx = norm(B);
        s.Rt.x = B/bx;

        // Compute JtRtX = JbtRtB + JctRtC + JdtRtD

        float3 JtRtX0,JtRtX1;

        bx -= e.bx;
        //                    ( bx)
        // RtB =              ( 0 )
        //                    ( 0 )
        // Jtb = (Jbx  0   0 )
        //       ( 0  Jby  0 )
        //       ( 0   0  Jbz)
        //       (Jby Jbx  0 )
        //       ( 0  Jbz Jby)
        //       (Jbz  0  Jbx)
        JtRtX0.x = e.Jbx_bx * bx;
        JtRtX0.y = 0;
        JtRtX0.z = 0;
        JtRtX1.x = e.Jby_bx * bx;
        JtRtX1.y = 0;
        JtRtX1.z = e.Jbz_bx * bx;

        float3 C = getX(e.ic); //((const float3*)x)[e.ic];
        C -= A;
        s.Rt.z = cross(B,C);
        s.Rt.y = cross(s.Rt.z,B);
        s.Rt.y *= invnorm(s.Rt.y);
        s.Rt.z *= invnorm(s.Rt.z);

        float cx = dot(C,s.Rt.x) - e.cx;
        float cy = dot(C,s.Rt.y) - e.cy;
        //                    ( cx)
        // RtC =              ( cy)
        //                    ( 0 )
        // Jtc = ( 0   0   0 )
        //       ( 0   dz  0 )
        //       ( 0   0  -dy)
        //       ( dz  0   0 )
        //       ( 0  -dy  dz)
        //       (-dy  0   0 )
        //JtRtX0.x += 0;
        JtRtX0.y += e.dz * cy;
        //JtRtX0.z += 0;
        JtRtX1.x += e.dz * cx;
        JtRtX1.y -= e.dy * cy;
        JtRtX1.z -= e.dy * cx;

        float3 D = getX(e.id); //((const float3*)x)[e.id];
        D -= A;

        float dx = dot(D,s.Rt.x) - e.dx;
        float dy = dot(D,s.Rt.y) - e.dy;
        float dz = dot(D,s.Rt.z) - e.dz;
        //                    ( dx)
        // RtD =              ( dy)
        //                    ( dz)
        // Jtd = ( 0   0   0 )
        //       ( 0   0   0 )
        //       ( 0   0   cy)
        //       ( 0   0   0 )
        //       ( 0   cy  0 )
        //       ( cy  0   0 )
        //JtRtX0.x += 0;
        //JtRtX0.y += 0;
        JtRtX0.z += e.cy * dz;
        //JtRtX1.x += 0;
        JtRtX1.y += e.cy * dy;
        JtRtX1.z += e.cy * dx;

        // Compute S = K JtRtX

        // K = [ gamma+mu2 gamma gamma 0 0 0 ]
        //     [ gamma gamma+mu2 gamma 0 0 0 ]
        //     [ gamma gamma gamma+mu2 0 0 0 ]
        //     [ 0 0 0             mu2/2 0 0 ]
        //     [ 0 0 0             0 mu2/2 0 ]
        //     [ 0 0 0             0 0 mu2/2 ]
        // S0 = JtRtX0*mu2 + dot(JtRtX0,(gamma gamma gamma))
        // S1 = JtRtX1*mu2/2

        s.S0  = JtRtX0*e.mu2_bx2;
        s.S0 += (JtRtX0.x+JtRtX0.y+JtRtX0.z)*e.gamma_bx2;
        s.S1  = JtRtX1*(e.mu2_bx2*0.5f);
    }

    state[index] = s;

}

__global__ void TetrahedronFEMForceFieldCuda3f_addForce_kernel(int nbVertex, unsigned int nbElemPerVertex, const GPUElement* elems, GPUElementState* state, const int* velems, float* f, const float* x)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index3 = umul24(index1,3); //3*index1;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];

    // First copy x inside temp
    int iext = umul24(blockIdx.x,BSIZE*3)+index1; //index0*3+index1;
    /*
        temp[index1        ] = x[iext        ];
        temp[index1+  BSIZE] = x[iext+  BSIZE];
        temp[index1+2*BSIZE] = x[iext+2*BSIZE];

        __syncthreads();

        float3 pos1 = make_float3(temp[index3  ],temp[index3+1],temp[index3+2]);
    */
    float3 force = make_float3(0.0f,0.0f,0.0f);

    velems+=index0*nbElemPerVertex+index1;

    if (index0+index1 < nbVertex)
        for (int s = 0; s < nbElemPerVertex; s++)
        {
            int i = *velems -1;
            velems+=BSIZE;
            if (i != -1)
            {
                int eindex = i >> 2;
                i &= 3;
                GPUElement e = elems[eindex];
                GPUElementState s = state[eindex];

                float3 Ji;

                switch (i)
                {
                case 0: // point a
                    Ji.x = -e.Jbx_bx;
                    Ji.y = -e.Jby_bx-e.dz;
                    Ji.z = -e.Jbz_bx+e.dy-e.cy;
                    break;
                case 1: // point b
                    Ji.x = e.Jbx_bx;
                    Ji.y = e.Jby_bx;
                    Ji.z = e.Jbz_bx;
                    break;
                case 2: // point c
                    Ji.x = 0;
                    Ji.y = e.dz;
                    Ji.z = -e.dy;
                    break;
                case 3: // point d
                    Ji.x = 0;
                    Ji.y = 0;
                    Ji.z = e.cy;
                    break;
                }
                // Ji  = (Jix 0   0  Jiy  0  Jiz)
                //       ( 0 Jiy  0  Jix Jiz  0 )
                //       ( 0  0  Jiz  0  Jiy Jix)
                float3 JiS;
                JiS.x = Ji.x*s.S0.x                             + Ji.y*s.S1.x               + Ji.z*s.S1.z;
                JiS.y =               Ji.y*s.S0.y               + Ji.x*s.S1.x + Ji.z*s.S1.y              ;
                JiS.z =                             Ji.z*s.S0.z               + Ji.y*s.S1.y + Ji.x*s.S1.z;

                force -= s.Rt.mulT(JiS);
            }
        }

    __syncthreads();

    temp[index3  ] = force.x;
    temp[index3+1] = force.y;
    temp[index3+2] = force.z;

    __syncthreads();

    f[iext        ] += temp[index1        ];
    f[iext+  BSIZE] += temp[index1+  BSIZE];
    f[iext+2*BSIZE] += temp[index1+2*BSIZE];
}

__global__ void TetrahedronFEMForceFieldCuda3f_calcDForce_kernel(int nbElem, const GPUElement* elems, GPUElementState* state, const float* x)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index = index0+index1;

    GPUElement e = elems[index];

    GPUElementState s = state[index];

    if (index < nbElem)
    {
        // Compute JtRtX = JbtRtB + JctRtC + JdtRtD

        float3 A = getX(e.ia); //((const float3*)x)[e.ia];
        float3 JtRtX0,JtRtX1;


        float3 B = getX(e.ib); //((const float3*)x)[e.ib];
        B = s.Rt * (B-A);

        // Jtb = (Jbx  0   0 )
        //       ( 0  Jby  0 )
        //       ( 0   0  Jbz)
        //       (Jby Jbx  0 )
        //       ( 0  Jbz Jby)
        //       (Jbz  0  Jbx)
        JtRtX0.x = e.Jbx_bx * B.x;
        JtRtX0.y =                  e.Jby_bx * B.y;
        JtRtX0.z =                                   e.Jbz_bx * B.z;
        JtRtX1.x = e.Jby_bx * B.x + e.Jbx_bx * B.y;
        JtRtX1.y =                  e.Jbz_bx * B.y + e.Jby_bx * B.z;
        JtRtX1.z = e.Jbz_bx * B.x                  + e.Jbx_bx * B.z;

        float3 C = getX(e.ic); //((const float3*)x)[e.ic];
        C = s.Rt * (C-A);

        // Jtc = ( 0   0   0 )
        //       ( 0   dz  0 )
        //       ( 0   0  -dy)
        //       ( dz  0   0 )
        //       ( 0  -dy  dz)
        //       (-dy  0   0 )
        //JtRtX0.x += 0;
        JtRtX0.y +=              e.dz * C.y;
        JtRtX0.z +=                         - e.dy * C.z;
        JtRtX1.x += e.dz * C.x;
        JtRtX1.y +=            - e.dy * C.y + e.dz * C.z;
        JtRtX1.z -= e.dy * C.x;

        // Jtd = ( 0   0   0 )
        //       ( 0   0   0 )
        //       ( 0   0   cy)
        //       ( 0   0   0 )
        //       ( 0   cy  0 )
        //       ( cy  0   0 )
        float3 D = getX(e.id); //((const float3*)x)[e.id];
        D = s.Rt * (D-A);

        //JtRtX0.x += 0;
        //JtRtX0.y += 0;
        JtRtX0.z +=                           e.cy * D.z;
        //JtRtX1.x += 0;
        JtRtX1.y +=              e.cy * D.y;
        JtRtX1.z += e.cy * D.x;

        // Compute S = K JtRtX

        // K = [ gamma+mu2 gamma gamma 0 0 0 ]
        //     [ gamma gamma+mu2 gamma 0 0 0 ]
        //     [ gamma gamma gamma+mu2 0 0 0 ]
        //     [ 0 0 0             mu2/2 0 0 ]
        //     [ 0 0 0             0 mu2/2 0 ]
        //     [ 0 0 0             0 0 mu2/2 ]
        // S0 = JtRtX0*mu2 + dot(JtRtX0,(gamma gamma gamma))
        // S1 = JtRtX1*mu2/2

        s.S0  = JtRtX0*e.mu2_bx2;
        s.S0 += (JtRtX0.x+JtRtX0.y+JtRtX0.z)*e.gamma_bx2;
        s.S1  = JtRtX1*(e.mu2_bx2*0.5f);
    }

    state[index] = s;

}

//////////////////////
// CPU-side methods //
//////////////////////

void TetrahedronFEMForceFieldCuda3f_addForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, const void* velems, void* f, const void* x, const void* v)
{
    setX(x);
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    TetrahedronFEMForceFieldCuda3f_calcForce_kernel<<< grid1, threads1>>>(nbElem, (const GPUElement*)elems, (GPUElementState*)state, (const float*)x);
    dim3 threads2(BSIZE,1);
    dim3 grid2((nbVertex+BSIZE-1)/BSIZE,1);
    TetrahedronFEMForceFieldCuda3f_addForce_kernel<<< grid2, threads2, BSIZE*3*sizeof(float) >>>(nbVertex, nbElemPerVertex, (const GPUElement*)elems, (GPUElementState*)state, (const int*)velems, (float*)f, (const float*)x);
}

void TetrahedronFEMForceFieldCuda3f_addDForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, const void* velems, void* df, const void* dx)
{
    setX(dx);
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    TetrahedronFEMForceFieldCuda3f_calcDForce_kernel<<< grid1, threads1>>>(nbElem, (const GPUElement*)elems, (GPUElementState*)state, (const float*)dx);
    dim3 threads2(BSIZE,1);
    dim3 grid2((nbVertex+BSIZE-1)/BSIZE,1);
    TetrahedronFEMForceFieldCuda3f_addForce_kernel<<< grid2, threads2, BSIZE*3*sizeof(float) >>>(nbVertex, nbElemPerVertex, (const GPUElement*)elems, (GPUElementState*)state, (const int*)velems, (float*)df, (const float*)dx);
}

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
