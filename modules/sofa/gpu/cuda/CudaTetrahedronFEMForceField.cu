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

__global__ void TetrahedronFEMForceFieldCuda3f_calcForce_kernel(int nbElem, const GPUElement* elems, GPUElementState* state, const float* x)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index = index0+index1;

    GPUElement e = elems[index];

    GPUElementState s;

    if (index < nbElem)
    {
        float3 B = ((const float3*)x)[e.ib];
        float3 C = ((const float3*)x)[e.ic];
        float3 D = ((const float3*)x)[e.id];
        float3 A = ((const float3*)x)[e.ia];

        B -= A;
        C -= A;
        D -= A;

        // Compute R
        float bx = norm(B);
        s.Rt.x = B/bx;
        s.Rt.z = cross(B,C);
        s.Rt.y = cross(s.Rt.z,B);
        s.Rt.y *= invnorm(s.Rt.y);
        s.Rt.z *= invnorm(s.Rt.z);

        // Compute JtRtX = JbtRtB + JctRtC + JdtRtD

        float3 JtRtX0,JtRtX1;

        // RtB = (bx 0 0)
        // Jb  = (Jbx 0 0 Jby 0 Jbz)
        //       (0 Jby 0 Jbx Jbz 0)
        //       (0 0 Jbz 0 Jby Jbx)
        JtRtX0.x = e.Jbx_bx * bx;
        JtRtX0.y = 0;
        JtRtX0.z = 0;
        JtRtX1.x = e.Jby_bx * bx;
        JtRtX1.y = 0;
        JtRtX1.z = e.Jbz_bx * bx;

        // RtC = (cx cy 0)
        // Jc  = (0 0   0 dz 0 -dy)
        //       (0 dz  0 0 -dy  0)
        //       (0 0 -dy 0  dz  0)
        float cx = dot(C,s.Rt.x);
        float cy = dot(C,s.Rt.y);

        JtRtX0.y += e.dz * cy;
        JtRtX1.x += e.dz * cx;
        JtRtX1.y -= e.dy * cy;
        JtRtX1.z -= e.dy * cx;

        // RtD = (dx dy dz)
        // Jd  = (0 0  0 0 0 cy)
        //       (0 0  0 0 cy 0)
        //       (0 0 cy 0  0 0)
        float dx = dot(D,s.Rt.x);
        float dy = dot(D,s.Rt.y);
        float dz = dot(D,s.Rt.z);

        JtRtX0.z += e.cy * dz;
        JtRtX1.y -= e.dy * dy;
        JtRtX1.z -= e.dy * dx;

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
        s.S0 += JtRtX0*e.gamma_bx2;
        s.S1  = JtRtX1*(e.mu2_bx2*0.5f);
    }

    state[index0+index1] = s;

}

__global__ void TetrahedronFEMForceFieldCuda3f_addForce_kernel(unsigned int nbElemPerVertex, const GPUElement* elems, GPUElementState* state, const int* velems, float* f, const float* x)
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

    for (int s = 0; s < nbElemPerVertex; s++)
    {
        int i = *velems;
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
            float3 JiS = mul(Ji,s.S0);
            JiS.x += Ji.y*s.S1.x + Ji.z*s.S1.z;
            JiS.y += Ji.x*s.S1.x + Ji.z*s.S1.y;
            JiS.z += Ji.y*s.S1.y + Ji.x*s.S1.z;

            force += s.Rt.mulT(JiS);
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

        float3 A = ((const float3*)x)[e.ia];
        float3 JtRtX0,JtRtX1;


        float3 B = ((const float3*)x)[e.ib];
        B = s.Rt * (B-A);

        // Jb  = (Jbx 0 0 Jby 0 Jbz)
        //       (0 Jby 0 Jbx Jbz 0)
        //       (0 0 Jbz 0 Jby Jbx)
        JtRtX0.x = e.Jbx_bx * B.x;
        JtRtX0.y = e.Jby_bx * B.y;
        JtRtX0.z = e.Jbz_bx * B.x;
        JtRtX1.x = e.Jby_bx * B.x + e.Jbx_bx * B.y;
        JtRtX1.y = e.Jbz_bx * B.y + e.Jby_bx * B.z;
        JtRtX1.z = e.Jbz_bx * B.x + e.Jbx_bx * B.z;

        float3 C = ((const float3*)x)[e.ic];
        C = s.Rt * (C-A);

        // Jc  = (0 0   0 dz 0 -dy)
        //       (0 dz  0 0 -dy  0)
        //       (0 0 -dy 0  dz  0)

        JtRtX0.y += e.dz * C.y;
        JtRtX0.z -= e.dy * C.z;
        JtRtX1.x += e.dz * C.x;
        JtRtX1.y += e.dz * C.z - e.dy * C.y;
        JtRtX1.z -= e.dy * C.x;

        // Jd  = (0 0  0 0 0 cy)
        //       (0 0  0 0 cy 0)
        //       (0 0 cy 0  0 0)
        float3 D = ((const float3*)x)[e.id];
        D = s.Rt * (D-A);

        JtRtX0.z += e.cy * D.z;
        JtRtX1.y -= e.dy * D.y;
        JtRtX1.z -= e.dy * D.x;

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
        s.S0 += JtRtX0*e.gamma_bx2;
        s.S1  = JtRtX1*(e.mu2_bx2*0.5f);
    }

    state[index0+index1] = s;

}

//////////////////////
// CPU-side methods //
//////////////////////

void TetrahedronFEMForceFieldCuda3f_addForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, const void* velems, void* f, const void* x, const void* v)
{
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    TetrahedronFEMForceFieldCuda3f_calcForce_kernel<<< grid1, threads1>>>(nbElem, (const GPUElement*)elems, (GPUElementState*)state, (const float*)x);
    dim3 threads2(BSIZE,1);
    dim3 grid2((nbVertex+BSIZE-1)/BSIZE,1);
    TetrahedronFEMForceFieldCuda3f_addForce_kernel<<< grid2, threads2, BSIZE*3*sizeof(float) >>>(nbElemPerVertex, (const GPUElement*)elems, (GPUElementState*)state, (const int*)velems, (float*)f, (const float*)x);
}

void TetrahedronFEMForceFieldCuda3f_addDForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, const void* velems, void* df, const void* dx)
{
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    TetrahedronFEMForceFieldCuda3f_calcDForce_kernel<<< grid1, threads1>>>(nbElem, (const GPUElement*)elems, (GPUElementState*)state, (const float*)dx);
    dim3 threads2(BSIZE,1);
    dim3 grid2((nbVertex+BSIZE-1)/BSIZE,1);
    TetrahedronFEMForceFieldCuda3f_addForce_kernel<<< grid2, threads2, BSIZE*3*sizeof(float) >>>(nbElemPerVertex, (const GPUElement*)elems, (GPUElementState*)state, (const int*)velems, (float*)df, (const float*)dx);
}

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
