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
    void TetrahedronFEMForceFieldCuda3f_addForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, void* eforce, const void* velems, void* f, const void* x, const void* v);
    void TetrahedronFEMForceFieldCuda3f_addDForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* state, void* eforce, const void* velems, void* df, const void* dx);
}

class __align__(16) GPUElement
{
public:
    /// index of the 4 connected vertices
    //Vec<4,int> tetra;
    int ia[BSIZE];
    int ib[BSIZE];
    int ic[BSIZE];
    int id[BSIZE];
    /// material stiffness matrix
    //Mat<6,6,Real> K;
    float gamma_bx2[BSIZE], mu2_bx2[BSIZE];
    /// initial position of the vertices in the local (rotated) coordinate system
    //Vec3f initpos[4];
    float bx[BSIZE],cx[BSIZE];
    float cy[BSIZE],dx[BSIZE],dy[BSIZE],dz[BSIZE];
    /// strain-displacement matrix
    //Mat<12,6,Real> J;
    float Jbx_bx[BSIZE],Jby_bx[BSIZE],Jbz_bx[BSIZE];
    /// unused value to align to 64 bytes
    float dummy[BSIZE];
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


class GPUElementForce
{
public:
    float4 fA,fB,fC,fD;
};

//////////////////////
// GPU-side methods //
//////////////////////

#define USE_TEXTURE
#define USE_TEXTURE_ELEMENT_FORCE

#ifdef USE_TEXTURE

static texture<float,1,cudaReadModeElementType> texX;
static const void* curX = NULL;

static void setX(const void* x)
{
    if (x!=curX)
    {
        cudaBindTexture((size_t*)NULL, texX, x);
        curX = x;
    }
}

__device__ float3 getX(int i)
{
    int i3 = umul24(i,3);
    float x1 = tex1Dfetch(texX, i3);
    float x2 = tex1Dfetch(texX, i3+1);
    float x3 = tex1Dfetch(texX, i3+2);
    return make_float3(x1,x2,x3);
}

#else

static void setX(const void* x)
{
}

#define getX(i) (((const float3*)x)[i])

#endif

#ifdef USE_TEXTURE_ELEMENT_FORCE

static texture<float4,1,cudaReadModeElementType> texElementForce;
static const void* curElementForce = NULL;

static void setElementForce(const void* x)
{
    if (x!=curElementForce)
    {
        cudaBindTexture((size_t*)NULL, texElementForce, x);
        curElementForce = x;
    }
}

#define getElementForce(i) make_float3(tex1Dfetch(texElementForce, i));

#else

static void setElementForce(const void* x)
{
}

#define getElementForce(i) make_float3(((const float4*)eforce)[i])

#endif

__global__ void TetrahedronFEMForceFieldCuda3f_calcForce_kernel(int nbElem, const GPUElement* elems, float* rotations, float* eforce, const float* x)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index = index0+index1;

    //GPUElement e = elems[index];
    //GPUElementState s;
    const GPUElement* e = elems + blockIdx.x;
    matrix3 Rt;
    rotations += umul24(index0,9)+index1;
    //GPUElementForce f;
    float3 fB,fC,fD;

    if (index < nbElem)
    {
        float3 A = getX(e->ia[index1]); //((const float3*)x)[e.ia];
        float3 B = getX(e->ib[index1]); //((const float3*)x)[e.ib];
        B -= A;

        // Compute R
        float bx = norm(B);
        Rt.x = B/bx;
        // Compute JtRtX = JbtRtB + JctRtC + JdtRtD

        float3 JtRtX0,JtRtX1;

        bx -= e->bx[index1];
        //                    ( bx)
        // RtB =              ( 0 )
        //                    ( 0 )
        // Jtb = (Jbx  0   0 )
        //       ( 0  Jby  0 )
        //       ( 0   0  Jbz)
        //       (Jby Jbx  0 )
        //       ( 0  Jbz Jby)
        //       (Jbz  0  Jbx)
        float e_Jbx_bx = e->Jbx_bx[index1];
        float e_Jby_bx = e->Jby_bx[index1];
        float e_Jbz_bx = e->Jbz_bx[index1];
        JtRtX0.x = e_Jbx_bx * bx;
        JtRtX0.y = 0;
        JtRtX0.z = 0;
        JtRtX1.x = e_Jby_bx * bx;
        JtRtX1.y = 0;
        JtRtX1.z = e_Jbz_bx * bx;

        float3 C = getX(e->ic[index1]); //((const float3*)x)[e.ic];
        C -= A;
        Rt.z = cross(B,C);
        Rt.y = cross(Rt.z,B);
        Rt.y *= invnorm(Rt.y);
        Rt.z *= invnorm(Rt.z);

        float e_cx = e->cx[index1];
        float e_cy = e->cy[index1];
        float cx = Rt.mulX(C) - e_cx;
        float cy = Rt.mulY(C) - e_cy;
        //                    ( cx)
        // RtC =              ( cy)
        //                    ( 0 )
        // Jtc = ( 0   0   0 )
        //       ( 0   dz  0 )
        //       ( 0   0  -dy)
        //       ( dz  0   0 )
        //       ( 0  -dy  dz)
        //       (-dy  0   0 )
        float e_dy = e->dy[index1];
        float e_dz = e->dz[index1];
        //JtRtX0.x += 0;
        JtRtX0.y += e_dz * cy;
        //JtRtX0.z += 0;
        JtRtX1.x += e_dz * cx;
        JtRtX1.y -= e_dy * cy;
        JtRtX1.z -= e_dy * cx;

        float3 D = getX(e->id[index1]); //((const float3*)x)[e.id];
        D -= A;

        float e_dx = e->dx[index1];
        float dx = Rt.mulX(D) - e_dx;
        float dy = Rt.mulY(D) - e_dy;
        float dz = Rt.mulZ(D) - e_dz;
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
        JtRtX0.z += e_cy * dz;
        //JtRtX1.x += 0;
        JtRtX1.y += e_cy * dy;
        JtRtX1.z += e_cy * dx;

        // Compute S = K JtRtX

        // K = [ gamma+mu2 gamma gamma 0 0 0 ]
        //     [ gamma gamma+mu2 gamma 0 0 0 ]
        //     [ gamma gamma gamma+mu2 0 0 0 ]
        //     [ 0 0 0             mu2/2 0 0 ]
        //     [ 0 0 0             0 mu2/2 0 ]
        //     [ 0 0 0             0 0 mu2/2 ]
        // S0 = JtRtX0*mu2 + dot(JtRtX0,(gamma gamma gamma))
        // S1 = JtRtX1*mu2/2

        float e_mu2_bx2 = e->mu2_bx2[index1];
        float3 S0  = JtRtX0*e_mu2_bx2;
        S0 += (JtRtX0.x+JtRtX0.y+JtRtX0.z)*e->gamma_bx2[index1];
        float3 S1  = JtRtX1*(e_mu2_bx2*0.5f);

        // Jd = ( 0   0   0   0   0  cy )
        //      ( 0   0   0   0  cy   0 )
        //      ( 0   0   cy  0   0   0 )
        fD = (Rt.mulT(make_float3(
                e_cy * S1.z,
                e_cy * S1.y,
                e_cy * S0.z)));
        // Jc = ( 0   0   0  dz   0 -dy )
        //      ( 0   dz  0   0 -dy   0 )
        //      ( 0   0  -dy  0  dz   0 )
        fC = (Rt.mulT(make_float3(
                e_dz * S1.x - e_dy * S1.z,
                e_dz * S0.y - e_dy * S1.y,
                e_dz * S1.y - e_dy * S0.z)));
        // Jb = (Jbx  0   0  Jby  0  Jbz)
        //      ( 0  Jby  0  Jbx Jbz  0 )
        //      ( 0   0  Jbz  0  Jby Jbx)
        fB = (Rt.mulT(make_float3(
                e_Jbx_bx * S0.x                                     + e_Jby_bx * S1.x                   + e_Jbz_bx * S1.z,
                e_Jby_bx * S0.y                   + e_Jbx_bx * S1.x + e_Jbz_bx * S1.y,
                e_Jbz_bx * S0.z                   + e_Jby_bx * S1.y + e_Jbx_bx * S1.z)));
        //fA.x = -(fB.x+fC.x+fD.x);
        //fA.y = -(fB.y+fC.y+fD.y);
        //fA.z = -(fB.z+fC.z+fD.z);
    }

    //state[index] = s;
    Rt.writeAoS(rotations);
    //((rmatrix3*)rotations)[index] = Rt;
    //((GPUElementForce*)eforce)[index] = f;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];
    int index13 = umul24(index1,13);
    temp[index13+0 ] = -(fB.x+fC.x+fD.x);
    temp[index13+1 ] = -(fB.y+fC.y+fD.y);
    temp[index13+2 ] = -(fB.z+fC.z+fD.z);
    temp[index13+3 ] = fB.x;
    temp[index13+4 ] = fB.y;
    temp[index13+5 ] = fB.z;
    temp[index13+6 ] = fC.x;
    temp[index13+7 ] = fC.y;
    temp[index13+8 ] = fC.z;
    temp[index13+9 ] = fD.x;
    temp[index13+10] = fD.y;
    temp[index13+11] = fD.z;
    __syncthreads();
    float* out = ((float*)eforce)+(umul24(blockIdx.x,BSIZE*16))+index1;
    float v = 0;
    bool read = true; //(index1&4)<3;
    index1 += (index1>>4) - (index1>>2); // remove one for each 4-values before this thread, but add an extra one each 16 threads (so each 12 input cells, to align to 13)

    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;

}

__global__ void TetrahedronFEMForceFieldCuda3f_addForce_kernel(int nbVertex, unsigned int nbElemPerVertex, const float* eforce, /* const GPUElement* elems, GPUElementState* state, */ const int* velems, float* f, const float* x)
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

    velems+=umul24(index0,nbElemPerVertex)+index1;

    if (index0+index1 < nbVertex)
        for (int s = 0; s < nbElemPerVertex; s++)
        {
            int i = *velems -1;
            velems+=BSIZE;
            if (i != -1)
            {
                force -= getElementForce(i);
#if 0
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
#endif
            }
        }

//    __syncthreads();

    temp[index3  ] = force.x;
    temp[index3+1] = force.y;
    temp[index3+2] = force.z;

    __syncthreads();

    f[iext        ] += temp[index1        ];
    f[iext+  BSIZE] += temp[index1+  BSIZE];
    f[iext+2*BSIZE] += temp[index1+2*BSIZE];
}

__global__ void TetrahedronFEMForceFieldCuda3f_calcDForce_kernel(int nbElem, const GPUElement* elems, const float* rotations, float* eforce, const float* x)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index = index0+index1;

    //GPUElement e = elems[index];
    const GPUElement* e = elems + blockIdx.x;
    //GPUElementState s = state[index];
    //GPUElementForce f;
    float3 fB,fC,fD;
    matrix3 Rt;
    rotations += umul24(index0,9)+index1;
    Rt.readAoS(rotations);
    //Rt = ((const rmatrix3*)rotations)[index];

    if (index < nbElem)
    {
        // Compute JtRtX = JbtRtB + JctRtC + JdtRtD

        float3 A = getX(e->ia[index1]); //((const float3*)x)[e.ia];
        float3 JtRtX0,JtRtX1;


        float3 B = getX(e->ib[index1]); //((const float3*)x)[e.ib];
        B = Rt * (B-A);

        // Jtb = (Jbx  0   0 )
        //       ( 0  Jby  0 )
        //       ( 0   0  Jbz)
        //       (Jby Jbx  0 )
        //       ( 0  Jbz Jby)
        //       (Jbz  0  Jbx)
        float e_Jbx_bx = e->Jbx_bx[index1];
        float e_Jby_bx = e->Jby_bx[index1];
        float e_Jbz_bx = e->Jbz_bx[index1];
        JtRtX0.x = e_Jbx_bx * B.x;
        JtRtX0.y =                  e_Jby_bx * B.y;
        JtRtX0.z =                                   e_Jbz_bx * B.z;
        JtRtX1.x = e_Jby_bx * B.x + e_Jbx_bx * B.y;
        JtRtX1.y =                  e_Jbz_bx * B.y + e_Jby_bx * B.z;
        JtRtX1.z = e_Jbz_bx * B.x                  + e_Jbx_bx * B.z;

        float3 C = getX(e->ic[index1]); //((const float3*)x)[e.ic];
        C = Rt * (C-A);

        // Jtc = ( 0   0   0 )
        //       ( 0   dz  0 )
        //       ( 0   0  -dy)
        //       ( dz  0   0 )
        //       ( 0  -dy  dz)
        //       (-dy  0   0 )
        float e_dy = e->dy[index1];
        float e_dz = e->dz[index1];
        //JtRtX0.x += 0;
        JtRtX0.y +=              e_dz * C.y;
        JtRtX0.z +=                         - e_dy * C.z;
        JtRtX1.x += e_dz * C.x;
        JtRtX1.y +=            - e_dy * C.y + e_dz * C.z;
        JtRtX1.z -= e_dy * C.x;

        // Jtd = ( 0   0   0 )
        //       ( 0   0   0 )
        //       ( 0   0   cy)
        //       ( 0   0   0 )
        //       ( 0   cy  0 )
        //       ( cy  0   0 )
        float3 D = getX(e->id[index1]); //((const float3*)x)[e.id];
        D = Rt * (D-A);

        float e_cy = e->cy[index1];
        //JtRtX0.x += 0;
        //JtRtX0.y += 0;
        JtRtX0.z +=                           e_cy * D.z;
        //JtRtX1.x += 0;
        JtRtX1.y +=              e_cy * D.y;
        JtRtX1.z += e_cy * D.x;

        // Compute S = K JtRtX

        // K = [ gamma+mu2 gamma gamma 0 0 0 ]
        //     [ gamma gamma+mu2 gamma 0 0 0 ]
        //     [ gamma gamma gamma+mu2 0 0 0 ]
        //     [ 0 0 0             mu2/2 0 0 ]
        //     [ 0 0 0             0 mu2/2 0 ]
        //     [ 0 0 0             0 0 mu2/2 ]
        // S0 = JtRtX0*mu2 + dot(JtRtX0,(gamma gamma gamma))
        // S1 = JtRtX1*mu2/2

        float e_mu2_bx2 = e->mu2_bx2[index1];
        float3 S0  = JtRtX0*e_mu2_bx2;
        S0 += (JtRtX0.x+JtRtX0.y+JtRtX0.z)*e->gamma_bx2[index1];
        float3 S1  = JtRtX1*(e_mu2_bx2*0.5f);

        // Jd = ( 0   0   0   0   0  cy )
        //      ( 0   0   0   0  cy   0 )
        //      ( 0   0   cy  0   0   0 )
        fD = (Rt.mulT(make_float3(
                e_cy * S1.z,
                e_cy * S1.y,
                e_cy * S0.z)));
        // Jc = ( 0   0   0  dz   0 -dy )
        //      ( 0   dz  0   0 -dy   0 )
        //      ( 0   0  -dy  0  dz   0 )
        fC = (Rt.mulT(make_float3(
                e_dz * S1.x - e_dy * S1.z,
                e_dz * S0.y - e_dy * S1.y,
                e_dz * S1.y - e_dy * S0.z)));
        // Jb = (Jbx  0   0  Jby  0  Jbz)
        //      ( 0  Jby  0  Jbx Jbz  0 )
        //      ( 0   0  Jbz  0  Jby Jbx)
        fB = (Rt.mulT(make_float3(
                e_Jbx_bx * S0.x                                     + e_Jby_bx * S1.x                   + e_Jbz_bx * S1.z,
                e_Jby_bx * S0.y                   + e_Jbx_bx * S1.x + e_Jbz_bx * S1.y,
                e_Jbz_bx * S0.z                   + e_Jby_bx * S1.y + e_Jbx_bx * S1.z)));
        //fA.x = -(fB.x+fC.x+fD.x);
        //fA.y = -(fB.y+fC.y+fD.y);
        //fA.z = -(fB.z+fC.z+fD.z);
    }

    //state[index] = s;
    //((GPUElementForce*)eforce)[index] = f;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];
    int index13 = umul24(index1,13);
    temp[index13+0 ] = -(fB.x+fC.x+fD.x);
    temp[index13+1 ] = -(fB.y+fC.y+fD.y);
    temp[index13+2 ] = -(fB.z+fC.z+fD.z);
    temp[index13+3 ] = fB.x;
    temp[index13+4 ] = fB.y;
    temp[index13+5 ] = fB.z;
    temp[index13+6 ] = fC.x;
    temp[index13+7 ] = fC.y;
    temp[index13+8 ] = fC.z;
    temp[index13+9 ] = fD.x;
    temp[index13+10] = fD.y;
    temp[index13+11] = fD.z;
    __syncthreads();
    float* out = ((float*)eforce)+(umul24(blockIdx.x,BSIZE*16))+index1;
    float v = 0;
    bool read = true; //(index1&4)<3;
    index1 += (index1>>4) - (index1>>2); // remove one for each 4-values before this thread, but add an extra one each 16 threads (so each 12 input cells, to align to 13)

    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
}

//////////////////////
// CPU-side methods //
//////////////////////

void TetrahedronFEMForceFieldCuda3f_addForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, void* eforce, const void* velems, void* f, const void* x, const void* v)
{
    setX(x);
    setElementForce(eforce);
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    TetrahedronFEMForceFieldCuda3f_calcForce_kernel<<< grid1, threads1, BSIZE*13*sizeof(float)>>>(nbElem, (const GPUElement*)elems, (float*)state, (float*)eforce, (const float*)x);
    dim3 threads2(BSIZE,1);
    dim3 grid2((nbVertex+BSIZE-1)/BSIZE,1);
    TetrahedronFEMForceFieldCuda3f_addForce_kernel<<< grid2, threads2, BSIZE*3*sizeof(float) >>>(nbVertex, nbElemPerVertex, (const float*)eforce, (const int*)velems, (float*)f, (const float*)x);
}

void TetrahedronFEMForceFieldCuda3f_addDForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* state, void* eforce, const void* velems, void* df, const void* dx)
{
    setX(dx);
    setElementForce(eforce);
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    TetrahedronFEMForceFieldCuda3f_calcDForce_kernel<<< grid1, threads1, BSIZE*13*sizeof(float)>>>(nbElem, (const GPUElement*)elems, (const float*)state, (float*)eforce, (const float*)dx);
    dim3 threads2(BSIZE,1);
    dim3 grid2((nbVertex+BSIZE-1)/BSIZE,1);
    TetrahedronFEMForceFieldCuda3f_addForce_kernel<<< grid2, threads2, BSIZE*3*sizeof(float) >>>(nbVertex, nbElemPerVertex, (const float*)eforce, (const int*)velems, (float*)df, (const float*)dx);
}

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
