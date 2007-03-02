#include "CudaCommon.h"
#include "CudaMath.h"

extern "C"
{
    void SpringForceFieldCuda3f_addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v);
    void SpringForceFieldCuda3f_addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2);
    void StiffSpringForceFieldCuda3f_addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v, void* dfdx);
    void StiffSpringForceFieldCuda3f_addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2, void* dfdx);
    void StiffSpringForceFieldCuda3f_addDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* dx, const void* x, const void* dfdx);
    void StiffSpringForceFieldCuda3f_addExternalDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* dx1, const void* x1, const void* dx2, const void* x2, const void* dfdx);
}

struct GPUSpring
{
    int index; ///< -1 if no spring
    float initpos;
    float ks;
    float kd;
};

//////////////////////
// GPU-side methods //
//////////////////////

__global__ void SpringForceFieldCuda3f_addExternalForce_kernel(unsigned int nbSpringPerVertex, const GPUSpring* springs, float* f1, const float* x1, const float* v1, const float* x2, const float* v2)
{
    int index0 = blockIdx.x*BSIZE; //blockDim.x;
    int index1 = threadIdx.x;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];

    // First copy x and v inside temp
    int iext = index0*3+index1;
    temp[index1        ] = x1[iext        ];
    temp[index1+  BSIZE] = x1[iext+  BSIZE];
    temp[index1+2*BSIZE] = x1[iext+2*BSIZE];
    temp[index1+3*BSIZE] = v1[iext        ];
    temp[index1+4*BSIZE] = v1[iext+  BSIZE];
    temp[index1+5*BSIZE] = v1[iext+2*BSIZE];

    __syncthreads();

    int index3 = 3*index1;
    float3 pos1 = make_float3(temp[index3  ],temp[index3+1],temp[index3+2]);
    float3 vel1 = make_float3(temp[index3  +3*BSIZE],temp[index3+1+3*BSIZE],temp[index3+2+3*BSIZE]);
    float3 force = make_float3(0.0f,0.0f,0.0f);

    springs+=index0*nbSpringPerVertex+index1;

    for (int s = 0; s < nbSpringPerVertex; s++)
    {
        GPUSpring spring = *springs;
        springs+=BSIZE;
        if (spring.index != -1)
        {
            //Coord u = p2[b]-p1[a];
            //Real d = u.norm();
            //Real inverseLength = 1.0f/d;
            //u *= inverseLength;
            //Real elongation = (Real)(d - spring.initpos);
            //ener += elongation * elongation * spring.ks /2;
            //Deriv relativeVelocity = v2[b]-v1[a];
            //Real elongationVelocity = dot(u,relativeVelocity);
            //Real forceIntensity = (Real)(spring.ks*elongation+spring.kd*elongationVelocity);
            //Deriv force = u*forceIntensity;
            //f1[a]+=force;
            //f2[b]-=force;

            float3 u, relativeVelocity;

            {
                // general case
                u = ((const float3*)x2)[spring.index];
                relativeVelocity = ((const float3*)v2)[spring.index];
            }

            u -= pos1;
            relativeVelocity -= vel1;

            float inverseLength = 1/sqrt(dot(u,u));
            float d = 1/inverseLength;
            u *= inverseLength;
            float elongation = d - spring.initpos;
            float elongationVelocity = dot(u,relativeVelocity);
            float forceIntensity = spring.ks*elongation+spring.kd*elongationVelocity;
            force += u*forceIntensity;
        }
    }

    __syncthreads();

    temp[index3  ] = force.x;
    temp[index3+1] = force.y;
    temp[index3+2] = force.z;

    __syncthreads();

    f1[iext        ] += temp[index1        ];
    f1[iext+  BSIZE] += temp[index1+  BSIZE];
    f1[iext+2*BSIZE] += temp[index1+2*BSIZE];
}

__global__ void SpringForceFieldCuda3f_addForce_kernel(unsigned int nbSpringPerVertex, const GPUSpring* springs, float* f, const float* x, const float* v)
{
    int index0 = blockIdx.x*BSIZE; //blockDim.x;
    int index1 = threadIdx.x;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];

    // First copy x and v inside temp
    int iext = index0*3+index1;
    temp[index1        ] = x[iext        ];
    temp[index1+  BSIZE] = x[iext+  BSIZE];
    temp[index1+2*BSIZE] = x[iext+2*BSIZE];
    temp[index1+3*BSIZE] = v[iext        ];
    temp[index1+4*BSIZE] = v[iext+  BSIZE];
    temp[index1+5*BSIZE] = v[iext+2*BSIZE];

    __syncthreads();

    int index3 = 3*index1;
    float3 pos1 = make_float3(temp[index3  ],temp[index3+1],temp[index3+2]);
    float3 vel1 = make_float3(temp[index3  +3*BSIZE],temp[index3+1+3*BSIZE],temp[index3+2+3*BSIZE]);
    float3 force = make_float3(0.0f,0.0f,0.0f);

    springs+=index0*nbSpringPerVertex+index1;

    for (int s = 0; s < nbSpringPerVertex; s++)
    {
        GPUSpring spring = *springs;
        springs+=BSIZE;
        if (spring.index != -1)
        {
            //Coord u = p2[b]-p1[a];
            //Real d = u.norm();
            //Real inverseLength = 1.0f/d;
            //u *= inverseLength;
            //Real elongation = (Real)(d - spring.initpos);
            //ener += elongation * elongation * spring.ks /2;
            //Deriv relativeVelocity = v2[b]-v1[a];
            //Real elongationVelocity = dot(u,relativeVelocity);
            //Real forceIntensity = (Real)(spring.ks*elongation+spring.kd*elongationVelocity);
            //Deriv force = u*forceIntensity;
            //f1[a]+=force;
            //f2[b]-=force;

            float3 u, relativeVelocity;

            if (spring.index >= index0 && spring.index < index0+BSIZE)
            {
                // 'local' point
                int i = spring.index - index0;
                u = make_float3(temp[3*i  ], temp[3*i+1], temp[3*i+2]);
                relativeVelocity = make_float3(temp[3*i  +3*BSIZE], temp[3*i+1+3*BSIZE], temp[3*i+2+3*BSIZE]);
            }
            else
            {
                // general case
                u = ((const float3*)x)[spring.index];
                relativeVelocity = ((const float3*)v)[spring.index];
            }

            u -= pos1;
            relativeVelocity -= vel1;

            float inverseLength = 1/sqrt(dot(u,u));
            float d = 1/inverseLength;
            u *= inverseLength;
            float elongation = d - spring.initpos;
            float elongationVelocity = dot(u,relativeVelocity);
            float forceIntensity = spring.ks*elongation+spring.kd*elongationVelocity;
            force += u*forceIntensity;
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

__global__ void StiffSpringForceFieldCuda3f_addExternalForce_kernel(unsigned int nbSpringPerVertex, const GPUSpring* springs, float* f1, const float* x1, const float* v1, const float* x2, const float* v2, float* dfdx)
{
    int index0 = blockIdx.x*BSIZE; //blockDim.x;
    int index1 = threadIdx.x;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];

    // First copy x and v inside temp
    int iext = index0*3+index1;
    temp[index1        ] = x1[iext        ];
    temp[index1+  BSIZE] = x1[iext+  BSIZE];
    temp[index1+2*BSIZE] = x1[iext+2*BSIZE];
    temp[index1+3*BSIZE] = v1[iext        ];
    temp[index1+4*BSIZE] = v1[iext+  BSIZE];
    temp[index1+5*BSIZE] = v1[iext+2*BSIZE];

    __syncthreads();

    int index3 = 3*index1;
    float3 pos1 = make_float3(temp[index3  ],temp[index3+1],temp[index3+2]);
    float3 vel1 = make_float3(temp[index3  +3*BSIZE],temp[index3+1+3*BSIZE],temp[index3+2+3*BSIZE]);
    float3 force = make_float3(0.0f,0.0f,0.0f);

    springs+=index0*nbSpringPerVertex+index1;
    dfdx+=index0*nbSpringPerVertex+index1;

    for (int s = 0; s < nbSpringPerVertex; s++)
    {
        GPUSpring spring = *springs;
        springs+=BSIZE;
        if (spring.index != -1)
        {
            //Coord u = p2[b]-p1[a];
            //Real d = u.norm();
            //Real inverseLength = 1.0f/d;
            //u *= inverseLength;
            //Real elongation = (Real)(d - spring.initpos);
            //ener += elongation * elongation * spring.ks /2;
            //Deriv relativeVelocity = v2[b]-v1[a];
            //Real elongationVelocity = dot(u,relativeVelocity);
            //Real forceIntensity = (Real)(spring.ks*elongation+spring.kd*elongationVelocity);
            //Deriv force = u*forceIntensity;
            //f1[a]+=force;
            //f2[b]-=force;

            float3 u, relativeVelocity;

            {
                // general case
                u = ((const float3*)x2)[spring.index];
                relativeVelocity = ((const float3*)v2)[spring.index];
            }

            u -= pos1;
            relativeVelocity -= vel1;

            float inverseLength = 1/sqrt(dot(u,u));
            float d = 1/inverseLength;
            u *= inverseLength;
            float elongation = d - spring.initpos;
            float elongationVelocity = dot(u,relativeVelocity);
            float forceIntensity = spring.ks*elongation+spring.kd*elongationVelocity;
            force += u*forceIntensity;

            *dfdx = forceIntensity*inverseLength;
        }
        dfdx+=BSIZE;
    }

    __syncthreads();

    temp[index3  ] = force.x;
    temp[index3+1] = force.y;
    temp[index3+2] = force.z;

    __syncthreads();

    f1[iext        ] += temp[index1        ];
    f1[iext+  BSIZE] += temp[index1+  BSIZE];
    f1[iext+2*BSIZE] += temp[index1+2*BSIZE];
}

__global__ void StiffSpringForceFieldCuda3f_addForce_kernel(unsigned int nbSpringPerVertex, const GPUSpring* springs, float* f, const float* x, const float* v, float* dfdx)
{
    int index0 = blockIdx.x*BSIZE; //blockDim.x;
    int index1 = threadIdx.x;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];

    // First copy x and v inside temp
    int iext = index0*3+index1;
    temp[index1        ] = x[iext        ];
    temp[index1+  BSIZE] = x[iext+  BSIZE];
    temp[index1+2*BSIZE] = x[iext+2*BSIZE];
    temp[index1+3*BSIZE] = v[iext        ];
    temp[index1+4*BSIZE] = v[iext+  BSIZE];
    temp[index1+5*BSIZE] = v[iext+2*BSIZE];

    __syncthreads();

    int index3 = 3*index1;
    float3 pos1 = make_float3(temp[index3  ],temp[index3+1],temp[index3+2]);
    float3 vel1 = make_float3(temp[index3  +3*BSIZE],temp[index3+1+3*BSIZE],temp[index3+2+3*BSIZE]);
    float3 force = make_float3(0.0f,0.0f,0.0f);

    springs+=index0*nbSpringPerVertex+index1;
    dfdx+=index0*nbSpringPerVertex+index1;

    for (int s = 0; s < nbSpringPerVertex; s++)
    {
        GPUSpring spring = *springs;
        springs+=BSIZE;
        if (spring.index != -1)
        {
            //Coord u = p2[b]-p1[a];
            //Real d = u.norm();
            //Real inverseLength = 1.0f/d;
            //u *= inverseLength;
            //Real elongation = (Real)(d - spring.initpos);
            //ener += elongation * elongation * spring.ks /2;
            //Deriv relativeVelocity = v2[b]-v1[a];
            //Real elongationVelocity = dot(u,relativeVelocity);
            //Real forceIntensity = (Real)(spring.ks*elongation+spring.kd*elongationVelocity);
            //Deriv force = u*forceIntensity;
            //f1[a]+=force;
            //f2[b]-=force;

            float3 u, relativeVelocity;

            if (spring.index >= index0 && spring.index < index0+BSIZE)
            {
                // 'local' point
                int i = spring.index - index0;
                u = make_float3(temp[3*i  ], temp[3*i+1], temp[3*i+2]);
                relativeVelocity = make_float3(temp[3*i  +3*BSIZE], temp[3*i+1+3*BSIZE], temp[3*i+2+3*BSIZE]);
            }
            else
            {
                // general case
                u = ((const float3*)x)[spring.index];
                relativeVelocity = ((const float3*)v)[spring.index];
            }

            u -= pos1;
            relativeVelocity -= vel1;

            float inverseLength = 1/sqrt(dot(u,u));
            float d = 1/inverseLength;
            u *= inverseLength;
            float elongation = d - spring.initpos;
            float elongationVelocity = dot(u,relativeVelocity);
            float forceIntensity = spring.ks*elongation+spring.kd*elongationVelocity;
            force += u*forceIntensity;

            *dfdx = forceIntensity*inverseLength;
        }
        dfdx+=BSIZE;
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

__global__ void StiffSpringForceFieldCuda3f_addExternalDForce_kernel(unsigned int nbSpringPerVertex, const GPUSpring* springs, float* f1, const float* dx1, const float* x1, const float* dx2, const float* x2, const float* dfdx)
{
    int index0 = blockIdx.x*BSIZE; //blockDim.x;
    int index1 = threadIdx.x;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];

    // First copy dx and x inside temp
    int iext = index0*3+index1;
    temp[index1        ] = dx1[iext        ];
    temp[index1+  BSIZE] = dx1[iext+  BSIZE];
    temp[index1+2*BSIZE] = dx1[iext+2*BSIZE];
    temp[index1+3*BSIZE] = x1[iext        ];
    temp[index1+4*BSIZE] = x1[iext+  BSIZE];
    temp[index1+5*BSIZE] = x1[iext+2*BSIZE];

    __syncthreads();

    int index3 = 3*index1;
    float3 dpos1 = make_float3(temp[index3  ],temp[index3+1],temp[index3+2]);
    float3 pos1 = make_float3(temp[index3  +3*BSIZE],temp[index3+1+3*BSIZE],temp[index3+2+3*BSIZE]);
    float3 dforce = make_float3(0.0f,0.0f,0.0f);

    springs+=index0*nbSpringPerVertex+index1;
    dfdx+=index0*nbSpringPerVertex+index1;

    for (int s = 0; s < nbSpringPerVertex; s++)
    {
        GPUSpring spring = *springs;
        springs+=BSIZE;
        if (spring.index != -1)
        {
            float tgt = *dfdx;
            float3 du;
            float3 u;

            {
                // general case
                du = ((const float3*)dx2)[spring.index];
                u = ((const float3*)x2)[spring.index];
            }

            du -= dpos1;
            u -= pos1;

            float uxux = u.x*u.x;
            float uyuy = u.y*u.y;
            float uzuz = u.z*u.z;
            float uxuy = u.x*u.y;
            float uxuz = u.x*u.z;
            float uyuz = u.y*u.z;
            float fact = (spring.ks-tgt)/(uxux+uyuy+uzuz);
            dforce.x += fact*(uxux*du.x+uxuy*du.y+uxuz*du.z)+tgt*du.x;
            dforce.y += fact*(uxuy*du.x+uyuy*du.y+uyuz*du.z)+tgt*du.y;
            dforce.z += fact*(uxuz*du.x+uyuz*du.y+uzuz*du.z)+tgt*du.z;
        }
        dfdx+=BSIZE;
    }

    __syncthreads();

    temp[index3  ] = dforce.x;
    temp[index3+1] = dforce.y;
    temp[index3+2] = dforce.z;

    __syncthreads();

    f1[iext        ] += temp[index1        ];
    f1[iext+  BSIZE] += temp[index1+  BSIZE];
    f1[iext+2*BSIZE] += temp[index1+2*BSIZE];
}

__global__ void StiffSpringForceFieldCuda3f_addDForce_kernel(unsigned int nbSpringPerVertex, const GPUSpring* springs, float* f, const float* dx, const float* x, const float* dfdx)
{
    int index0 = blockIdx.x*BSIZE; //blockDim.x;
    int index1 = threadIdx.x;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];

    // First copy dx and x inside temp
    int iext = index0*3+index1;
    temp[index1        ] = dx[iext        ];
    temp[index1+  BSIZE] = dx[iext+  BSIZE];
    temp[index1+2*BSIZE] = dx[iext+2*BSIZE];
    temp[index1+3*BSIZE] = x[iext        ];
    temp[index1+4*BSIZE] = x[iext+  BSIZE];
    temp[index1+5*BSIZE] = x[iext+2*BSIZE];

    __syncthreads();

    int index3 = 3*index1;
    float3 dpos1 = make_float3(temp[index3  ],temp[index3+1],temp[index3+2]);
    float3 pos1 = make_float3(temp[index3  +3*BSIZE],temp[index3+1+3*BSIZE],temp[index3+2+3*BSIZE]);
    float3 dforce = make_float3(0.0f,0.0f,0.0f);

    springs+=index0*nbSpringPerVertex+index1;
    dfdx+=index0*nbSpringPerVertex+index1;

    for (int s = 0; s < nbSpringPerVertex; s++)
    {
        GPUSpring spring = *springs;
        springs+=BSIZE;
        if (spring.index != -1)
        {
            float tgt = *dfdx;
            float3 du;
            float3 u;

            if (spring.index >= index0 && spring.index < index0+BSIZE)
            {
                // 'local' point
                int i = spring.index - index0;
                du = make_float3(temp[3*i  ], temp[3*i+1], temp[3*i+2]);
                u = make_float3(temp[3*i  +3*BSIZE], temp[3*i+1+3*BSIZE], temp[3*i+2+3*BSIZE]);
            }
            else
            {
                // general case
                du = ((const float3*)dx)[spring.index];
                u = ((const float3*)x)[spring.index];
            }

            du -= dpos1;
            u -= pos1;

            float uxux = u.x*u.x;
            float uyuy = u.y*u.y;
            float uzuz = u.z*u.z;
            float uxuy = u.x*u.y;
            float uxuz = u.x*u.z;
            float uyuz = u.y*u.z;
            float fact = (spring.ks-tgt)/(uxux+uyuy+uzuz);
            dforce.x += fact*(uxux*du.x+uxuy*du.y+uxuz*du.z)+tgt*du.x;
            dforce.y += fact*(uxuy*du.x+uyuy*du.y+uyuz*du.z)+tgt*du.y;
            dforce.z += fact*(uxuz*du.x+uyuz*du.y+uzuz*du.z)+tgt*du.z;
        }
        dfdx+=BSIZE;
    }

    __syncthreads();

    temp[index3  ] = dforce.x;
    temp[index3+1] = dforce.y;
    temp[index3+2] = dforce.z;

    __syncthreads();

    f[iext        ] += temp[index1        ];
    f[iext+  BSIZE] += temp[index1+  BSIZE];
    f[iext+2*BSIZE] += temp[index1+2*BSIZE];
}

//////////////////////
// CPU-side methods //
//////////////////////

void SpringForceFieldCuda3f_addForce(unsigned int size, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    SpringForceFieldCuda3f_addForce_kernel<<< grid, threads, BSIZE*6*sizeof(float) >>>(nbSpringPerVertex, (const GPUSpring*)springs, (float*)f, (const float*)x, (const float*)v);
}

void SpringForceFieldCuda3f_addExternalForce(unsigned int size, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    SpringForceFieldCuda3f_addExternalForce_kernel<<< grid, threads, BSIZE*6*sizeof(float) >>>(nbSpringPerVertex, (const GPUSpring*)springs, (float*)f1, (const float*)x1, (const float*)v1, (const float*)x2, (const float*)v2);
}

void StiffSpringForceFieldCuda3f_addForce(unsigned int size, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v, void* dfdx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    StiffSpringForceFieldCuda3f_addForce_kernel<<< grid, threads, BSIZE*6*sizeof(float) >>>(nbSpringPerVertex, (const GPUSpring*)springs, (float*)f, (const float*)x, (const float*)v, (float*)dfdx);
}

void StiffSpringForceFieldCuda3f_addExternalForce(unsigned int size, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2, void* dfdx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    StiffSpringForceFieldCuda3f_addExternalForce_kernel<<< grid, threads, BSIZE*6*sizeof(float) >>>(nbSpringPerVertex, (const GPUSpring*)springs, (float*)f1, (const float*)x1, (const float*)v1, (const float*)x2, (const float*)v2, (float*)dfdx);
}

void StiffSpringForceFieldCuda3f_addDForce(unsigned int size, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* dx, const void* x, const void* dfdx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    StiffSpringForceFieldCuda3f_addDForce_kernel<<< grid, threads, BSIZE*6*sizeof(float) >>>(nbSpringPerVertex, (const GPUSpring*)springs, (float*)f, (const float*)dx, (const float*)x, (const float*)dfdx);
}

void StiffSpringForceFieldCuda3f_addExternalDForce(unsigned int size, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* dx1, const void* x1, const void* dx2, const void* x2, const void* dfdx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    StiffSpringForceFieldCuda3f_addExternalDForce_kernel<<< grid, threads, BSIZE*6*sizeof(float) >>>(nbSpringPerVertex, (const GPUSpring*)springs, (float*)f1, (const float*)dx1, (const float*)x1, (const float*)dx2, (const float*)x2, (const float*)dfdx);
}
