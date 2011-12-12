
#define BSIZE 16

typedef struct
{
    int index; ///< 0 if no spring
    Real ks;
} GPUSpring;

typedef struct
{
    Real initpos;
    Real kd;
} GPUSpring2;



__kernel void SpringForceField3t_addForce(
    unsigned int nbSpringPerVertex,
    __global GPUSpring* springs,
    __global Real* f,
    __global const Real* x,
    __global const Real* v
)
{
    __local Real temp[BSIZE*6];
    const int index0 = get_group_id(0)*BSIZE;
    const int index1 = get_local_id(0);

    //! Dynamically allocated shared memory to reorder global memory access


    // First copy x and v inside temp
    const int iext = index0*3+index1;
    temp[index1        ] = x[iext        ];
    temp[index1+  BSIZE] = x[iext+  BSIZE];
    temp[index1+2*BSIZE] = x[iext+2*BSIZE];
    temp[index1+3*BSIZE] = v[iext        ];
    temp[index1+4*BSIZE] = v[iext+  BSIZE];
    temp[index1+5*BSIZE] = v[iext+2*BSIZE];

    barrier(CLK_LOCAL_MEM_FENCE);

    const int index3 = 3*index1;
    Real4 pos1 = (Real4)(temp[index3  ],temp[index3+1],temp[index3+2],0);
    Real4 vel1 = (Real4)(temp[index3  +3*BSIZE],temp[index3+1+3*BSIZE],temp[index3+2+3*BSIZE],0);
    Real4 force = (Real4)(0,0,0,0);


    springs+=((index0*nbSpringPerVertex)<<1)+index1;

    for (int s = 0; s < nbSpringPerVertex; s++)
    {
        GPUSpring spring = *springs;
        --spring.index;
        springs+=BSIZE;
        GPUSpring2 spring2 = *(const GPUSpring2*)springs;
        springs+=BSIZE;
        if (spring.index != -1)
        {

            Real4 u, relativeVelocity;
            int spring_index3 = spring.index*3;

            if (spring.index >= index0 && spring.index < index0+BSIZE)
            {
                // 'local' point
                int i = spring.index - index0;
                u = (Real4)(temp[3*i  ], temp[3*i+1], temp[3*i+2],0);
                relativeVelocity = (Real4)(temp[3*i  +3*BSIZE], temp[3*i+1+3*BSIZE], temp[3*i+2+3*BSIZE],0);
            }
            else
            {
                // general case
                u = (Real4)(x[spring_index3],x[spring_index3+1],x[spring_index3+2],0);
                relativeVelocity = (Real4)(v[spring_index3],v[spring_index3+1],v[spring_index3+2],0);
            }

            u -= pos1;
            relativeVelocity -= vel1;

            Real inverseLength = 1/sqrt(dot(u,u));
            Real d = 1/inverseLength;
            u *= inverseLength;
            Real elongation = d - spring2.initpos;
            Real elongationVelocity = dot(u,relativeVelocity);
            Real forceIntensity = spring.ks*elongation+spring2.kd*elongationVelocity;
            force += u*forceIntensity;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    temp[index3  ] = force.x;
    temp[index3+1] = force.y;
    temp[index3+2] = force.z;

    barrier(CLK_LOCAL_MEM_FENCE);

    f[iext        ] += temp[index1        ];
    f[iext+  BSIZE] += temp[index1+  BSIZE];
    f[iext+2*BSIZE] += temp[index1+2*BSIZE];
}





__kernel void StiffSpringForceField3t_addForce(
    unsigned int nbSpringPerVertex,
    __global GPUSpring* springs,
    __global Real* f,
    __global Real* x,
    __global Real* v,
    __global Real* dfdx)
{
    Real4 u, relativeVelocity,pos1,vel1,force,force2;
    int spring_index3,i,s;
    Real d,inverseLength,elongation,elongationVelocity, forceIntensity;
    GPUSpring spring;
    GPUSpring2 spring2;

    const int index0 = get_group_id(0)*BSIZE;
    const int index1 = get_local_id(0);
    const int index3 = 3*index1;

    //! Dynamically allocated shared memory to reorder global memory access
    __local Real temp[BSIZE*6];

    // First copy x and v inside temp
    const int iext = index0*3+index1;
    temp[index1        ] = x[iext        ];
    temp[index1+  BSIZE] = x[iext+  BSIZE];
    temp[index1+2*BSIZE] = x[iext+2*BSIZE];
    temp[index1+3*BSIZE] = v[iext        ];
    temp[index1+4*BSIZE] = v[iext+  BSIZE];
    temp[index1+5*BSIZE] = v[iext+2*BSIZE];

    barrier(CLK_LOCAL_MEM_FENCE);


    pos1 = (Real4)(temp[index3  ],temp[index3+1],temp[index3+2],0);
    vel1 = (Real4)(temp[index3  +3*BSIZE],temp[index3+1+3*BSIZE],temp[index3+2+3*BSIZE],0);
    springs+=((index0*nbSpringPerVertex)<<1)+index1;

    dfdx+=index0*nbSpringPerVertex+index1;

    force2 = (Real4)(0,0,0,0);

    for (s = 0; s < nbSpringPerVertex; s++)
    {

        spring.ks = springs->ks;
        spring.index = springs->index;
        --spring.index;
        springs+=BSIZE;
        //spring2 = *(const GPUSpring2*)springs;
        spring2.kd = springs->ks;
        long i = springs->index;
        spring2.initpos = *((Real*)&i);
        springs+=BSIZE;
        if ( spring.index > -1)
        {

            int spring_index3 = spring.index*3;

            if (spring.index >= index0 && spring.index < index0+BSIZE)
            {
                // 'local' point
                int j = spring.index - index0;
                u = (Real4)(temp[3*j  ], temp[3*j+1], temp[3*j+2],0);
                relativeVelocity = (Real4)(temp[3*j  +3*BSIZE], temp[3*j+1+3*BSIZE], temp[3*j+2+3*BSIZE],0);
            }
            else
            {
                // general case
                u = (Real4)(x[spring_index3],x[spring_index3+1],x[spring_index3+2],0);
                relativeVelocity = (Real4)(v[spring_index3],v[spring_index3+1],v[spring_index3+2],0);
            }

            u -= pos1;
            relativeVelocity -= vel1;

            d = sqrt(dot(u,u));
            inverseLength =  1.0f/d;
            u *= inverseLength;
            elongation = d - spring2.initpos;

            elongationVelocity = dot(u,relativeVelocity);
            //Real kd = ;
            forceIntensity = spring.ks*elongation+elongationVelocity*spring2.kd;
            force2 += u*forceIntensity;

            *dfdx = forceIntensity*inverseLength;

        }
        dfdx+=BSIZE;


    }
    force=(Real4)force2;

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    temp[index3  ] = force.x;
    temp[index3+1] = force.y;
    temp[index3+2] = force.z;

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);


    f[iext        ] += temp[index1        ];
    f[iext+  BSIZE] += temp[index1+  BSIZE];
    f[iext+2*BSIZE] += temp[index1+2*BSIZE];
}



__kernel void StiffSpringForceField3t_addDForce(
    unsigned int nbSpringPerVertex,
    __global GPUSpring* springs,
    __global Real* f,
    __global Real* dx,
    __global Real* x,
    __global Real* dfdx,
    Real factor
)
{
    int s,i3;
    Real4  dpos1, pos1, dforce, du,u;
    Real tgt,uxux ,uyuy ,uzuz ,uxuy,uxuz,uyuz,fact;

    GPUSpring spring;
    const int index0 = get_group_id(0)*BSIZE;
    const int index1 = get_local_id(0);
    const int index3 = 3*index1;

    //! Dynamically allocated shared memory to reorder global memory access
    __local Real temp[BSIZE*6];

    // First copy x and v inside temp
    const int iext = index0*3+index1;


    // First copy dx and x inside temp
    temp[index1        ] = dx[iext        ];
    temp[index1+  BSIZE] = dx[iext+  BSIZE];
    temp[index1+2*BSIZE] = dx[iext+2*BSIZE];
    temp[index1+3*BSIZE] = x[iext        ];
    temp[index1+4*BSIZE] = x[iext+  BSIZE];
    temp[index1+5*BSIZE] = x[iext+2*BSIZE];

    barrier(CLK_LOCAL_MEM_FENCE);

    dpos1 = (float4)(temp[index3  ],temp[index3+1],temp[index3+2],0);
    pos1 = (float4)(temp[index3  +3*BSIZE],temp[index3+1+3*BSIZE],temp[index3+2+3*BSIZE],0);

    dforce = (float)(0.0f,0.0f,0.0f,0.0f);

    springs+=((index0*nbSpringPerVertex)<<1)+index1;
    dfdx+=(index0*nbSpringPerVertex)+index1;

    for (s = 0; s < nbSpringPerVertex; s++)
    {
        spring.ks = springs->ks;
        spring.index=springs->index;
        --spring.index;
        springs+=BSIZE;
        springs+=BSIZE;
        tgt = *dfdx;
        dfdx+=BSIZE;
        if (spring.index != -1)
        {


            if (spring.index >= index0 && spring.index < index0+BSIZE)
            {
                // 'local' point
                i3 = (spring.index - index0)*3;
                du = (float4)(temp[i3  ], temp[i3+1], temp[i3+2],0);
                u = (float4)(temp[i3  +3*BSIZE], temp[i3+1+3*BSIZE], temp[i3+2+3*BSIZE],0);
            }
            else
            {
                // general case
                i3 = (spring.index)*3;
                du = (float4)(dx[i3],dx[i3+1],dx[i3+2],0);
                u = (float4)(x[i3],x[i3+1],x[i3+2],0);
            }

            du -= dpos1;
            u -= pos1;


            uxux = u.x*u.x;
            uyuy = u.y*u.y;
            uzuz = u.z*u.z;
            uxuy = u.x*u.y;
            uxuz = u.x*u.z;
            uyuz = u.y*u.z;
            fact = (spring.ks-tgt)/(uxux+uyuy+uzuz);
            dforce.x += fact*(uxux*du.x+uxuy*du.y+uxuz*du.z)+tgt*du.x;
            dforce.y += fact*(uxuy*du.x+uyuy*du.y+uyuz*du.z)+tgt*du.y;
            dforce.z += fact*(uxuz*du.x+uyuz*du.y+uzuz*du.z)+tgt*du.z;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    temp[index3  ] = dforce.x*factor;
    temp[index3+1] = dforce.y*factor;
    temp[index3+2] = dforce.z*factor;

    barrier(CLK_LOCAL_MEM_FENCE);

    f[iext        ] += temp[index1        ];
    f[iext+  BSIZE] += temp[index1+  BSIZE];
    f[iext+2*BSIZE] += temp[index1+2*BSIZE];
}

/* INTERACTION FORCES */

__kernel void SpringForceField3t_addExternalForce(
    unsigned int nbSpringPerVertex,
    __global GPUSpring* springs,
    unsigned int offset1,
    __global Real* f1,
    __global const Real* x1,
    __global const Real* v1,
    unsigned int offset2,
    __global const Real* x2,
    __global const Real* v2
)
{
    __local Real temp[BSIZE*6];
    const int index0 = get_group_id(0)*BSIZE;
    const int index1 = get_local_id(0);

    //! Dynamically allocated shared memory to reorder global memory access


    // First copy x and v inside temp
    const int iext = (offset1+index0)*3+index1;
    temp[index1        ] = x1[iext        ];
    temp[index1+  BSIZE] = x1[iext+  BSIZE];
    temp[index1+2*BSIZE] = x1[iext+2*BSIZE];
    temp[index1+3*BSIZE] = v1[iext        ];
    temp[index1+4*BSIZE] = v1[iext+  BSIZE];
    temp[index1+5*BSIZE] = v1[iext+2*BSIZE];

    barrier(CLK_LOCAL_MEM_FENCE);

    const int index3 = 3*index1;
    Real4 pos1 = (Real4)(temp[index3  ],temp[index3+1],temp[index3+2],0);
    Real4 vel1 = (Real4)(temp[index3  +3*BSIZE],temp[index3+1+3*BSIZE],temp[index3+2+3*BSIZE],0);
    Real4 force = (Real4)(0,0,0,0);


    springs+=((index0*nbSpringPerVertex)<<1)+index1;

    for (int s = 0; s < nbSpringPerVertex; s++)
    {
        GPUSpring spring = *springs;
        --spring.index;
        springs+=BSIZE;
        GPUSpring2 spring2 = *(const GPUSpring2*)springs;
        springs+=BSIZE;
        if (spring.index != -1)
        {

            Real4 u, relativeVelocity;
            int spring_index3 = (offset2+spring.index)*3;

            {
                // general case
                u = (Real4)(x2[spring_index3],x2[spring_index3+1],x2[spring_index3+2],0);
                relativeVelocity = (Real4)(v2[spring_index3],v2[spring_index3+1],v2[spring_index3+2],0);
            }

            u -= pos1;
            relativeVelocity -= vel1;

            Real inverseLength = 1/sqrt(dot(u,u));
            Real d = 1/inverseLength;
            u *= inverseLength;
            Real elongation = d - spring2.initpos;
            Real elongationVelocity = dot(u,relativeVelocity);
            Real forceIntensity = spring.ks*elongation+spring2.kd*elongationVelocity;
            force += u*forceIntensity;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    temp[index3  ] = force.x;
    temp[index3+1] = force.y;
    temp[index3+2] = force.z;

    barrier(CLK_LOCAL_MEM_FENCE);

    f1[iext        ] += temp[index1        ];
    f1[iext+  BSIZE] += temp[index1+  BSIZE];
    f1[iext+2*BSIZE] += temp[index1+2*BSIZE];
}





__kernel void StiffSpringForceField3t_addExternalForce(
    unsigned int nbSpringPerVertex,
    __global GPUSpring* springs,
    unsigned int offset1,
    __global Real* f1,
    __global Real* x1,
    __global Real* v1,
    unsigned int offset2,
    __global Real* x2,
    __global Real* v2,
    __global Real* dfdx)
{
    Real4 u, relativeVelocity,pos1,vel1,force,force2;
    int spring_index3,i,s;
    Real d,inverseLength,elongation,elongationVelocity, forceIntensity;
    GPUSpring spring;
    GPUSpring2 spring2;

    const int index0 = get_group_id(0)*BSIZE;
    const int index1 = get_local_id(0);
    const int index3 = 3*index1;

    //! Dynamically allocated shared memory to reorder global memory access
    __local Real temp[BSIZE*6];

    // First copy x and v inside temp
    const int iext = (offset1+index0)*3+index1;
    temp[index1        ] = x1[iext        ];
    temp[index1+  BSIZE] = x1[iext+  BSIZE];
    temp[index1+2*BSIZE] = x1[iext+2*BSIZE];
    temp[index1+3*BSIZE] = v1[iext        ];
    temp[index1+4*BSIZE] = v1[iext+  BSIZE];
    temp[index1+5*BSIZE] = v1[iext+2*BSIZE];

    barrier(CLK_LOCAL_MEM_FENCE);


    pos1 = (Real4)(temp[index3  ],temp[index3+1],temp[index3+2],0);
    vel1 = (Real4)(temp[index3  +3*BSIZE],temp[index3+1+3*BSIZE],temp[index3+2+3*BSIZE],0);
    springs+=((index0*nbSpringPerVertex)<<1)+index1;

    dfdx+=index0*nbSpringPerVertex+index1;

    force2 = (Real4)(0,0,0,0);

    for (s = 0; s < nbSpringPerVertex; s++)
    {

        spring.ks = springs->ks;
        spring.index = springs->index;
        --spring.index;
        springs+=BSIZE;
        //spring2 = *(const GPUSpring2*)springs;
        spring2.kd = springs->ks;
        long i = springs->index;
        spring2.initpos = *((Real*)&i);
        springs+=BSIZE;
        if ( spring.index > -1)
        {

            int spring_index3 = (offset2+spring.index)*3;

            {
                // general case
                u = (Real4)(x2[spring_index3],x2[spring_index3+1],x2[spring_index3+2],0);
                relativeVelocity = (Real4)(v2[spring_index3],v2[spring_index3+1],v2[spring_index3+2],0);
            }

            u -= pos1;
            relativeVelocity -= vel1;

            d = sqrt(dot(u,u));
            inverseLength =  1.0f/d;
            u *= inverseLength;
            elongation = d - spring2.initpos;

            elongationVelocity = dot(u,relativeVelocity);
            //Real kd = ;
            forceIntensity = spring.ks*elongation+elongationVelocity*spring2.kd;
            force2 += u*forceIntensity;

            *dfdx = forceIntensity*inverseLength;

        }
        dfdx+=BSIZE;


    }
    force=(Real4)force2;

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    temp[index3  ] = force.x;
    temp[index3+1] = force.y;
    temp[index3+2] = force.z;

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);


    f1[iext        ] += temp[index1        ];
    f1[iext+  BSIZE] += temp[index1+  BSIZE];
    f1[iext+2*BSIZE] += temp[index1+2*BSIZE];
}



__kernel void StiffSpringForceField3t_addExternalDForce(
    unsigned int nbSpringPerVertex,
    __global GPUSpring* springs,
    unsigned int offset1,
    __global Real* f1,
    __global Real* dx1,
    __global Real* x1,
    unsigned int offset2,
    __global Real* dx2,
    __global Real* x2,
    __global Real* dfdx,
    Real factor
)
{
    int s,i3;
    Real4  dpos1, pos1, dforce, du,u;
    Real tgt,uxux ,uyuy ,uzuz ,uxuy,uxuz,uyuz,fact;

    GPUSpring spring;
    const int index0 = get_group_id(0)*BSIZE;
    const int index1 = get_local_id(0);
    const int index3 = 3*index1;

    //! Dynamically allocated shared memory to reorder global memory access
    __local Real temp[BSIZE*6];

    // First copy x and v inside temp
    const int iext = (offset1+index0)*3+index1;


    // First copy dx and x inside temp
    temp[index1        ] = dx1[iext        ];
    temp[index1+  BSIZE] = dx1[iext+  BSIZE];
    temp[index1+2*BSIZE] = dx1[iext+2*BSIZE];
    temp[index1+3*BSIZE] = x1[iext        ];
    temp[index1+4*BSIZE] = x1[iext+  BSIZE];
    temp[index1+5*BSIZE] = x1[iext+2*BSIZE];

    barrier(CLK_LOCAL_MEM_FENCE);

    dpos1 = (float4)(temp[index3  ],temp[index3+1],temp[index3+2],0);
    pos1 = (float4)(temp[index3  +3*BSIZE],temp[index3+1+3*BSIZE],temp[index3+2+3*BSIZE],0);

    dforce = (float)(0.0f,0.0f,0.0f,0.0f);

    springs+=((index0*nbSpringPerVertex)<<1)+index1;
    dfdx+=(index0*nbSpringPerVertex)+index1;

    for (s = 0; s < nbSpringPerVertex; s++)
    {
        spring.ks = springs->ks;
        spring.index=springs->index;
        --spring.index;
        springs+=BSIZE;
        springs+=BSIZE;
        tgt = *dfdx;
        dfdx+=BSIZE;
        if (spring.index != -1)
        {

            {
                // general case
                i3 = (offset2+spring.index)*3;
                du = (float4)(dx2[i3],dx2[i3+1],dx2[i3+2],0);
                u = (float4)(x2[i3],x2[i3+1],x2[i3+2],0);
            }

            du -= dpos1;
            u -= pos1;


            uxux = u.x*u.x;
            uyuy = u.y*u.y;
            uzuz = u.z*u.z;
            uxuy = u.x*u.y;
            uxuz = u.x*u.z;
            uyuz = u.y*u.z;
            fact = (spring.ks-tgt)/(uxux+uyuy+uzuz);
            dforce.x += fact*(uxux*du.x+uxuy*du.y+uxuz*du.z)+tgt*du.x;
            dforce.y += fact*(uxuy*du.x+uyuy*du.y+uyuz*du.z)+tgt*du.y;
            dforce.z += fact*(uxuz*du.x+uyuz*du.y+uzuz*du.z)+tgt*du.z;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    temp[index3  ] = dforce.x*factor;
    temp[index3+1] = dforce.y*factor;
    temp[index3+2] = dforce.z*factor;

    barrier(CLK_LOCAL_MEM_FENCE);

    f1[iext        ] += temp[index1        ];
    f1[iext+  BSIZE] += temp[index1+  BSIZE];
    f1[iext+2*BSIZE] += temp[index1+2*BSIZE];
}
