//obsolete

#define BSIZE 32

__kernel void PlaneForceField_3f_addForce
(
    Real4 planeNormal,
    Real4 planeData,
    __global Real* penetration,
    __global Real* f,
    __global Real* x,
    __global Real* v
)
{
    int index = get_global_id(0);
    int index3 = index*3;

    Real4 xi = (Real4)(x[index3],x[index3+1],x[index3+2],0);
    Real4 vi = (Real4)(v[index3],v[index3+1],v[index3+2],0);
    Real4 force = (Real4)(0,0,0,0);

    Real d = dot(xi,planeNormal) -planeData.s0;	// dot(xi,planeNormal)-planeD

    penetration[index] = d;

    if (d<0)
    {
        Real forceIntensity = -planeData.s1*d;		// -planeStiffness*d;
        Real dampingIntensity = -planeData.s2*d;	// -planeDamping*d;
        force = planeNormal*forceIntensity- vi*dampingIntensity;

        f[index3]+=force.s0;
        f[index3+1]+=force.s1;
        f[index3+2]+=force.s2;
    }
}


__kernel void PlaneForceField_3f_addForce_v2(
    Real4 planeNormal,
    Real4 planeData,
    __global Real* penetration,
    __global Real* f,
    __global Real* x,
    __global Real* v
)
{
    int index0 = get_group_id(0)*BSIZE;
    int index0_3 = index0*3;

    penetration += index0;
    f += index0_3;
    x += index0_3;
    v += index0_3;

    int index = get_local_id(0);
    int index_3 = index*3;

    //! Dynamically allocated shared memory to reorder global memory access
    __local Real temp[BSIZE*3];

    temp[index        ] = x[index        ];
    temp[index+  BSIZE] = x[index+  BSIZE];
    temp[index+2*BSIZE] = x[index+2*BSIZE];

    barrier(CLK_LOCAL_MEM_FENCE);

    Real4 xi = (Real4)(temp[index_3  ], temp[index_3+1], temp[index_3+2],0);
    Real d = dot(xi,planeNormal)-planeData.s0;	//dot(xi,planeNormal)-planeD

    penetration[index] = d;

    barrier(CLK_LOCAL_MEM_FENCE);

    temp[index        ] = v[index        ];
    temp[index+  BSIZE] = v[index+  BSIZE];
    temp[index+2*BSIZE] = v[index+2*BSIZE];

    barrier(CLK_LOCAL_MEM_FENCE);

    Real4 vi = (Real4)(temp[index_3  ], temp[index_3+1], temp[index_3+2],0);
    Real4 force = (Real4)(0,0,0,0);

    if (d<0)
    {
        Real forceIntensity =  -planeData.s1*d;		// -planeStiffness*d;
        Real dampingIntensity = -planeData.s2*d;	// -planeDamping*d;
        force = planeNormal*forceIntensity - vi*dampingIntensity;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    temp[index        ] = f[index        ];
    temp[index+  BSIZE] = f[index+  BSIZE];
    temp[index+2*BSIZE] = f[index+2*BSIZE];

    barrier(CLK_LOCAL_MEM_FENCE);

    temp[index_3+0] += force.x;
    temp[index_3+1] += force.y;
    temp[index_3+2] += force.z;

    barrier(CLK_LOCAL_MEM_FENCE);

    f[index        ] = temp[index        ];
    f[index+  BSIZE] = temp[index+  BSIZE];
    f[index+2*BSIZE] = temp[index+2*BSIZE];
}



__kernel void PlaneForceField_3f_addDForce(
    Real4 plane,
    Real planeStiffness,
    __global Real* penetration,
    __global Real* df,
    __global Real* dx
)
{
    int index = get_global_id(0);
    int index3 = index*3;
    Real4 dxi = (Real4)(dx[index3],dx[index3+1],dx[index3+2],0);
    Real d = penetration[index];
    Real4 dforce = (Real4)(0,0,0,0);

    if (d<0)
    {
        dforce = plane * (-planeStiffness * dot(dxi,plane));

        df[index3] += dforce.s0;
        df[index3+1] += dforce.s1;
        df[index3+2] += dforce.s2;
    }
}





