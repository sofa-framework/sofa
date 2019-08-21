_DEF_()
_TYPES_()

__kernel void GenericParticleForceField_3f_addForce__METHODNAME()(
    __INPUTS_AddForce__(),
    __global IDATA_TYPE* idata,
    __global Real* f,
    __global Real* x,
    __global Real* v
)
{
    int index0 = get_group_id(0)*BSIZE;
    int index0_3 = index0*3;

    idata += index0;
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

    barrier(CLK_LOCAL_MEM_FENCE);

    temp[index        ] = v[index        ];
    temp[index+  BSIZE] = v[index+  BSIZE];
    temp[index+2*BSIZE] = v[index+2*BSIZE];

    barrier(CLK_LOCAL_MEM_FENCE);

    Real4 vi = (Real4)(temp[index_3  ], temp[index_3+1], temp[index_3+2],0);
    Real4 force = (Real4)(0,0,0,0);

    IDATA_TYPE IDATA_NAME = idata[index];

    __OP_AddForce__()

    idata[index] = IDATA_NAME;

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


__kernel void GenericParticleForceField_3f_addDForce__METHODNAME()(
    __INPUTS_AddDForce__(),
    __global IDATA_TYPE* idata,
    __global Real* df,
    __global Real* dx
)
{
    int index0 = get_group_id(0)*BSIZE;
    int index0_3 = index0*3; //index0*3;

    idata += index0;
    df += index0_3;
    dx += index0_3;

    int index = get_local_id(0);
    int index_3 = index*3;

    //! Dynamically allocated shared memory to reorder global memory access
    __local Real temp[BSIZE*3];

    temp[index        ] = dx[index        ];
    temp[index+  BSIZE] = dx[index+  BSIZE];
    temp[index+2*BSIZE] = dx[index+2*BSIZE];

    barrier(CLK_LOCAL_MEM_FENCE);
    Real4 dforce = (Real4)(0,0,0,0);
    Real4 dxi = (float4)(temp[index_3  ], temp[index_3+1], temp[index_3+2], 0);

    IDATA_TYPE IDATA_NAME = idata[index];

    __OP_AddDForce__();

    barrier(CLK_LOCAL_MEM_FENCE);

    temp[index        ] = df[index        ];
    temp[index+  BSIZE] = df[index+  BSIZE];
    temp[index+2*BSIZE] = df[index+2*BSIZE];

    barrier(CLK_LOCAL_MEM_FENCE);

    temp[index_3+0] += dforce.x;
    temp[index_3+1] += dforce.y;
    temp[index_3+2] += dforce.z;

    barrier(CLK_LOCAL_MEM_FENCE);

    df[index        ] = temp[index        ];
    df[index+  BSIZE] = temp[index+  BSIZE];
    df[index+2*BSIZE] = temp[index+2*BSIZE];
}
