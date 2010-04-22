#define BSIZE 32


__kernel void SphereForceField_3f_addForce(
    Real4 sphereCenter,
    Real4 sphereData,
    __global Real4* penetration,
    __global Real* f,
    __global Real* x,
    __global Real* v
)
{
    int index = get_global_id(0);
    int index3 = index*3;

    Real4 temp = (Real4)(x[index3],x[index3+1],x[index3+2],0);
    Real4 dp = temp - sphereCenter;
    Real d2 = dot(dp,dp);

    Real4 vi = (Real4)(v[index3],v[index3+1],v[index3+2],0);
    Real4 force = (Real4)(0,0,0,0);

    if (d2 < sphereData.x*sphereData.x)	// (d2<sphere.r*sphere.r)
    {
        Real length = sqrt(d2);
        Real inverseLength = 1/length;
        dp.x *= inverseLength;								//dp.x *= ..
        dp.y *= inverseLength;								//dp.y *= ..
        dp.z *= inverseLength;								//dp.z *= ..
        d2 = -sphereData.x*inverseLength;					//d2 = -sphere.r*inverseLength;
        Real d = sphereData.x - length;

        Real forceIntensity = sphereData.y*d;			//sphere.stiffness*d;
        Real dampingIntensity = sphereData.z*d;		//sphere.damping*d;
        force = dp*forceIntensity - vi*dampingIntensity;
    }
    penetration[index] = (Real4)(dp.x,dp.y,dp.z,d2);

    f[index3] += force.x;		//force.x
    f[index3+1] += force.y;	//force.y
    f[index3+2] += force.z;	//force.z
}


__kernel void SphereForceField_3f_addDForce(
    Real4 sphereCenter,
    Real sphereStiffness,
    __global Real4* penetration,
    __global Real* df,
    __global Real* dx
)
{
    int index = get_global_id(0); //umul24(blockIdx.x,BSIZE)+threadIdx.x;
    int index3 = index*3;

    Real4 dxi = (Real4)(dx[index3],dx[index3+1],dx[index3+2],0);
    Real4 d = penetration[index];

    Real4 dforce = (Real4)(0,0,0,0);

    if (d.s3<0)
    {
        Real4 dp = (Real4)(d.x, d.y, d.z,0);
        dforce = sphereStiffness*(dot(dxi,dp)*d.w * dp - (1+d.w) * dxi);  //sphere.stiffness*...
    }

    df[index3] += dforce.x;
    df[index3+1] += dforce.y;
    df[index3+2] += dforce.z;

}

__kernel void SphereForceField_3f_addDForce_v2(
    Real4 sphereCenter,
    Real sphereStiffness,
    __global Real4* penetration,
    __global Real* df,
    __global Real* dx
)
{
    int index0 = get_group_id(0)*BSIZE;
    int index0_3 = index0*3; //index0*3;

    penetration += index0;
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

    Real4 dxi = (float4)(temp[index_3  ], temp[index_3+1], temp[index_3+2], 0);
    Real4 d = penetration[index];

    Real4 dforce = (Real4)(0,0,0,0);

    if (d.w<0)
    {
        float4 dp = (float4)(d.x, d.y, d.z, 0);
        dforce = sphereStiffness*(dot(dxi,dp)*d.w * dp - (1+d.w) * dxi);
    }

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
__kernel void SphereForceField_3f_addForce_v2(
    Real4 sphereCenter,
    Real4 sphereData,
    __global Real4* penetration,
    __global Real* f,
    __global Real* x,
    __global Real* v
)
{
    int index0 = get_group_id(0)*BSIZE;
    int index0_3 = index0*3; //index0*3;

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

    Real4 dp = ((Real4)(temp[index_3  ], temp[index_3+1], temp[index_3+2],0)) - sphereCenter;
    Real d2 = dot(dp,dp);

    barrier(CLK_LOCAL_MEM_FENCE);

    temp[index        ] = v[index        ];
    temp[index+  BSIZE] = v[index+  BSIZE];
    temp[index+2*BSIZE] = v[index+2*BSIZE];

    barrier(CLK_LOCAL_MEM_FENCE);

    Real4 vi = (Real4)(temp[index_3  ], temp[index_3+1], temp[index_3+2],0);
    Real4 force = (Real4)(0,0,0,0);

    if (d2 <  sphereData.x* sphereData.x)	//(d2<sphere.r*sphere.r)
    {
        Real length = sqrt(d2);
        Real inverseLength = 1/length;
        dp.x *= inverseLength;
        dp.y *= inverseLength;
        dp.z *= inverseLength;
        d2 = -sphereData.x*inverseLength;	//d2 = -sphere.r*inverseLength;
        Real d = sphereData.x - 1/inverseLength;

        Real forceIntensity = sphereData.y*d;		//sphere.stiffness*d;
        Real dampingIntensity = sphereData.z*d;	//sphere.damping*d;
        force = dp*forceIntensity - vi*dampingIntensity;
    }
    penetration[index] = (Real4)(dp.x,dp.y,dp.z,d2);

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
