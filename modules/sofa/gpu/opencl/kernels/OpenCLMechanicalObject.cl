//#define BSIZE 32

__kernel void MechanicalObject_Vec3t_vMEq(
    __global Real* res,
    Real f
)
{
    int index = get_group_id(0)*BSIZE*3 + get_local_id(0);
    //if (index < size)
    {
        res[index] *= f;
        index += BSIZE;
        res[index] *= f;
        index += BSIZE;
        res[index] *= f;
    }
}


__kernel void MechanicalObject_Vec3t_vOp(
    __global Real* res,
    __global const Real* a,
    __global const Real* b,
    Real f
)
{
    int index = get_group_id(0)*BSIZE*3 + get_local_id(0);
    //if (index < size)
    {
        res[index] = a[index] + b[index] * f;
        index += BSIZE;
        res[index] = a[index] + b[index] * f;
        index += BSIZE;
        res[index] = a[index] + b[index] * f;
    }

}

__kernel void  MechanicalObject_Vec3t_vEqBF(
    __global Real* res,
    __global const Real* b,
    Real f
)
{
    int index = get_group_id(0)*BSIZE*3 + get_local_id(0);
    //if (index < size)
    {
        res[index] = b[index] * f;
        index += BSIZE;
        res[index] = b[index] * f;
        index += BSIZE;
        res[index] = b[index] * f;
    }
}

__kernel void MechanicalObject_Vec3t_vPEq(
    __global Real* res,
    __global const Real* a
)
{
    int index = get_group_id(0)*BSIZE*3 + get_local_id(0);
    //if (index < size)
    {
        res[index] += a[index];
        index += BSIZE;
        res[index] += a[index];
        index += BSIZE;
        res[index] += a[index];
    }
}

__kernel void MechanicalObject_Vec3t_vPEqBF(
    __global Real* res,
    __global const Real* b,
    Real f
)
{
    int index = get_group_id(0)*BSIZE*3 + get_local_id(0);

    //if (index < size)
    {
        res[index] += b[index] * f;
        index += BSIZE;
        res[index] += b[index] * f;
        index += BSIZE;
        res[index] += b[index] * f;
    }
}


#define RED_SIZE 512
#define RED_ITER 9


__kernel void MechanicalObject_Vec1t_vDot(
    int size,
    __global Real* res,
    __global const Real4* a,
    __global const Real4* b

)
{
    __local Real l[RED_SIZE];
    float tmp;
    int i;
    int NumBlocks = size/4;
    int NumBlocksR = size%4;

    int g_id =  get_global_id(0);
    int l_id = get_local_id(0);
    int l_id2 =l_id*2;
    int l_id2Plus1 = l_id2+1;

    if(g_id<(NumBlocks)) l[l_id]=dot(a[g_id],b[g_id]);
    else if(g_id>(NumBlocks))l[l_id]=0;
    else if(NumBlocksR==0)l[l_id]=0;
    else
    {
        //	l[l_id]=0;
        l[l_id]=a[g_id].s0 * b[g_id].s0;
        if(NumBlocksR>1)l[l_id]+=a[g_id].s1 * b[g_id].s1;
        if(NumBlocksR>2)l[l_id]+=a[g_id].s2 * b[g_id].s2;
    }

    barrier(CLK_LOCAL_MEM_FENCE);


    int iter = RED_SIZE;

    do
    {
        iter /= 2;
        if(l_id<iter)l[l_id] += l[l_id+iter];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    while(iter!=1);

    if(l_id==0)res[get_group_id(0)]=l[l_id];

}



__kernel void MechanicalObject_Vec3t_vAdd(
    __global Real* res,
    __global const Real* a,
    __global const Real* b
)
{
    int index = get_group_id(0)*BSIZE*3 + get_local_id(0);
    //if (index < size)
    {
        res[index] = a[index] + b[index];
        index += BSIZE;
        res[index] = a[index] + b[index];
        index += BSIZE;
        res[index] = a[index] + b[index];
    }
}


__kernel void MechanicalObject_Vec3t_vPEqBF2(
    __global Real* res1,
    __global const Real* b1,
    Real f1,
    __global Real* res2,
    __global const Real* b2,
    Real f2)
{
    int index = get_group_id(0)*BSIZE*3 + get_local_id(0);
    //if (index < size)
    {
        res1[index] += b1[index] * f1;
        res2[index] += b2[index] * f2;
        index += BSIZE;
        res1[index] += b1[index] * f1;
        res2[index] += b2[index] * f2;
        index += BSIZE;
        res1[index] += b1[index] * f1;
        res2[index] += b2[index] * f2;
    }
}

__kernel void MechanicalObject_Vec3t_vIntegrate(
    __global const Real* a,
    __global Real* v,
    __global Real* x,
    Real f_v_v,
    Real f_v_a,
    Real f_x_x,
    Real f_x_v
)
{
    int index = get_group_id(0)*BSIZE*3 + get_local_id(0);
    //if (index < size)
    {
        Real vi;
        vi = v[index]*f_v_v + a[index] * f_v_a;
        v[index] = vi;
        x[index] = x[index]*f_x_x + vi * f_x_v;
        index += BSIZE;
        vi = v[index]*f_v_v + a[index] * f_v_a;
        v[index] = vi;
        x[index] = x[index]*f_x_x + vi * f_x_v;
        index += BSIZE;
        vi = v[index]*f_v_v + a[index] * f_v_a;
        v[index] = vi;
        x[index] = x[index]*f_x_x + vi * f_x_v;
    }

}

__kernel void MechanicalObject_Vec3t_vClear(
    __global Real* res
)
{
    int index = get_group_id(0)*BSIZE*3 + get_local_id(0);

    res[index] = 0.0;
    index += BSIZE;
    res[index] = 0.0;
    index += BSIZE;
    res[index] = 0.0;
}

__kernel void MechanicalObject_Vec3t_vOp2(
    __global Real* res1,
    __global Real* a1,
    __global Real* b1,
    Real f1,
    __global Real* res2,
    __global Real* a2,
    __global Real* b2,
    Real f2
)
{
    int index = get_group_id(0)*BSIZE*3 + get_local_id(0);
    //if (index < size)
    {
        res1[index] = a1[index] + b1[index] * f1;
        res2[index] = a2[index] + b2[index] * f2;
        index += BSIZE;
        res1[index] = a1[index] + b1[index] * f1;
        res2[index] = a2[index] + b2[index] * f2;
        index += BSIZE;
        res1[index] = a1[index] + b1[index] * f1;
        res2[index] = a2[index] + b2[index] * f2;
    }
}


__kernel void MechanicalObject_Vec3t_vPEq4BF2(
    __global Real* res1,
    __global Real* b11,
    Real f11,
    __global Real* b12,
    Real f12,
    __global Real* b13,
    Real f13,
    __global Real* b14,
    Real f14,
    __global Real* res2,
    __global Real* b21,
    Real f21,
    __global Real* b22,
    Real f22,
    __global Real* b23,
    Real f23,
    __global Real* b24,
    Real f24
)
{
    int index = get_group_id(0)*BSIZE*3 + get_local_id(0);
    //if (index < size)
    {
        Real r1,r2;
        r1 = res1[index];
        r2 = res2[index];
        r1 += b11[index] * f11;
        r2 += b21[index] * f21;
        r1 += b12[index] * f12;
        r2 += b22[index] * f22;
        r1 += b13[index] * f13;
        r2 += b23[index] * f23;
        r1 += b14[index] * f14;
        r2 += b24[index] * f24;
        res1[index] = r1;
        res2[index] = r2;
        index += BSIZE;
        r1 = res1[index];
        r2 = res2[index];
        r1 += b11[index] * f11;
        r2 += b21[index] * f21;
        r1 += b12[index] * f12;
        r2 += b22[index] * f22;
        r1 += b13[index] * f13;
        r2 += b23[index] * f23;
        r1 += b14[index] * f14;
        r2 += b24[index] * f24;
        res1[index] = r1;
        res2[index] = r2;
        index += BSIZE;
        r1 = res1[index];
        r2 = res2[index];
        r1 += b11[index] * f11;
        r2 += b21[index] * f21;
        r1 += b12[index] * f12;
        r2 += b22[index] * f22;
        r1 += b13[index] * f13;
        r2 += b23[index] * f23;
        r1 += b14[index] * f14;
        r2 += b24[index] * f24;
        res1[index] = r1;
        res2[index] = r2;
    }
}





#undef RED_SIZE
#undef RED_ITER

