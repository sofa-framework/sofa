

__kernel void Vec1t_vMEq(
    __global Real* res,
    Real f
)
{
    int index = get_global_id(0);


    res[index] *= f;


}


__kernel void Vec1t_vMEq_v2(
    __global Real* res,
    Real f
)
{
    int index = get_global_id(0)*4;

    res[index] *= f;
    res[index+1] *= f;
    res[index+2] *= f;
    res[index+3] *= f;
}



__kernel void Vec1t_vOp(
    __global Real* res,
    __global const Real* a,
    __global const Real* b,
    Real f
)
{
    int index = get_global_id(0);

    res[index] = a[index] + b[index] * f;
}

__kernel void Vec1t_vOp_v2(
    __global Real* res,
    __global const Real* a,
    __global const Real* b,
    Real f
)
{
    int index = get_global_id(0)*4;

    res[index] = a[index] + b[index] * f;
    res[index+1] = a[index+1] + b[index+1] * f;
    res[index+2] = a[index+2] + b[index+2] * f;
    res[index+3] = a[index+3] + b[index+3] * f;
}


__kernel void Vec1t_vEqBF(
    __global Real* res,
    __global const Real* b,
    Real f
)
{
    int index = get_global_id(0);

    res[index] = b[index] * f;
}

__kernel void Vec1t_vPEq(
    __global Real* res,
    __global const Real* b
)
{
    int index = get_global_id(0);

    res[index] += b[index];
}

__kernel void Vec1t_vPEqBF(
    __global Real* res,
    __global const Real* b,
    Real f
)
{
    int index =  get_global_id(0);

    res[index] += b[index] * f;
}


#define RED_SIZE 512
#define RED_ITER 9


__kernel void Vec1t_vDot(
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



__kernel void Vec1t_vAdd(
    __global Real* res,
    __global const Real* a,
    __global const Real* b
)
{
    int index = get_global_id(0);

    res[index] = a[index] + b[index];
}


__kernel void Vec1t_vPEqBF2(
    __global Real* res1,
    __global const Real* b1,
    Real f1,
    __global Real* res2,
    __global const Real* b2,
    Real f2)
{
    int index = get_global_id(0);

    res1[index] += b1[index] * f1;
    res2[index] += b2[index] * f2;
}


__kernel void Vec1t_vPEqBF2_v2(
    __global Real* res1,
    __global const Real* b1,
    Real f1,
    __global Real* res2,
    __global const Real* b2,
    Real f2)
{
    int index = get_global_id(0)*4;

    res1[index] += b1[index] * f1;
    res2[index] += b2[index] * f2;

    res1[index+1] += b1[index+1] * f1;
    res2[index+1] += b2[index+1] * f2;

    res1[index+2] += b1[index+2] * f1;
    res2[index+2] += b2[index+2] * f2;

    res1[index+3] += b1[index+3] * f1;
    res2[index+3] += b2[index+3] * f2;
}

__kernel void Vec1t_vIntegrate(
    __global const Real* a,
    __global Real* v,
    __global Real* x,
    Real f_v_v,
    Real f_v_a,
    Real f_x_x,
    Real f_x_v
)
{
    int index = get_global_id(0);

    Real vi = v[index]*f_v_v + a[index] * f_v_a;
    v[index] = vi;
    x[index] = x[index]*f_x_x + vi * f_x_v;

}

__kernel void Vec1t_vClear(
    __global Real* res
)
{
    int index = get_global_id(0)*4;

    res[index] = 0.0;
    res[index+1] = 0.0;
    res[index+2] = 0.0;
    res[index+3] = 0.0;
}










#undef RED_SIZE
#undef RED_ITER

