//#define BSIZE 32


__kernel void UniformMass_addForce(
    int size,
    Real mg_x,
    Real mg_y,
    Real mg_z,
    __global Real* f
)
{
    int index = 3*get_global_id(0);

    f[index] += mg_x;
    f[index+1] += mg_y;
    f[index+3] += mg_z;
}


__kernel void UniformMass_addForce_v2(
    int size,
    Real mg_x,
    Real mg_y,
    Real mg_z,
    __global Real* f
)
{

    f += get_group_id(0)*BSIZE*3;
    int index = get_local_id(0);
    __local Real temp[BSIZE*3];
    temp[index] = f[index];
    temp[index+BSIZE] = f[index+BSIZE];
    temp[index+2*BSIZE] = f[index+2*BSIZE];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_global_id(0) < size)
    {
        int index3 = 3*index;
        temp[index3+0] += mg_x;
        temp[index3+1] += mg_y;
        temp[index3+2] += mg_z;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    f[index] = temp[index];
    f[index+BSIZE] = temp[index+BSIZE];
    f[index+2*BSIZE] = temp[index+2*BSIZE];
}



__kernel void UniformMass_addMDx(
    Real mass,
    __global Real* res,
    __global Real* dx
)
{
    int index = get_group_id(0)*BSIZE*3 + get_local_id(0);

    res[index] += dx[index] * mass;
    index+=BSIZE;
    res[index] += dx[index] * mass;
    index+=BSIZE;
    res[index] += dx[index] * mass;
}


__kernel void UniformMass_accFromF(
    Real inv_mass,
    __global Real* a,
    __global Real* f)
{
    int index = get_global_id(0);

    a[index] = f[index] * inv_mass;
}
