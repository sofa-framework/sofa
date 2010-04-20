


__kernel void UniformMass_addForce(
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



__kernel void UniformMass_addMDx(
    Real mass,
    __global Real* res,
    __global Real* dx
)
{
    int index = get_global_id(0);

    res[index] += dx[index] * mass;

}
