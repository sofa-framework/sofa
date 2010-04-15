

__kernel void addForce
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


__kernel void addDForce(
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





