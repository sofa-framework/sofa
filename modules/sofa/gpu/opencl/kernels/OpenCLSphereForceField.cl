


__kernel void addForce(
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


__kernel void addDForce(
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




