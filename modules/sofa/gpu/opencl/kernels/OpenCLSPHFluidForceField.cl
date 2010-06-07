

typedef struct
{
    Real h;			///< particles radius
    Real h2;		///< particles radius squared
    Real stiffness; ///< pressure stiffness
    Real mass;		///< particles mass
    Real mass2;		///< particles mass squared
    Real density0;	///< 1000 kg/m3 for water
    Real viscosity;
    Real surfaceTension;

    // Precomputed constants for smoothing kernels
    Real CWd;			///< =	constWd(h)
    Real CgradWd;		///< =	constGradWd(h)
    Real CgradWp;		///< =	constGradWp(h)
    Real ClaplacianWv;	///< =	constLaplacianWv(h)
    Real CgradWc;		///< =	constGradWc(h)
    Real ClaplacianWc;	///< =	constLaplacianWc(h)
} GPUSPHFluid;

typedef Real4 Deriv;

#define R_PI 3.141592654
#define BSIZE 32


//////////////////////////////////////////////////

Real constWd(Real h)
{
    return (Real)(315 / (64*R_PI*h*h*h));
}

Real Wd(Real r_h, Real C)
{
    Real a = (1-r_h*r_h);
    return C*a*a*a;
}
Real Wd2(Real r2_h2, Real C)
{
    Real a = (1-r2_h2);
    return C*a*a*a;
}
Real Wd2_1(Real r2_h2)
{
    Real a = (1-r2_h2);
    return a*a*a;
}

// grad W = d(W)/dr Ur			in spherical coordinates, with Ur = D/|D| = D/r
//		= d( C(1-r2/h2)3 )/dr D/r
//		= d( C/h6 (h2-r2)3 )/dr D/r
//		= d( C/h6 (h2-r2)(h4+r4-2h2r2) )/dr D/r
//		= ( C/h6 (h2-r2)(4r3-4h2r) + (-2r)(h4+r4-2h2r2) ) D/r
//		= C/h6 ( 4h2r3-4h4r-4r5+4h2r3 -2h4r -2r5 +4h2r3 ) D/r
//		= C/h6 ( -6r5 +12h2r3 -6h4r ) D/r
//		= -6C/h6 ( r4 -2h2r2 +h4 ) D
//		= -6C/h6 ( h2 - r2 )2 D
//		= -6C/h2 ( 1 - r2/h2 )2 D
Real constGradWd(Real h)
{
    return -6*constWd(h)/(h*h);
}
Deriv gradWd(const Deriv d, Real r_h, Real C)
{
    Real a = (1-r_h*r_h);
    return d*(C*a*a);
}
Deriv gradWd2(const Deriv d, Real r2_h2, Real C)
{
    Real a = (1-r2_h2);
    return d*(C*a*a);
}

// laplacian(W) = d(W)/dx2 + d(W)/dy2 + d(W)/dz2
//		= 1/r d2(rW)/dr2			in spherical coordinate, as f only depends on r
//		= C/r d2(r(1-r2/h2)3)/dr2
//		= C/rh6 d2(r(h2-r2)3)/dr2
//		= C/rh6 d2(r(h2-r2)(h4-2h2r2+r4))/dr2
//		= C/rh6 d2(r(h6-3h4r2+3h2r4-r6))/dr2
//		= C/rh6 d2(h6r-3h4r3+3h2r5-r7)/dr2
//		= C/rh6 d(h6-9h4r2+15h2r4-7r6)/dr
//		= C/rh6 (-18h4r+60h2r3-42r5)
//		= C/h6 (-18h4+60h2r2-42r4)
//		= 6C/h2 (-3+10r2/h2-7r4/h4)
//		= CL (-3+10r2/h2-7r4/h4)
Real constLaplacianWd(Real h)
{
    return 6*constWd(h)/(h*h);
}
Real laplacianWd(Real r_h, Real C)
{
    Real r2_h2 = r_h*r_h;
    return C*(-3+10*r2_h2-7*r2_h2*r2_h2);
}
Real laplacianWd2(Real r2_h2, Real C)
{
    return C*(-3+10*r2_h2-7*r2_h2*r2_h2);
}

/// Pressure Smoothing Kernel: W = 15 / pih6 (h - r)3 = 15 / pih3 (1 - r/h)3
Real constWp(Real h)
{
    return (Real)(15 / (R_PI*h*h*h));
}
Real Wp(Real r_h, Real C)
{
    Real a = (1-r_h);
    return C*a*a*a;
}

// grad W = d(W)/dr Ur			in spherical coordinates, with Ur = D/|D| = D/r
//		= d( C(1-r/h)3 )/dr D/r
//		= d( C/h3 (h-r)3 )/dr D/r
//		= d( C/h6 (h-r)(h2+r2-2hr) )/dr D/r
//		= C/h6 ( (h-r)(2r-2h) -(h2+r2-2hr) ) D/r
//		= C/h6 ( -2r2+4hr-2h2 -r2+2hr-h2 ) D/r
//		= C/h6 ( -2r2+4hr-2h2 -r2+2hr-h2 ) D/r
//		= C/h6 ( -3r2+6hr-3h2 ) D/r
//		= 3C/h4 ( -r2/h2+2r/h-1 ) D/r
//		= -3C/h4 ( 1-r/h )2 D/r
Real constGradWp(Real h)
{
    return (-3*constWp(h)) / (h*h*h*h);
}
Deriv gradWp(const Deriv d, Real r_h, Real C)
{
    Real a = (1-r_h);
    return d * (C*a*a);
}

//Real laplacianWp(Real r_h, Real C);

/// Viscosity Smoothing Kernel: W = 15/(2pih3) (-r3/2h3 + r2/h2 + h/2r - 1)
Real constWv(Real h)
{
    return (Real)(15/(2*R_PI*h*h*h));
}
Real Wv(Real r_h, Real C)
{
    Real r2_h2 = r_h*r_h;
    Real r3_h3 = r2_h2*r_h;
    return C*(-0.5f*r3_h3 + r2_h2 + 0.5f/r_h - 1);
}
Real Wv2(Real r2_h2, Real r_h, Real C)
{
    Real r3_h3 = r2_h2*r_h;
    return C*(-0.5f*r3_h3 + r2_h2 + 0.5f/r_h - 1);
}

// grad W = d(W)/dr Ur			in spherical coordinates, with Ur = D/|D| = D/r
//		= d( C(-r3/2h3 + r2/h2 + h/2r - 1) )/dr D/r
//		= C(-3r2/2h3 + 2r/h2 - h/2r2) D/r
//		= C(-3r/2h3 + 2/h2 - h/2r3) D
//		= C/2h2 (-3r/h + 4 - h3/r3) D

Real constGradWv(Real h)
{
    return constWv(h)/(2*h*h);
}
Deriv gradWv(const Deriv d, Real r_h, Real C)
{
    Real r3_h3 = r_h*r_h*r_h;
    return d * (C*(-3*r_h + 4 - 1/r3_h3));
}

Deriv gradWv2(const Deriv d, Real r2_h2, Real r_h, Real C)
{
    Real r3_h3 = r2_h2*r_h;
    return d * (C*(-3*r_h + 4 - 1/r3_h3));
}

// laplacian(W) = d(W)/dx2 + d(W)/dy2 + d(W)/dz2
//		= 1/r d2(rW)/dr2			in spherical coordinate, as f only depends on r
//		= C/r d2(r(-r3/2h3 + r2/h2 + h/2r - 1))/dr2
//		= C/r d2(-r4/2h3 + r3/h2 + h/2 - r)/dr2
//		= C/r d(-4r3/2h3 + 3r2/h2 - 1)/dr
//		= C/r (-6r2/h3 + 6r/h2)
//		= C (-6r/h3 + 6/h2)
//		= 6C/h2 (1 - r/h)

// laplacian(W) = d(W)/dx2 + d(W)/dy2 + d(W)/dz2
//		= 1/r2 d(r2 d(W)/dr)/dr			in spherical coordinate, as f only depends on r
//		= C/r2 d(r2 d(-r3/2h3 + r2/h2 + h/2r - 1)/dr)/dr
//		= C/r2 d(r2 (-3r2/2h3 + 2r/h2 - h/2r2))/dr
//		= C/r2 d(-3r4/2h3 + 2r3/h2 - h/2))/dr
//		= C/r2 (-6r3/h3 + 6r2/h2)
//		= 6C/h2 (1 -r/h)

inline Real constLaplacianWv(Real h)
{
    return 6*constWv(h)/(h*h);
    //return 75/(R_PI*h*h*h*h*h);
}

inline Real laplacianWv(Real r_h, Real C)
{
    return C*(1-r_h);
}

/// Color Smoothing Kernel: same as Density
inline Real constWc(Real h)
{
    return (Real)(315 / (64*R_PI*h*h*h));
}
inline Real Wc(Real r_h, Real C)
{
    Real a = (1-r_h*r_h);
    return C*a*a*a;
}
inline Real constGradWc(Real h)
{
    return -6*constWc(h)/(h*h);
}
inline Deriv gradWc(const Deriv d, Real r_h, Real C)
{
    Real a = (1-r_h*r_h);
    return d*(C*a*a);
}
inline Real constLaplacianWc(Real h)
{
    return 6*constWc(h)/(h*h);
}
inline Real laplacianWc(Real r_h, Real C)
{
    Real r2_h2 = r_h*r_h;
    return C*(-3+10*r2_h2-7*r2_h2*r2_h2);
}

//////////////////////////////////////

inline Real SPHFluidInitDensity(Real4 x1, Real density, GPUSPHFluid* params)
{
    return 0; //params.mass * params.CWd; //SPH<Real>::Wd(0,params.CWd);
}

inline Real SPHFluidCalcDensity(Real4 x1, Real4 x2, Real density, GPUSPHFluid* params)
{
    Real4 n = x2-x1;
//	Real d2;
    Real d2 = n.x*n.x+n.y*n.y+n.z*n.z;
    if (d2 < params->h2)
    {
        //Real r_h = rsqrtf(params->h2/d2);
        //Real r_h = sqrtf(d2/params->h2);
        Real r2_h2 = (d2/params->h2);
        //Real inv_d = rsqrtf(d2);
        //n *= inv_d;
        //Real d = d2*inv_d;
        //Real d = params.mass * SPH<Real>::Wd2(r2_h2,params->CWd);
        Real d = Wd2_1(r2_h2);
        density += d;
    }
    return density;
}

inline Real SPHFluidFinalDensity(Real4 x1, Real density, GPUSPHFluid* params)
{
    return (1+density) * params->CWd * params->mass; //SPH<Real>::Wd(0,params.CWd);
}



__kernel void SPHFluidForceField_computeDensity(
    int size,
    __global int *cells,
    __global int *cellGhost,
    GPUSPHFluid params,
    __global Real4* pos4,
    __global Real* x
)
{
    __local int range_x, range_y;
    __local int ghost;
    __local Real temp_x[BSIZE*3];

    float test0=0,test1=0,test2=0,test3=0;

    int tx3 = get_local_id(0)*3;

    //for (int cell = group_id; cell < size; cell += num_groups)
    for (int cell = get_group_id(0); cell < size; cell += get_num_groups(0))
    {
        if (!get_local_id(0))
        {
            //range = *(const int2*)(cells+cell);
            range_x = cells[cell];
            range_y = cells[cell+1]; // & ~(1U<<31);
            range_y &= ~(1U<<31);
            ghost = cellGhost[cell];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (range_x <= 0) continue; // no actual particle in this cell

        for (int px0 = range_x; px0 < ghost; px0 += BSIZE)
        {
            int px = px0 +get_local_id(0);
            Real4 xi;
            Real density;
            int index;

            Real xi_x=0;

            if (px < range_y)
            {
                index = cells[px];
                xi = (Real4)(x[index*3],x[index*3+1],x[index*3+2],0);
                temp_x[tx3  ]=xi.x;
                temp_x[tx3+1]=xi.y;
                temp_x[tx3+2]=xi.z;
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            if (px < ghost)
            {
                // actual particle -> compute interactions
                density = SPHFluidInitDensity(xi, density, &params);

                int np = min(range_y-px0,BSIZE);
                for (int i=0; i < np; ++i)
                {
                    if (i != get_local_id(0))
                    {
                        Real4 tempx = (Real4)(temp_x[i*3],temp_x[i*3+1],temp_x[i*3+2],0);
                        density = SPHFluidCalcDensity(xi,tempx, density, &params);
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // loop through other groups of particles
            for (int py0 = range_x; py0 < range_y; py0 += BSIZE)
            {
                if (py0 == px0) continue;
                int py = py0 + get_local_id(0);

                if (py < range_y)
                {
                    int index2 = cells[py];

                    Real4 xj = (Real4)(x[3*index2],x[3*index2+1],x[3*index2+2],0);
//test0+=xj.x;
//test1+=xj.y;
//test2+=xj.z;
                    temp_x[tx3  ] = xj.x;
                    temp_x[tx3+1] = xj.y;
                    temp_x[tx3+2] = xj.z;
                }

                barrier(CLK_LOCAL_MEM_FENCE);

                if (px < ghost)
                {
                    // actual particle -> compute interactions
                    int np = min(range_y-py0,BSIZE);

                    for (int i=0; i < np; ++i)
                    {
                        Real4 tempx = (Real4)(temp_x[i*3],temp_x[i*3+1],temp_x[i*3+2],0);
                        density = SPHFluidCalcDensity(xi,tempx, density, &params);
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (px < ghost)
            {
                // actual particle -> write computed density
                density = SPHFluidFinalDensity(xi, density, &params);
                Real4 res = (Real4)(xi.x,xi.y,xi.z,density);
//Real4 res = (Real4)(test0,test1,test2,test3);
                pos4[index] = res;
            }
        }
    }
}



//////////////////////////////////////////////////

inline Real4 SPHFluidCalcForce(Real4 x1, Real4 v1, Real4 x2, Real4 v2, __private Real4 force, GPUSPHFluid* params)
{
    Real4 n = (Real4)(x2.x-x1.x,x2.y-x1.y,x2.z-x1.z,0);
    Real d2 = n.x*n.x+n.y*n.y+n.z*n.z;
    if (d2 < params->h2)
    {
        //Real r_h = rsqrtf(params->h2/d2);
        //Real r_h = sqrtf(d2/params->h2);
        Real r2_h2 = (d2/params->h2);
        Real r_h = native_sqrt(r2_h2);
        Real density1 = x1.w;
        Real density2 = x2.w;

        // Pressure
        Real pressure1 = params->stiffness * (density1 - params->density0);
        Real pressure2 = params->stiffness * (density2 - params->density0);
        Real4 f1 = gradWp(n, r_h, params->CgradWp) * ( params->mass2 * (pressure1 / (density1*density1) + pressure2 / (density2*density2) ));

        // Viscosity
        Real4 f2 = ( v2 - v1 ) * (params->mass2 * params->viscosity * laplacianWv(r_h,params->ClaplacianWv) / (density1 * density2) );

        return f1+f2+force;
    }
    return force; //(Real4)(0,0,0,0);
}

__kernel void SPHFluidForceField_addForce(
    int size,
    __global int *cells,
    __global int *cellGhost,
    GPUSPHFluid params,
    __global Real* f,
    __global Real4* pos4,
    __global Real* v
)
{
    __local int range_x, range_y;
    __local int ghost;
    __local Real4 temp_x[BSIZE];
    __local Real4 temp_v[BSIZE];

    int tx  = get_local_id(0);

    for (int cell = get_group_id(0); cell < size; cell += get_num_groups(0))
    {
        if (!get_local_id(0))
        {
            //range = *(const int2*)(cells+cell);
            range_x = cells[cell];
            range_y = cells[cell+1];
            range_y &= ~(1U<<31);
            ghost = cellGhost[cell];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (range_x <= 0) continue; // no actual particle in this cell
        for (int px0 = range_x; px0 < ghost; px0 += BSIZE)
        {
            int px = px0 + get_local_id(0);
            Real4 xi;
            Real4 vi;
            Real4 force = (Real4)(0,0,0,0);
            int index;
            if (px < range_y)
            {
                index = cells[px];
                xi = pos4[index];
                temp_x[tx] = xi;
                vi = (Real4)(v[index*3],v[index*3+1],v[index*3+2],0);
                temp_v[tx] = vi;
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            if (px < ghost)
            {
                // actual particle -> compute interactions
                force = (Real4)(0,0,0,0);
                int np = min(range_y-px0,BSIZE);
                for (int i=0; i < np; ++i)
                {
                    if (i != get_local_id(0))
                    {
                        force = SPHFluidCalcForce(xi, vi, temp_x[i], temp_v[i], force, &params);	//template: surface
                    }
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            // loop through other groups of particles
            for (int py0 = range_x; py0 < range_y; py0 += BSIZE)
            {
                if (py0 == px0) continue;
                int py = py0 + get_local_id(0);
                if (py < range_y)
                {
                    int index2 = cells[py];
                    Real4 xj = pos4[index2];
                    temp_x[tx] = xj;
                    Real4 vj = v[index2];
                    temp_v[tx] = vj;
                }

                barrier(CLK_LOCAL_MEM_FENCE);

                if (px < ghost)
                {
                    // actual particle -> compute interactions
                    int np = min(range_y-py0,BSIZE);
                    for (int i=0; i < np; ++i)
                    {
                        force = SPHFluidCalcForce(xi, vi, temp_x[i], temp_v[i], force, &params);
                    }
                }

                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (px < ghost)
            {
                // actual particle -> write computed force
                f[index*3]   += force.x;
                f[index*3+1] += force.y;
                f[index*3+2] += force.z;
            }
        }
    }
}


///////////////////////////////////////////////////////



