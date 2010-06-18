#ifndef CPUSPHFLUIDFORCEFIELD_H
#define CPUSPHFLUIDFORCEFIELD_H

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include <sofa/defaulttype/Vec.h>


class CPUSPHFluidForceField
{
public:
    enum {NUM_ELEMENTS=1000};

    typedef sofa::defaulttype::Vec4f float4 ;
    typedef sofa::defaulttype::Vec3f float3 ;

    typedef struct
    {
        float h;         ///< particles radius
        float h2;        ///< particles radius squared
        float stiffness; ///< pressure stiffness
        float mass;      ///< particles mass
        float mass2;     ///< particles mass squared
        float density0;  ///< 1000 kg/m3 for water
        float viscosity;
        float surfaceTension;

        // Precomputed constants for smoothing kernels
        float CWd;          ///< =     constWd(h)
        float CgradWd;      ///< = constGradWd(h)
        float CgradWp;      ///< = constGradWp(h)
        float ClaplacianWv; ///< =  constLaplacianWv(h)
        float CgradWc;      ///< = constGradWc(h)
        float ClaplacianWc; ///< =  constLaplacianWc(h)
    } GPUSPHFluid;

    static float3 gradWp(const float3 d, float r_h, float C)
    {
        float a = 1-r_h;
        return d*C*a*a;
    }

    static float laplacianWv(float r_h, float C)
    {
        return C*(1-r_h);
    }

    static float3 SPHFluidCalcForce(float4 x1, float3 v1, float4 x2, float3 v2, float3 force, GPUSPHFluid params);

    static void vectorAddForce(unsigned int gsize, const int*cells, const int*cellGhost,GPUSPHFluid params,float3* f, const float4 *pos4, const float3* v);

};



#endif // CPUSPHFLUIDFORCEFIELD_H
