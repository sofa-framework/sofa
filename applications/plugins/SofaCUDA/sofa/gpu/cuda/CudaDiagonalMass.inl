/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_COLLISION_CUDADIAGONALMASS_INL
#define SOFA_COMPONENT_COLLISION_CUDADIAGONALMASS_INL

#include "CudaDiagonalMass.h"
#include <SofaBaseMechanics/DiagonalMass.inl>


namespace sofa
{
namespace gpu
{
namespace cuda
{

using namespace sofa::gpu::cuda;

extern "C"
{
    void DiagonalMassCuda_addMDxf(unsigned int size, float factor, const void * mass, const void* dx, void* res);
    void DiagonalMassCuda_addMDxd(unsigned int size, double factor, const void * mass, const void* dx, void* res);

    void DiagonalMassCuda_accFromFf(unsigned int size, const void * mass, const void* f, void* a);
    void DiagonalMassCuda_accFromFd(unsigned int size, const void * mass, const void* f, void* a);

    void DiagonalMassCuda_addForcef(unsigned int size, const void * mass,const void * g, const void* f);
    void DiagonalMassCuda_addForced(unsigned int size, const void * mass,const void * g, const void* f);
}


}

}

namespace component
{

namespace mass
{

template<>
void DiagonalMass<CudaVec3fTypes, float>::addMDx(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f, const DataVecDeriv& d_dx, SReal d_factor)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecDeriv& dx = d_dx.getValue();

    DiagonalMassCuda_addMDxf(dx.size(),(float) d_factor, d_mass.getValue().deviceRead() , dx.deviceRead(), f.deviceWrite());
//     const MassVector &masses= d_mass.getValue();
//     for (unsigned int i=0;i<dx.size();i++) {
// 	res[i] += dx[i] * masses[i] * (Real)factor;
//     }

    d_f.endEdit();
}

template<>
void DiagonalMass<CudaVec3fTypes, float>::accFromF(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_a, const DataVecDeriv& d_f)
{
    VecDeriv& a = *d_a.beginEdit();
    const VecDeriv& f = d_f.getValue();

    DiagonalMassCuda_accFromFf(f.size(),  d_mass.getValue().deviceRead(), f.deviceRead(), a.deviceWrite());
//     const MassVector &masses= d_mass.getValue();
//     for (unsigned int i=0;i<f.size();i++) {
//         a[i] = f[i] / masses[i];
//     }

    d_a.endEdit();
}

template <>
void DiagonalMass<CudaVec3fTypes, float>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f, const DataVecCoord& /*d_x*/, const DataVecDeriv& /*d_v*/)
{
    VecDeriv& f = *d_f.beginEdit();
    //const VecCoord& x = d_x.getValue();
    //const VecDeriv& v = d_v.getValue();

    defaulttype::Vec3d g ( this->getContext()->getGravity() );
    const MassVector &masses= d_mass.getValue();
    DiagonalMassCuda_addForcef(masses.size(),masses.deviceRead(),g.ptr(), f.deviceWrite());

//     // gravity
//     Vec3d g ( this->getContext()->getGravity() );
//     Deriv theGravity;
//     DataTypes::set ( theGravity, g[0], g[1], g[2]);
//
//     for (unsigned int i=0;i<masses.size();i++) {
//         f[i] += theGravity*masses[i];
//     }

    d_f.endEdit();
}


#ifdef SOFA_GPU_CUDA_DOUBLE

template<>
//void DiagonalMass<CudaVec3dTypes, double>::addMDx(VecDeriv& res, const VecDeriv& dx, double factor)
void DiagonalMass<CudaVec3dTypes, double>::addMDx(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f, const DataVecDeriv& d_dx, SReal d_factor)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecDeriv& dx = d_dx.getValue();

    DiagonalMassCuda_addMDxd(dx.size(),(double) d_factor, d_mass.getValue().deviceRead() , dx.deviceRead(), f.deviceWrite());

    d_f.endEdit();
}

template<>
void DiagonalMass<CudaVec3dTypes, double>::accFromF(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_a, const DataVecDeriv& d_f)
{
    VecDeriv& a = *d_a.beginEdit();
    const VecDeriv& f = d_f.getValue();

    DiagonalMassCuda_accFromFd(f.size(),  d_mass.getValue().deviceRead(), f.deviceRead(), a.deviceWrite());

    d_a.endEdit();
}

template<>
void DiagonalMass<CudaVec3dTypes, double>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f, const DataVecCoord& /*d_x*/, const DataVecDeriv& /*d_v*/)
{
    VecDeriv& f = *d_f.beginEdit();
    //const VecCoord& x = d_x.getValue();
    //const VecDeriv& v = d_v.getValue();

    Vec3d g ( this->getContext()->getGravity() );
    const MassVector &masses= d_mass.getValue();
    DiagonalMassCuda_addForced(masses.size(),masses.deviceRead(),g.ptr(), f.deviceWrite());

    d_f.endEdit();
}

// template <>
// bool DiagonalMass<CudaVec3dTypes, double>::addBBox(double* minBBox, double* maxBBox) {
//     const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
//     //if (!x.isHostValid()) return false; // Do not recompute bounding box if it requires to transfer data from device
//     for (unsigned int i=0; i<x.size(); i++) {
//         //const Coord& p = x[i];
//         const Coord& p = x.getCached(i);
//         for (int c=0;c<3;c++) {
//             if (p[c] > maxBBox[c]) maxBBox[c] = p[c];
//             if (p[c] < minBBox[c]) minBBox[c] = p[c];
//         }
//     }
//     return true;
// }

#endif

} // namespace mass

} // namespace component

} // namespace sofa

#endif
