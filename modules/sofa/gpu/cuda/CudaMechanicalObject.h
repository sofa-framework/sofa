/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_GPU_CUDA_CUDAMECHANICALOBJECT_H
#define SOFA_GPU_CUDA_CUDAMECHANICALOBJECT_H

#include "CudaTypes.h"
#include <sofa/component/container/MechanicalObject.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

template<class DataTypes>
class CudaKernelsMechanicalObject;

} // namespace cuda

} // namespace gpu

namespace component
{

template<class TCoord, class TDeriv, class TReal>
class MechanicalObjectInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >
{
public:
    typedef gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> DataTypes;
    typedef MechanicalObject<DataTypes> Main;
    typedef typename Main::VecId VecId;
    typedef typename Main::VMultiOp VMultiOp;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;


    typedef gpu::cuda::CudaKernelsMechanicalObject<DataTypes> Kernels;

    /// Temporary storate for dot product operation
    VecDeriv tmpdot;
    double dotres;

    static void accumulateForce(Main* m);
    static void vAlloc(Main* m, VecId v);
    static void vOp(Main* m, VecId v, VecId a, VecId b, double f);
    static void vMultiOp(Main* m, const VMultiOp& ops);
    static double vDot(Main* m, VecId a, VecId b);
    static void resetForce(Main* m);
};

// I know using macros is bad design but this is the only way not to repeat the code for all CUDA types
#define CudaMechanicalObject_DeclMethods(T) \
    template<> bool MechanicalObject< T >::canPrefetch() const; \
    template<> void MechanicalObject< T >::accumulateForce(); \
    template<> void MechanicalObject< T >::vOp(VecId v, VecId a, VecId b, double f); \
    template<> void MechanicalObject< T >::vMultiOp(const VMultiOp& ops); \
    template<> double MechanicalObject< T >::vDot(VecId a, VecId b); \
    template<> void MechanicalObject< T >::resetForce();

CudaMechanicalObject_DeclMethods(gpu::cuda::CudaVec3fTypes);
CudaMechanicalObject_DeclMethods(gpu::cuda::CudaVec3f1Types);

#ifdef SOFA_GPU_CUDA_DOUBLE

CudaMechanicalObject_DeclMethods(gpu::cuda::CudaVec3dTypes);
CudaMechanicalObject_DeclMethods(gpu::cuda::CudaVec3d1Types);

#endif // SOFA_GPU_CUDA_DOUBLE

#undef CudaMechanicalObject_DeclMethods

} // namespace component

} // namespace sofa

#endif
