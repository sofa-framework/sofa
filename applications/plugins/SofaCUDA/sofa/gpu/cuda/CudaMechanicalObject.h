/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_GPU_CUDA_CUDAMECHANICALOBJECT_H
#define SOFA_GPU_CUDA_CUDAMECHANICALOBJECT_H

#include "CudaTypes.h"
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/gpu/cuda/CudaBaseVector.h>

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

namespace container
{

template<class TCoord, class TDeriv, class TReal>
class MechanicalObjectInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >
{
public:
    typedef gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> DataTypes;
    typedef MechanicalObject<DataTypes> Main;
    typedef core::VecId VecId;
    typedef core::ConstVecId ConstVecId;
    typedef typename Main::VMultiOp VMultiOp;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;


    typedef gpu::cuda::CudaKernelsMechanicalObject<DataTypes> Kernels;

    /// Temporary storate for dot product operation
    VecDeriv tmpdot;

    MechanicalObjectInternalData(MechanicalObject< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >* = NULL)
    {}
    static void accumulateForce(Main* m);
    static void vAlloc(Main* m, VecId v);
    static void vOp(Main* m, VecId v, ConstVecId a, ConstVecId b, double f);
    static void vMultiOp(Main* m, const core::ExecParams* params, const VMultiOp& ops);
    static double vDot(Main* m, ConstVecId a, ConstVecId b);
    static void resetForce(Main* m);

    //loadInBaseVector
    static void copyToBaseVector(Main* m,defaulttype::BaseVector * dest, ConstVecId src, unsigned int &offset);
    //loadToBaseVector
    static void copyFromBaseVector(Main* m, VecId dest, const defaulttype::BaseVector * src, unsigned int &offset);
    //loadInCudaBaseVector
    static void copyToCudaBaseVector(Main* m,sofa::gpu::cuda::CudaBaseVectorType<Real> * dest, ConstVecId src, unsigned int &offset);
    //loadInCudaBaseVector
    static void copyFromCudaBaseVector(Main* m, VecId src, const sofa::gpu::cuda::CudaBaseVectorType<Real> * dest, unsigned int &offset);

    //addBaseVectorToState
    static void addFromBaseVectorSameSize(Main* m, VecId dest, const defaulttype::BaseVector *src, unsigned int &offset);
    //addCudaBaseVectorToState
    static void addFromCudaBaseVectorSameSize(Main* m, VecId dest, const sofa::gpu::cuda::CudaBaseVectorType<Real> *src, unsigned int &offset);
};


template< int N, class real>
class MechanicalObjectInternalData< gpu::cuda::CudaRigidTypes<N, real > >
{
public:
    typedef gpu::cuda::CudaRigidTypes<N, real> DataTypes;
    typedef MechanicalObject<DataTypes> Main;
    typedef core::VecId VecId;
    typedef core::ConstVecId ConstVecId;
    typedef typename Main::VMultiOp VMultiOp;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;


    typedef gpu::cuda::CudaKernelsMechanicalObject<DataTypes> Kernels;

    /// Temporary storate for dot product operation
    VecDeriv tmpdot;

    MechanicalObjectInternalData(MechanicalObject< gpu::cuda::CudaRigidTypes<N, real > >* = NULL)
    {}
    static void accumulateForce(Main* m);
    static void vAlloc(Main* m, VecId v);
    static void vOp(Main* m, VecId v, ConstVecId a, ConstVecId b, double f);
    static void vMultiOp(Main* m, const core::ExecParams* params, const VMultiOp& ops);
    static double vDot(Main* m, ConstVecId a, ConstVecId b);
    static void resetForce(Main* m);

//    static void loadInBaseVector(Main* m,defaulttype::BaseVector * dest, VecId src, unsigned int &offset);
    static void copyToBaseVector(Main* m,defaulttype::BaseVector * dest, ConstVecId src, unsigned int &offset);
    //loadToBaseVector
    static void copyFromBaseVector(Main* m, VecId dest, const defaulttype::BaseVector * src, unsigned int &offset);


    // static void loadInCudaBaseVector(Main* m,sofa::gpu::cuda::CudaBaseVector<Real> * dest, VecId src, unsigned int &offset);
    static void copyToCudaBaseVector(Main* m,sofa::gpu::cuda::CudaBaseVectorType<Real> * dest, ConstVecId src, unsigned int &offset);
    // static void loadInCudaBaseVector(Main* m,sofa::gpu::cuda::CudaBaseVector<Real> * dest, VecId src, unsigned int &offset);
    static void copyFromCudaBaseVector(Main* m, VecId src, const sofa::gpu::cuda::CudaBaseVectorType<Real> * dest,  unsigned int &offset);

//    static void addBaseVectorToState(Main* m, VecId dest, defaulttype::BaseVector *src, unsigned int &offset);
    static void addFromBaseVectorSameSize(Main* m, VecId dest, const defaulttype::BaseVector *src, unsigned int &offset);
//    static void addCudaBaseVectorToState(Main* m, VecId dest, sofa::gpu::cuda::CudaBaseVector<Real> *src, unsigned int &offset);
    static void addFromCudaBaseVectorSameSize(Main* m, VecId dest, const sofa::gpu::cuda::CudaBaseVectorType<Real> *src, unsigned int &offset);
};


// I know using macros is bad design but this is the only way not to repeat the code for all CUDA types
#define CudaMechanicalObject_DeclMethods(T) \
    template<> inline void MechanicalObject< T >::accumulateForce(const core::ExecParams* params, core::VecDerivId); \
    template<> inline void MechanicalObject< T >::vOp(const core::ExecParams* params, core::VecId v, core::ConstVecId a, core::ConstVecId b, SReal f); \
    template<> inline void MechanicalObject< T >::vMultiOp(const core::ExecParams* params, const VMultiOp& ops); \
    template<> inline SReal MechanicalObject< T >::vDot(const core::ExecParams* params, core::ConstVecId a, core::ConstVecId b); \
    template<> inline void MechanicalObject< T >::resetForce(const core::ExecParams* params, core::VecDerivId); \
    template<> inline void MechanicalObject< T >::copyToBaseVector(defaulttype::BaseVector * dest, core::ConstVecId src, unsigned int &offset); \
    template<> inline void MechanicalObject< T >::copyFromBaseVector(core::VecId dest, const defaulttype::BaseVector * src,  unsigned int &offset); \
    template<> inline void MechanicalObject< T >::addFromBaseVectorSameSize(core::VecId dest, const defaulttype::BaseVector *src, unsigned int &offset);

CudaMechanicalObject_DeclMethods(gpu::cuda::CudaVec1fTypes)
CudaMechanicalObject_DeclMethods(gpu::cuda::CudaVec2fTypes)
CudaMechanicalObject_DeclMethods(gpu::cuda::CudaVec3fTypes)
CudaMechanicalObject_DeclMethods(gpu::cuda::CudaVec3f1Types)
CudaMechanicalObject_DeclMethods(gpu::cuda::CudaVec6fTypes)
CudaMechanicalObject_DeclMethods(gpu::cuda::CudaRigid3fTypes)

#ifdef SOFA_GPU_CUDA_DOUBLE

CudaMechanicalObject_DeclMethods(gpu::cuda::CudaVec3dTypes)
CudaMechanicalObject_DeclMethods(gpu::cuda::CudaVec3d1Types)
CudaMechanicalObject_DeclMethods(gpu::cuda::CudaVec6dTypes)
CudaMechanicalObject_DeclMethods(gpu::cuda::CudaRigid3dTypes)

#endif // SOFA_GPU_CUDA_DOUBLE

#undef CudaMechanicalObject_DeclMethods

}

} // namespace component

} // namespace sofa


#ifndef SOFA_GPU_CUDA_CUDAMECHANICALOBJECT_CPP

using sofa::gpu::cuda::CudaVec1fTypes;
using sofa::gpu::cuda::CudaVec2fTypes;
using sofa::gpu::cuda::CudaVec3fTypes;
using sofa::gpu::cuda::CudaVec3f1Types;
using sofa::gpu::cuda::CudaVec6fTypes;
using sofa::gpu::cuda::CudaRigid3fTypes;


// template specialization must be in the same namespace as original namespace for GCC 4.1
// g++ 4.1 requires template instantiations to be declared on a parent namespace from the template class.
extern template class sofa::component::container::MechanicalObject<CudaVec1fTypes>;
extern template class sofa::component::container::MechanicalObject<CudaVec2fTypes>;
extern template class sofa::component::container::MechanicalObject<CudaVec3fTypes>;
extern template class sofa::component::container::MechanicalObject<CudaVec3f1Types>;
extern template class sofa::component::container::MechanicalObject<CudaVec6fTypes>;
extern template class sofa::component::container::MechanicalObject<CudaRigid3fTypes>;
#ifdef SOFA_GPU_CUDA_DOUBLE
using sofa::gpu::cuda::CudaVec3dTypes;
using sofa::gpu::cuda::CudaVec3d1Types;
using sofa::gpu::cuda::CudaVec6dTypes;
using sofa::gpu::cuda::CudaRigid3dTypes;
extern template class sofa::component::container::MechanicalObject<CudaVec3dTypes>;
extern template class sofa::component::container::MechanicalObject<CudaVec3d1Types>;
extern template class sofa::component::container::MechanicalObject<CudaVec6dTypes>;
extern template class sofa::component::container::MechanicalObject<CudaRigid3dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE

#endif // SOFA_GPU_CUDA_CUDAMECHANICALOBJECT_CPP

#endif
