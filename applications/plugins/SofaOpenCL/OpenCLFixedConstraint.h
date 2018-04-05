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
#ifndef SOFAOPENCL_OPENCLFIXEDCONSTRAINT_H
#define SOFAOPENCL_OPENCLFIXEDCONSTRAINT_H

#include "OpenCLTypes.h"
#include <SofaBoundaryCondition/FixedConstraint.h>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{


template<class TCoord, class TDeriv, class TReal>
class FixedConstraintInternalData< gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> >
{
public:
    typedef FixedConstraintInternalData< gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> > Data;
    typedef gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> DataTypes;
    typedef FixedConstraint<DataTypes> Main;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef typename Main::SetIndex SetIndex;
    typedef typename Main::SetIndexArray SetIndexArray;

    // min/max fixed indices for contiguous constraints
    int minIndex;
    int maxIndex;
    // vector of indices for general case
    gpu::opencl::OpenCLVector<int> OpenCLIndices;

    static void init(Main* m);

    static void addConstraint(Main* m, unsigned int index);

    static void removeConstraint(Main* m, unsigned int index);

    static void projectResponse(Main* m, VecDeriv& dx);
};

#ifdef SOFA_DEV
template <int N, class real>
class FixedConstraintInternalData< gpu::opencl::OpenCLRigidTypes<N, real > >
{
public:
    typedef FixedConstraintInternalData< gpu::opencl::OpenCLRigidTypes<N, real> > Data;
    typedef gpu::opencl::OpenCLRigidTypes<N, real> DataTypes;
    typedef FixedConstraint<DataTypes> Main;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef typename Main::SetIndex SetIndex;
    typedef typename Main::SetIndexArray SetIndexArray;

    // min/max fixed indices for contiguous constraints
    int minIndex;
    int maxIndex;
    // vector of indices for general case
    gpu::opencl::OpenCLVector<int> OpenCLIndices;

    static void init(Main* m);

    static void addConstraint(Main* m, unsigned int index);

    static void removeConstraint(Main* m, unsigned int index);

    static void projectResponse(Main* m, VecDeriv& dx);
};
#endif // SOFA_DEV

// I know using macros is bad design but this is the only way not to repeat the code for all OpenCL types
#define OpenCLFixedConstraint_DeclMethods(T) \
	template<> void FixedConstraint< T >::init(); \
	template<> void FixedConstraint< T >::addConstraint(unsigned int index); \
	template<> void FixedConstraint< T >::removeConstraint(unsigned int index); \
	template<> void FixedConstraint< T >::projectResponse(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& resData);

OpenCLFixedConstraint_DeclMethods(gpu::opencl::OpenCLVec3fTypes);
OpenCLFixedConstraint_DeclMethods(gpu::opencl::OpenCLVec3f1Types);
#ifdef SOFA_DEV
OpenCLFixedConstraint_DeclMethods(gpu::opencl::OpenCLRigid3fTypes);
#endif // SOFA_DEV


OpenCLFixedConstraint_DeclMethods(gpu::opencl::OpenCLVec3dTypes);
OpenCLFixedConstraint_DeclMethods(gpu::opencl::OpenCLVec3d1Types);
#ifdef SOFA_DEV
OpenCLFixedConstraint_DeclMethods(gpu::opencl::OpenCLRigid3dTypes);
#endif // SOFA_DEV



#undef OpenCLFixedConstraint_DeclMethods



} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

#endif
