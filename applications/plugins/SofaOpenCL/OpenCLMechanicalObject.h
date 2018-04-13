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
#ifndef SOFAOPENCL_OPENCLMECHANICALOBJECT_H
#define SOFAOPENCL_OPENCLMECHANICALOBJECT_H

#include "OpenCLTypes.h"
#include <SofaBaseMechanics/MechanicalObject.h>

namespace sofa
{

namespace gpu
{

namespace opencl
{

template<class DataTypes>
class OpenCLKernelsMechanicalObject;

} // namespace opencl

} // namespace gpu




namespace component
{

namespace container
{

template<class TCoord, class TDeriv, class TReal>
class MechanicalObjectInternalData< gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> >
{
public:
    typedef gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> DataTypes;
    typedef MechanicalObject<DataTypes> Main;
    typedef core::VecId VecId;
    typedef core::ConstVecId ConstVecId;
    typedef typename Main::VMultiOp VMultiOp;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;


    typedef gpu::opencl::OpenCLKernelsMechanicalObject<DataTypes> Kernels;

    /// Temporary storate for dot product operation
    VecDeriv tmpdot;

    MechanicalObjectInternalData(MechanicalObject< gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> >* = NULL)
    {}
    static void accumulateForce(Main* m);
    static void vAlloc(Main* m, VecId v);
    static void vOp(Main* m, VecId v, ConstVecId a, ConstVecId b, SReal f);
    static void vMultiOp(Main* m, const core::ExecParams* params, const VMultiOp& ops);
    static SReal vDot(Main* m, ConstVecId a, ConstVecId b);
    static void resetForce(Main* m);
};


// I know using macros is bad design but this is the only way not to repeat the code for all OpenCL types
#define OpenCLMechanicalObject_DeclMethods(T) \
    template<> inline void MechanicalObject< T >::accumulateForce(const core::ExecParams* params, core::VecDerivId f); \
    template<> inline void MechanicalObject< T >::vOp(const core::ExecParams* params /* PARAMS FIRST */, core::VecId v, core::ConstVecId a, core::ConstVecId b, SReal f); \
    template<> inline void MechanicalObject< T >::vMultiOp(const core::ExecParams* params /* PARAMS FIRST */, const VMultiOp& ops); \
    template<> inline SReal MechanicalObject< T >::vDot(const core::ExecParams* params /* PARAMS FIRST */, core::ConstVecId a, core::ConstVecId b); \
    template<> inline void MechanicalObject< T >::resetForce(const core::ExecParams* params, core::VecDerivId f);

//OpenCLMechanicalObject_DeclMethods(gpu::opencl::OpenCLVec3fTypes);
OpenCLMechanicalObject_DeclMethods(gpu::opencl::OpenCLVec3f1Types)
OpenCLMechanicalObject_DeclMethods(gpu::opencl::OpenCLVec3dTypes)
OpenCLMechanicalObject_DeclMethods(gpu::opencl::OpenCLVec3d1Types)

#undef OpenCLMechanicalObject_DeclMethods

} // namespace container

} // namespace component

} // namespace sofa

#endif
