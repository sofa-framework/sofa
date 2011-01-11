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
#ifndef SOFA_GPU_OPENCL_OPENCLMECHANICALOBJECT_H
#define SOFA_GPU_OPENCL_OPENCLMECHANICALOBJECT_H

#include "OpenCLTypes.h"
#include <sofa/component/container/MechanicalObject.h>

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
    typedef typename Main::VecId VecId;
    typedef typename Main::VMultiOp VMultiOp;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;


    typedef gpu::opencl::OpenCLKernelsMechanicalObject<DataTypes> Kernels;

    /// Temporary storate for dot product operation
    VecDeriv tmpdot;

    static void accumulateForce(Main* m);
    static void addDxToCollisionModel(Main* m);
    static void vAlloc(Main* m, VecId v);
    static void vOp(Main* m, VecId v, VecId a, VecId b, double f);
    static void vMultiOp(Main* m, const VMultiOp& ops);
    static double vDot(Main* m, VecId a, VecId b);
    static void resetForce(Main* m);
};


// I know using macros is bad design but this is the only way not to repeat the code for all OpenCL types
#define OpenCLMechanicalObject_DeclMethods(T) \
	template<> double MechanicalObject< T >::vDot(VecId a, VecId b);					\
	template<> void MechanicalObject< T >::vOp(VecId v, VecId a, VecId b, double f);	\
	template<> void MechanicalObject< T >::accumulateForce();							\
	template <> void MechanicalObject< T >::addDxToCollisionModel();					\
	template<> void MechanicalObject< T >::resetForce();								\
	template<> void MechanicalObject< T >::vMultiOp(const VMultiOp& ops);				\
//*/


OpenCLMechanicalObject_DeclMethods(gpu::opencl::OpenCLVec3fTypes);
OpenCLMechanicalObject_DeclMethods(gpu::opencl::OpenCLVec3f1Types);
OpenCLMechanicalObject_DeclMethods(gpu::opencl::OpenCLVec3dTypes);
OpenCLMechanicalObject_DeclMethods(gpu::opencl::OpenCLVec3d1Types);

#undef OpenCLMechanicalObject_DeclMethods

//*/

} // namespace container

} // namespace component

} // namespace sofa

#endif
