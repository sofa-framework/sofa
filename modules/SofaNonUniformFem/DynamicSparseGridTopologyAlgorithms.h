/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_DYNAMICSPARSEGRIDTOPOLOGYALGORITHMS_H
#define SOFA_COMPONENT_TOPOLOGY_DYNAMICSPARSEGRIDTOPOLOGYALGORITHMS_H

#include <SofaBaseTopology/HexahedronSetTopologyAlgorithms.h>
#include <SofaNonUniformFem/DynamicSparseGridTopologyContainer.h>
#include <SofaNonUniformFem/DynamicSparseGridTopologyModifier.h>
#include <SofaNonUniformFem/DynamicSparseGridTopologyAlgorithms.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaNonUniformFem/DynamicSparseGridGeometryAlgorithms.h>
#include <algorithm>
#include <functional>

namespace sofa
{

namespace component
{

namespace topology
{
class DynamicSparseGridTopologyContainer;

class DynamicSparseGridTopologyModifier;

template < class DataTypes >
class DynamicSparseGridGeometryAlgorithms;

/**
* A class that performs topology algorithms on an HexahedronSet.
*/
template < class DataTypes >
class DynamicSparseGridTopologyAlgorithms : public HexahedronSetTopologyAlgorithms<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(DynamicSparseGridTopologyAlgorithms,DataTypes),SOFA_TEMPLATE(HexahedronSetTopologyAlgorithms,DataTypes));

    typedef typename DataTypes::Real Real;
protected:
    DynamicSparseGridTopologyAlgorithms()
        : HexahedronSetTopologyAlgorithms<DataTypes>()
    { }

    virtual ~DynamicSparseGridTopologyAlgorithms() {}
public:
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_TOPOLOGY_DYNAMICSPARSEGRIDTOPOLOGYALGORITHMS_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_NON_UNIFORM_FEM_API DynamicSparseGridTopologyAlgorithms<defaulttype::Vec3dTypes>;
extern template class SOFA_NON_UNIFORM_FEM_API DynamicSparseGridTopologyAlgorithms<defaulttype::Vec2dTypes>;
extern template class SOFA_NON_UNIFORM_FEM_API DynamicSparseGridTopologyAlgorithms<defaulttype::Vec1dTypes>;
//extern template class SOFA_NON_UNIFORM_FEM_API DynamicSparseGridTopologyAlgorithms<defaulttype::Rigid3dTypes>;
//extern template class SOFA_NON_UNIFORM_FEM_API DynamicSparseGridTopologyAlgorithms<defaulttype::Rigid2dTypes>;
#endif

#ifndef SOFA_DOUBLE
extern template class SOFA_NON_UNIFORM_FEM_API DynamicSparseGridTopologyAlgorithms<defaulttype::Vec3fTypes>;
extern template class SOFA_NON_UNIFORM_FEM_API DynamicSparseGridTopologyAlgorithms<defaulttype::Vec2fTypes>;
extern template class SOFA_NON_UNIFORM_FEM_API DynamicSparseGridTopologyAlgorithms<defaulttype::Vec1fTypes>;
//extern template class SOFA_NON_UNIFORM_FEM_API DynamicSparseGridGeometryAlgorithms<defaulttype::Rigid3fTypes>;
//extern template class SOFA_NON_UNIFORM_FEM_API DynamicSparseGridGeometryAlgorithms<defaulttype::Rigid2fTypes>;
#endif
#endif

} // namespace topology

} // namespace component

} // namespace sofa

#endif
