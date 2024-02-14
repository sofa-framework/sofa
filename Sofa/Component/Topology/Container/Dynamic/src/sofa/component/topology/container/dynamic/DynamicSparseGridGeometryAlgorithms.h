/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/component/topology/container/dynamic/config.h>

#include <sofa/component/topology/container/dynamic/HexahedronSetGeometryAlgorithms.h>
#include <sofa/component/topology/container/dynamic/DynamicSparseGridTopologyContainer.h>

namespace sofa::component::topology::container::dynamic
{

/**
* A class that provides geometry information on an HexahedronSet.
*/
template < class DataTypes >
class DynamicSparseGridGeometryAlgorithms : public HexahedronSetGeometryAlgorithms<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(DynamicSparseGridGeometryAlgorithms,DataTypes),SOFA_TEMPLATE(HexahedronSetGeometryAlgorithms,DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef sofa::core::State<DataTypes> MObject;
protected:
    DynamicSparseGridGeometryAlgorithms()
        : HexahedronSetGeometryAlgorithms<DataTypes>()
    {}

    virtual ~DynamicSparseGridGeometryAlgorithms() {}
public:
    /// finds a hexahedron, in its rest position, which is nearest to a given point. Computes barycentric coordinates and a distance measure.
    int findNearestElementInRestPos(const Coord& pos, type::Vec3& baryC, Real& distance) const override;

    void init() override;

    core::topology::BaseMeshTopology::HexaID getTopoIndexFromRegularGridIndex ( unsigned int index, bool& existing );
    unsigned int getRegularGridIndexFromTopoIndex ( core::topology::BaseMeshTopology::HexaID index );

protected:
    DynamicSparseGridTopologyContainer* topoContainer;
    MObject* dof;
};


template <>
int SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API DynamicSparseGridGeometryAlgorithms<defaulttype::Vec2Types>::findNearestElementInRestPos(const Coord& pos, sofa::type::Vec3& baryC, Real& distance) const;

template <>
int SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API DynamicSparseGridGeometryAlgorithms<defaulttype::Vec1Types>::findNearestElementInRestPos(const Coord& pos, sofa::type::Vec3& baryC, Real& distance) const;


#if !defined(SOFA_COMPONENT_TOPOLOGY_DYNAMICSPARSEGRIDGEOMETRYALGORITHMS_CPP)
extern template class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API DynamicSparseGridGeometryAlgorithms<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API DynamicSparseGridGeometryAlgorithms<defaulttype::Vec2Types>;
#endif

} // namespace sofa::component::topology::container::dynamic
