/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETTOPOLOGYALGORITHMS_H
#define SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETTOPOLOGYALGORITHMS_H

#include <sofa/component/base/topology/TriangleSetTopologyAlgorithms.h>

namespace sofa
{

namespace component
{

namespace topology
{
class TetrahedronSetTopologyContainer;

class TetrahedronSetTopologyModifier;

template < class DataTypes >
class TetrahedronSetGeometryAlgorithms;

/**
* A class that performs topology algorithms on an TetrahedronSet.
*/
template < class DataTypes >
class TetrahedronSetTopologyAlgorithms : public TriangleSetTopologyAlgorithms<DataTypes>
{
public:
    typedef typename DataTypes::Real Real;

    TetrahedronSetTopologyAlgorithms()
        : TriangleSetTopologyAlgorithms<DataTypes>()
    {}

    virtual ~TetrahedronSetTopologyAlgorithms() {}

    virtual void init();

private:
    TetrahedronSetTopologyContainer*					m_container;
    TetrahedronSetTopologyModifier*						m_modifier;
    TetrahedronSetGeometryAlgorithms< DataTypes >*		m_geometryAlgorithms;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
