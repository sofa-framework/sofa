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
#ifndef SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETTOPOLOGYALGORITHMS_H
#define SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETTOPOLOGYALGORITHMS_H
#include "config.h"

#include <SofaBaseTopology/QuadSetTopologyAlgorithms.h>

namespace sofa
{

namespace component
{

namespace topology
{
class HexahedronSetTopologyContainer;

class HexahedronSetTopologyModifier;

template < class DataTypes >
class HexahedronSetGeometryAlgorithms;

/**
* A class that performs topology algorithms on an HexahedronSet.
*/
template < class DataTypes >
class HexahedronSetTopologyAlgorithms : public QuadSetTopologyAlgorithms<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(HexahedronSetTopologyAlgorithms,DataTypes),SOFA_TEMPLATE(QuadSetTopologyAlgorithms,DataTypes));

    typedef typename DataTypes::Real Real;
protected:
    HexahedronSetTopologyAlgorithms()
        : QuadSetTopologyAlgorithms<DataTypes>()
    { }

    virtual ~HexahedronSetTopologyAlgorithms() {}
public:
    virtual void init();

private:
    HexahedronSetTopologyContainer*						m_container;
    HexahedronSetTopologyModifier*						m_modifier;
    HexahedronSetGeometryAlgorithms< DataTypes >*		m_geometryAlgorithms;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETTOPOLOGYALGORITHMS_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BASE_TOPOLOGY_API HexahedronSetTopologyAlgorithms<defaulttype::Vec3dTypes>;
extern template class SOFA_BASE_TOPOLOGY_API HexahedronSetTopologyAlgorithms<defaulttype::Vec2dTypes>;
extern template class SOFA_BASE_TOPOLOGY_API HexahedronSetTopologyAlgorithms<defaulttype::Vec1dTypes>;
//extern template class SOFA_BASE_TOPOLOGY_API HexahedronSetTopologyAlgorithms<defaulttype::Rigid3dTypes>;
//extern template class SOFA_BASE_TOPOLOGY_API HexahedronSetTopologyAlgorithms<defaulttype::Rigid2dTypes>;
#endif

#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_TOPOLOGY_API HexahedronSetTopologyAlgorithms<defaulttype::Vec3fTypes>;
extern template class SOFA_BASE_TOPOLOGY_API HexahedronSetTopologyAlgorithms<defaulttype::Vec2fTypes>;
extern template class SOFA_BASE_TOPOLOGY_API HexahedronSetTopologyAlgorithms<defaulttype::Vec1fTypes>;
//extern template class SOFA_BASE_TOPOLOGY_API HexahedronSetTopologyAlgorithms<defaulttype::Rigid3fTypes>;
//extern template class SOFA_BASE_TOPOLOGY_API HexahedronSetTopologyAlgorithms<defaulttype::Rigid2fTypes>;
#endif
#endif

} // namespace topology

} // namespace component

} // namespace sofa

#endif
