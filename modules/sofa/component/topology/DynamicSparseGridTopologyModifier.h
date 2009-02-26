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
#ifndef SOFA_COMPONENT_TOPOLOGY_DYNAMICSPARSEGRIDTOPOLOGYMODIFIER_H
#define SOFA_COMPONENT_TOPOLOGY_DYNAMICSPARSEGRIDTOPOLOGYMODIFIER_H

#include <sofa/component/topology/HexahedronSetTopologyModifier.h>

namespace sofa
{
namespace component
{
namespace topology
{
class DynamicSparseGridTopologyContainer;

/**
* A class that modifies the topology by adding and removing hexahedra
*/
class DynamicSparseGridTopologyModifier : public HexahedronSetTopologyModifier
{
public:
    DynamicSparseGridTopologyModifier()
        : HexahedronSetTopologyModifier()
    { }

    virtual ~DynamicSparseGridTopologyModifier() {}

    virtual void init();

    /** \brief Actually Add some hexahedra to this topology. Wrong way to add some hexas for the moment !
    *
    * TEMPORARY BUT THIS METHOD MUST NOT BE USED !!
    *
    * \sa addHexahedraWarning
    */
    virtual void addHexahedraProcess ( const sofa::helper::vector< Hexahedron > &hexahedra );

    /** \brief Actually Add some hexahedra to this topology.
    *
    * This overloaded function updates relation between hexahedra indices in the topology and hexahedra indices in the regular grid.
    *
    * \sa addHexahedraWarning
    */
    virtual void addHexahedraProcess ( const sofa::helper::vector< Hexahedron > &hexahedra, const sofa::helper::vector< unsigned int> &indices );

    /** \brief Sends a message to warn that some hexahedra are about to be deleted.
    *
    * \sa removeHexahedraProcess
    *
    * Important : parameter indices is not const because it is actually sorted from the highest index to the lowest one.
    */
    virtual void removeHexahedraWarning ( sofa::helper::vector<unsigned int> &hexahedra );

private:
    DynamicSparseGridTopologyContainer* m_DynContainer;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
