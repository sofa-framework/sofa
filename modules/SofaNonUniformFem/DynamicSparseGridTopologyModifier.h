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
#ifndef SOFA_COMPONENT_TOPOLOGY_DYNAMICSPARSEGRIDTOPOLOGYMODIFIER_H
#define SOFA_COMPONENT_TOPOLOGY_DYNAMICSPARSEGRIDTOPOLOGYMODIFIER_H
#include "config.h"

#include <SofaBaseTopology/HexahedronSetTopologyModifier.h>

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
class SOFA_NON_UNIFORM_FEM_API DynamicSparseGridTopologyModifier : public HexahedronSetTopologyModifier
{
public:
    SOFA_CLASS(DynamicSparseGridTopologyModifier,HexahedronSetTopologyModifier);
protected:
    DynamicSparseGridTopologyModifier()
        : HexahedronSetTopologyModifier()
    { }

    virtual ~DynamicSparseGridTopologyModifier() {}
public:
    virtual void init() override;

    /** \brief Actually Add some hexahedra to this topology. Wrong way to add some hexahedra for the moment !
    *
    * TEMPORARY BUT THIS METHOD MUST NOT BE USED !!
    *
    * \sa addHexahedraWarning
    */
    virtual void addHexahedraProcess ( const sofa::helper::vector< Hexahedron > &hexahedra ) override;

    /** \brief Actually Add some hexahedra to this topology.
    *
    * This overloaded function updates relation between hexahedra indices in the topology and hexahedra indices in the regular grid.
    *
    * \sa addHexahedraWarning
    */
    virtual void addHexahedraProcess ( const sofa::helper::vector< Hexahedron > &hexahedra, const sofa::helper::vector< unsigned int> &indices );

    /** \brief Remove a subset of hexahedra
    *
    * Elements corresponding to these points are removed form the mechanical object's state vectors.
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
    * \sa removeHexahedraWarning
    * @param removeIsolatedItems if true remove isolated quads, edges and vertices
    */
    virtual void removeHexahedraProcess(const sofa::helper::vector<unsigned int>&indices, const bool removeIsolatedItems = false) override;

    /** \brief Renumber the attributes of this class
    *
    * This method is automatically called in removeHexahedraProcess() if the user has not ever called it.
    */
    virtual void renumberAttributes( const sofa::helper::vector<unsigned int> &hexahedra );

private:
    bool everRenumbered;
    DynamicSparseGridTopologyContainer* m_DynContainer;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
