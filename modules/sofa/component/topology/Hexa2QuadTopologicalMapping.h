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
#ifndef SOFA_COMPONENT_TOPOLOGY_HEXA2QUADTOPOLOGICALMAPPING_H
#define SOFA_COMPONENT_TOPOLOGY_HEXA2QUADTOPOLOGICALMAPPING_H

#include <sofa/core/topology/TopologicalMapping.h>

#include <sofa/defaulttype/Vec.h>
#include <map>

#include <sofa/core/BaseMapping.h>
#include <sofa/component/component.h>

namespace sofa
{

namespace component
{

namespace topology
{
using namespace sofa::defaulttype;

using namespace sofa::component::topology;
using namespace sofa::core::topology;

using namespace sofa::core;

/**
* This class, called Hexa2QuadTopologicalMapping, is a specific implementation of the interface TopologicalMapping where :
*
* INPUT TOPOLOGY = HexahedronSetTopology
* OUTPUT TOPOLOGY = QuadSetTopology, as the boundary of the INPUT TOPOLOGY
*
* Hexa2QuadTopologicalMapping class is templated by the pair (INPUT TOPOLOGY, OUTPUT TOPOLOGY)
*
*/

class SOFA_COMPONENT_TOPOLOGY_API Hexa2QuadTopologicalMapping : public TopologicalMapping
{
public:
    SOFA_CLASS(Hexa2QuadTopologicalMapping,TopologicalMapping);

    /** \brief Constructor.
    *
    * @param from the topology issuing TopologyChange objects (the "source").
    * @param to   the topology for which the TopologyChange objects must be translated (the "target").
    */
    Hexa2QuadTopologicalMapping(In* from=NULL, Out* to=NULL);

    /** \brief Destructor.
    *
    * Does nothing.
    */
    virtual ~Hexa2QuadTopologicalMapping();

    /** \brief Initializes the target BaseTopology from the source BaseTopology.
    */
    virtual void init();


    /** \brief Translates the TopologyChange objects from the source to the target.
    *
    * Translates each of the TopologyChange objects waiting in the source list so that they have a meaning and
    * reflect the effects of the first topology changes on the second topology.
    *
    */
    virtual void updateTopologicalMappingTopDown();

    virtual unsigned int getFromIndex(unsigned int ind);

};

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_TOPOLOGY_HEXA2QUADTOPOLOGICALMAPPING_H
