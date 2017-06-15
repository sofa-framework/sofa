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
#ifndef SOFA_COMPONENT_TOPOLOGY_CENTERPOINTTOPOLOGICALMAPPING_H
#define SOFA_COMPONENT_TOPOLOGY_CENTERPOINTTOPOLOGICALMAPPING_H
#include "config.h"

#include <sofa/core/BaseMapping.h>
#include <sofa/core/topology/TopologicalMapping.h>

namespace sofa
{
namespace component
{
namespace topology
{

/**
 * This class, called CenterPointTopologicalMapping, is a specific implementation of the interface TopologicalMapping where :
 *
 * INPUT TOPOLOGY = any MeshTopology
 * OUTPUT TOPOLOGY = A PointSetTopologie, as the boundary of the INPUT TOPOLOGY
 *
 * Each primitive in the input Topology will be mapped to a point in the output topology.
 *
 * CenterPointTopologicalMapping class is templated by the pair (INPUT TOPOLOGY, OUTPUT TOPOLOGY)
 *
*/

class SOFA_TOPOLOGY_MAPPING_API CenterPointTopologicalMapping : public sofa::core::topology::TopologicalMapping
{
public:
    SOFA_CLASS(CenterPointTopologicalMapping,sofa::core::topology::TopologicalMapping);
protected:
    /** \brief Constructor.
     *
     */
    CenterPointTopologicalMapping ();

    /** \brief Destructor.
     *
     * Does nothing.
     */
    virtual ~CenterPointTopologicalMapping() {}
public:
    /** \brief Initializes the target BaseTopology from the source BaseTopology.
     */
    virtual void init();

    /// Method called at each topological changes propagation which comes from the INPUT topology to adapt the OUTPUT topology :
    virtual void updateTopologicalMappingTopDown();

    virtual unsigned int getGlobIndex(unsigned int ind)
    {
        return ind;
    }

    virtual unsigned int getFromIndex(unsigned int ind)
    {
        return ind;
    }
};

} // namespace topology
} // namespace component
} // namespace sofa

#endif
