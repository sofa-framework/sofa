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
#ifndef SOFA_COMPONENT_TOPOLOGY_SIMPLETESELATEDTETRATOPOLOGICALMAPPING_H
#define SOFA_COMPONENT_TOPOLOGY_SIMPLETESELATEDTETRATOPOLOGICALMAPPING_H
#include "config.h"

#include <sofa/core/topology/TopologicalMapping.h>

#include <sofa/defaulttype/Vec.h>
#include <map>
#include <set>

#include <sofa/core/BaseMapping.h>
#include <SofaBaseTopology/TopologyData.h>

namespace sofa
{
namespace component
{
namespace topology
{
/**
 * This class, called SimpleTesselatedTetraTopologicalMapping, is a specific implementation of the interface TopologicalMapping where :
 *
 * INPUT TOPOLOGY = TetrahedronSetTopology
 * OUTPUT TOPOLOGY = TetrahedronSetTopology which is a finer tesselated version of the INPUT TOPOLOGY
 *
 * Each tetrahedron in the input Topology will be divided in eight tetrahedra in the output topology
 *
*/

class SimpleTesselatedTetraTopologicalMapping : public sofa::core::topology::TopologicalMapping
{
public:
    SOFA_CLASS(SimpleTesselatedTetraTopologicalMapping,sofa::core::topology::TopologicalMapping);
protected:
    /** \brief Constructor.
         *
     */
    SimpleTesselatedTetraTopologicalMapping ();

    /** \brief Destructor.
     *
         * Does nothing.
         */
    virtual ~SimpleTesselatedTetraTopologicalMapping() {}
public:
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

    /** \brief Translates the TopologyChange objects from the target to the source.
     *
     * Translates each of the TopologyChange objects waiting in the source list so that they have a meaning and
     * reflect the effects of the second topology changes on the first topology.
     *
     */
    virtual void updateTopologicalMappingBottomUp();

    /// Return true if this mapping is able to propagate topological changes from input to output model
    virtual bool propagateFromInputToOutputModel() { return true; }

    /// Return true if this mapping is able to propagate topological changes from output to input model
    virtual bool propagateFromOutputToInputModel() { return true; }

    const helper::vector<int>& getPointMappedFromPoint() const { return d_pointMappedFromPoint.getValue(); }
    const helper::vector<int>& getPointMappedFromEdge() const { return d_pointMappedFromEdge.getValue(); }
    const helper::vector<int>& getPointSource() const { return d_pointSource.getValue(); }

protected:
    TetrahedronData< sofa::helper::vector<sofa::helper::fixed_array<int, 8> > > tetrahedraMappedFromTetra; ///< Each Tetrahedron of the input topology is mapped to the 8 tetrahedrons in which it can be divided.
    TetrahedronData< sofa::helper::vector<int> > tetraSource; ///<Which tetra from the input topology map to a given tetra in the output topology (-1 if none)

    Data< helper::vector<int> > d_pointMappedFromPoint; ///< Each point of the input topology is mapped to the same point.
    Data< helper::vector<int> > d_pointMappedFromEdge; ///< Each edge of the input topology is mapped to his midpoint.
    Data< helper::vector<int> > d_pointSource; ///< Which input topology element map to a given point in the output topology : 0 -> none, > 0 -> point index + 1, < 0 , - edge index -1

    void swapOutputPoints(int i1, int i2);
    void removeOutputPoints( const sofa::helper::vector<unsigned int>& tab );
    void renumberOutputPoints( const sofa::helper::vector<unsigned int>& tab );

    void swapOutputTetrahedra(int i1, int i2);
    void removeOutputTetrahedra( const sofa::helper::vector<unsigned int>& tab );

    void setPointSource(int i, int source)
    {
        helper::WriteAccessor< Data< sofa::helper::vector<int> > > pointSourceData = d_pointSource;
        helper::WriteAccessor< Data< sofa::helper::vector<int> > > pointMappedFromPointData = d_pointMappedFromPoint;
        helper::WriteAccessor< Data< sofa::helper::vector<int> > > pointMappedFromEdgeData = d_pointMappedFromEdge;


        if (i != -1)
            pointSourceData[i] = source;
        if (source > 0)
        {
            pointMappedFromPointData[source-1] = i;
        }
        else if (source < 0)
        {
            pointMappedFromEdgeData[-source-1] = i;
        }
    }
    std::set<unsigned int> tetrahedraToRemove;


    void swapInputPoints(int i1, int i2);
    void removeInputPoints( const sofa::helper::vector<unsigned int>& tab );
    void renumberInputPoints( const sofa::helper::vector<unsigned int>& tab );
    void swapInputEdges(int i1, int i2);
    void removeInputEdges( const sofa::helper::vector<unsigned int>& tab );

    void swapInputTetrahedra(int i1, int i2);
    void removeInputTetrahedra( const sofa::helper::vector<unsigned int>& tab );

};

} // namespace topology
} // namespace component
} // namespace sofa

#endif // SOFA_COMPONENT_TOPOLOGY_TETRA2TRIANGLETOPOLOGICALMAPPING_H
