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
#include <sofa/component/mapping/linear/config.h>

#include <sofa/core/topology/TopologicalMapping.h>

#include <sofa/type/Vec.h>
#include <map>
#include <set>

#include <sofa/core/BaseMapping.h>
#include <sofa/core/topology/TopologyData.h>

namespace sofa::component::mapping::linear
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

class SOFA_COMPONENT_MAPPING_LINEAR_API SimpleTesselatedTetraTopologicalMapping : public sofa::core::topology::TopologicalMapping
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
    ~SimpleTesselatedTetraTopologicalMapping() override {}
public:
    /** \brief Initializes the target BaseTopology from the source BaseTopology.
     */
    void init() override;

    /** \brief Translates the TopologyChange objects from the source to the target.
     *
     * Translates each of the TopologyChange objects waiting in the source list so that they have a meaning and
     * reflect the effects of the first topology changes on the second topology.
     *
     */
    void updateTopologicalMappingTopDown() override;

    /** \brief Translates the TopologyChange objects from the target to the source.
     *
     * Translates each of the TopologyChange objects waiting in the source list so that they have a meaning and
     * reflect the effects of the second topology changes on the first topology.
     *
     */
    void updateTopologicalMappingBottomUp() override;

    /// Return true if this mapping is able to propagate topological changes from input to output model
    bool propagateFromInputToOutputModel() override { return true; }

    /// Return true if this mapping is able to propagate topological changes from output to input model
    bool propagateFromOutputToInputModel() override { return true; }

    const type::vector<Index>& getPointMappedFromPoint() const { return d_pointMappedFromPoint.getValue(); }
    const type::vector<Index>& getPointMappedFromEdge() const { return d_pointMappedFromEdge.getValue(); }
    const type::vector<Index>& getPointSource() const { return d_pointSource.getValue(); }

protected:
    core::topology::TetrahedronData< sofa::type::vector<sofa::type::fixed_array<Index, 8> > > tetrahedraMappedFromTetra; ///< Each Tetrahedron of the input topology is mapped to the 8 tetrahedrons in which it can be divided.
    core::topology::TetrahedronData< sofa::type::vector<Index> > tetraSource; ///<Which tetra from the input topology map to a given tetra in the output topology (INVALID_INDEX if none)

    Data< type::vector<Index> > d_pointMappedFromPoint; ///< Each point of the input topology is mapped to the same point.
    Data< type::vector<Index> > d_pointMappedFromEdge; ///< Each edge of the input topology is mapped to his midpoint.
    Data< type::vector<Index> > d_pointSource; ///< Which input topology element map to a given point in the output topology : 0 -> none, > 0 -> point index + 1, < 0 , - edge index -1

    void swapOutputPoints(Index i1, Index i2);
    void removeOutputPoints( const sofa::type::vector<Index>& tab );
    void renumberOutputPoints( const sofa::type::vector<Index>& tab );

    void swapOutputTetrahedra(Index i1, Index i2);
    void removeOutputTetrahedra( const sofa::type::vector<Index>& tab );

    void setPointSource(int i, int source)
    {
        helper::WriteAccessor< Data< sofa::type::vector<Index> > > pointSourceData = d_pointSource;
        helper::WriteAccessor< Data< sofa::type::vector<Index> > > pointMappedFromPointData = d_pointMappedFromPoint;
        helper::WriteAccessor< Data< sofa::type::vector<Index> > > pointMappedFromEdgeData = d_pointMappedFromEdge;


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
    std::set<Index> tetrahedraToRemove;


    void swapInputPoints(Index i1, Index i2);
    void removeInputPoints( const sofa::type::vector<Index>& tab );
    void renumberInputPoints( const sofa::type::vector<Index>& tab );
    void swapInputEdges(Index i1, Index i2);
    void removeInputEdges( const sofa::type::vector<Index>& tab );

    void swapInputTetrahedra(Index i1, Index i2);
    void removeInputTetrahedra( const sofa::type::vector<Index>& tab );

};

} //namespace sofa::component::mapping::linear
