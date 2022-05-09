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
#include <sofa/component/topology/container/dynamic/PointSetTopologyModifier.h>

#include <sofa/type/Vec.h>
#include <map>
#include <set>

#include <sofa/core/BaseMapping.h>
#include <sofa/core/topology/TopologyData.h>


namespace sofa::component::mapping::linear
{
/**
 * This class, called Mesh2PointTopologicalMapping, is a specific implementation of the interface TopologicalMapping where :
 *
 * INPUT TOPOLOGY = any MeshTopology
 * OUTPUT TOPOLOGY = A PointSetTopologie, as the boundary of the INPUT TOPOLOGY
 *
 * Each primitive in the input Topology will be mapped to a point in the output topology computed from a parameter vector (pointBaryCoords, edgeBaryCoords, triangleBaryCoords, quadBaryCoords, tetraBaryCoords, hexaBaryCoords)
 *
 * Mesh2PointTopologicalMapping class is templated by the pair (INPUT TOPOLOGY, OUTPUT TOPOLOGY)
 *
*/

class SOFA_COMPONENT_MAPPING_LINEAR_API Mesh2PointTopologicalMapping : public sofa::core::topology::TopologicalMapping
{
public:
    SOFA_CLASS(Mesh2PointTopologicalMapping,sofa::core::topology::TopologicalMapping);
    typedef sofa::type::Vec3d Vec3d;

protected:
    /** \brief Constructor.
     *
     */
    Mesh2PointTopologicalMapping ();

    /** \brief Destructor.
     *
         * Does nothing.
         */
    ~Mesh2PointTopologicalMapping() override {}
public:
    /** \brief Initializes the target BaseTopology from the source BaseTopology.
     */
    void init() override;

    /// Method called at each topological changes propagation which comes from the INPUT topology to adapt the OUTPUT topology :
    void updateTopologicalMappingTopDown() override;

    Index getGlobIndex(Index ind) override
    {
        if(ind<pointSource.size())
        {
            return pointSource[ind].second;
        }
        else
        {
            return 0;
        }
    }

    Index getFromIndex(Index ind) override
    {
        return ind;
    }

    enum Element
    {
        POINT = 0,
        EDGE,
        TRIANGLE,
        QUAD,
        TETRA,
        HEXA,
        NB_ELEMENTS
    };

    const type::vector< type::vector<Index> >& getPointsMappedFromPoint() const { return pointsMappedFrom[POINT]; }
    const type::vector< type::vector<Index> >& getPointsMappedFromEdge() const { return pointsMappedFrom[EDGE]; }
    const type::vector< type::vector<Index> >& getPointsMappedFromTriangle() const { return pointsMappedFrom[TRIANGLE]; }
    const type::vector< type::vector<Index> >& getPointsMappedFromQuad() const { return pointsMappedFrom[QUAD]; }
    const type::vector< type::vector<Index> >& getPointsMappedFromTetra() const { return pointsMappedFrom[TETRA]; }
    const type::vector< type::vector<Index> >& getPointsMappedFromHexa() const { return pointsMappedFrom[HEXA]; }

    const type::vector< Vec3d >& getPointBaryCoords() const { return pointBaryCoords.getValue(); }
    const type::vector< Vec3d >& getEdgeBaryCoords() const { return edgeBaryCoords.getValue(); }
    const type::vector< Vec3d >& getTriangleBaryCoords() const { return triangleBaryCoords.getValue(); }
    const type::vector< Vec3d >& getQuadBaryCoords() const { return quadBaryCoords.getValue(); }
    const type::vector< Vec3d >& getTetraBaryCoords() const { return tetraBaryCoords.getValue(); }
    const type::vector< Vec3d >& getHexaBaryCoords() const { return hexaBaryCoords.getValue(); }

    const type::vector< std::pair<Element, Index> >& getPointSource() const { return pointSource;}

protected:

    Data< type::vector< Vec3d > > pointBaryCoords; ///< Coordinates for the points of the output topology created from the points of the input topology
    Data< type::vector< Vec3d > > edgeBaryCoords; ///< Coordinates for the points of the output topology created from the edges of the input topology
    Data< type::vector< Vec3d > > triangleBaryCoords; ///< Coordinates for the points of the output topology created from the triangles of the input topology
    Data< type::vector< Vec3d > > quadBaryCoords; ///< Coordinates for the points of the output topology created from the quads of the input topology
    Data< type::vector< Vec3d > > tetraBaryCoords; ///< Coordinates for the points of the output topology created from the tetra of the input topology
    Data< type::vector< Vec3d > > hexaBaryCoords; ///< Coordinates for the points of the output topology created from the hexa of the input topology

    Data< bool > copyEdges; ///< Activate mapping of input edges into the output topology (requires at least one item in pointBaryCoords)
    Data< bool > copyTriangles; ///< Activate mapping of input triangles into the output topology (requires at least one item in pointBaryCoords)
	Data< bool > copyTetrahedra; ///< Activate mapping of input tetrahedras into the output topology (requires at least one item in pointBaryCoords)

    type::fixed_array< type::vector< type::vector<Index> >, NB_ELEMENTS > pointsMappedFrom; ///< Points mapped from the differents elements (see the enum Element declared before)

    type::vector< std::pair<Element, Index> > pointSource; ///< Correspondance between the points mapped and the elements from which are mapped

    std::set<unsigned int> pointsToRemove;

    size_t addInputPoint(Index i, topology::container::dynamic::PointSetTopologyModifier* toPointMod=nullptr); ///< Returns the number of points added inside the output topology. 
    void addInputEdge(Index i, topology::container::dynamic::PointSetTopologyModifier* toPointMod=nullptr);
    void addInputTriangle(Index i, topology::container::dynamic::PointSetTopologyModifier* toPointMod=nullptr);
    void addInputTetrahedron(Index i, topology::container::dynamic::PointSetTopologyModifier* toPointMod=nullptr);

    void swapInput(Element elem, Index i1, Index i2);
    void removeInput(Element elem, const sofa::type::vector<Index>& tab );
    void renumberInput(Element elem, const sofa::type::vector<Index>& index );

    void swapOutputPoints(Index i1, Index i2, bool removeLast = false);
    void removeOutputPoints( const sofa::type::vector<Index>& tab );

protected:
    bool internalCheck(const char* step, const type::fixed_array <size_t, NB_ELEMENTS >& nbInputRemoved);
    
    bool internalCheck(const char* step)
    {
        type::fixed_array <size_t, NB_ELEMENTS > nbInputRemoved;
        nbInputRemoved.assign(0);
        return internalCheck(step, nbInputRemoved);
    }
    bool initDone;
};

} //namespace sofa::component::mapping::linear
