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
#ifndef SOFA_COMPONENT_TOPOLOGY_QUADSETTOPOLOGYCONTAINER_H
#define SOFA_COMPONENT_TOPOLOGY_QUADSETTOPOLOGYCONTAINER_H

#include <sofa/component/topology/EdgeSetTopologyContainer.h>

namespace sofa
{
namespace component
{
namespace topology
{
class QuadSetTopologyModifier;

using core::componentmodel::topology::BaseMeshTopology;

typedef BaseMeshTopology::PointID			PointID;
typedef BaseMeshTopology::EdgeID			EdgeID;
typedef BaseMeshTopology::QuadID			QuadID;
typedef BaseMeshTopology::Edge				Edge;
typedef BaseMeshTopology::Quad				Quad;
typedef BaseMeshTopology::SeqQuads			SeqQuads;
typedef BaseMeshTopology::QuadEdges			QuadEdges;
typedef BaseMeshTopology::VertexQuads		VertexQuads;
typedef BaseMeshTopology::EdgeQuads			EdgeQuads;

/** Object that stores a set of quads and provides access
to each quad and its edges and vertices */
class SOFA_COMPONENT_CONTAINER_API QuadSetTopologyContainer : public EdgeSetTopologyContainer
{
    friend class QuadSetTopologyModifier;

public:
    QuadSetTopologyContainer();

    QuadSetTopologyContainer(const sofa::helper::vector< Quad >& quads );

    virtual ~QuadSetTopologyContainer() {}

    /// Procedural creation methods
    /// @{
    virtual void clear();
    virtual void addQuad( int a, int b, int c, int d );
    /// @}

    virtual void init();

    /// BaseMeshTopology API
    /// @{

    const SeqQuads& getQuads()
    {
        return getQuadArray();
    }

    /// Returns the set of edges adjacent to a given quad.
    const QuadEdges& getEdgeQuadShell(QuadID i)
    {
        return getQuadEdge(i);
    }

    /** \brief Returns the set of quads adjacent to a given vertex.
    *
    */
    virtual const VertexQuads& getQuadVertexShell(PointID i);

    /** \brief Returns the set of quads adjacent to a given edge.
    *
    */
    virtual const EdgeQuads& getQuadEdgeShell(EdgeID i);

    /** Returns the indices of a quad given four vertex indices : returns -1 if none */
    virtual int getQuadIndex(PointID v1, PointID v2, PointID v3, PointID v4);

    /// @}

    /** \brief Checks if the topology is coherent
    *
    * Check if the shell arrays are coherent
    */
    virtual bool checkTopology() const;

    /** \brief Returns the Quad array.
    *
    */
    const sofa::helper::vector<Quad> &getQuadArray();

    /** \brief Returns the Quad Vertex Shells array.
    *
    */
    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &getQuadVertexShellArray();

    /** \brief Returns the QuadEdges array (ie provide the 4 edge indices for each quad)
    *
    */
    const sofa::helper::vector< QuadEdges > &getQuadEdgeArray() ;

    /** \brief Returns the Quad Edge Shells array (ie provides the quads adjacent to each edge)
    *
    */
    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &getQuadEdgeShellArray() ;

    /** \brief Returns the number of quads in this topology.
    *	The difference to getNbQuads() is that this method does not generate the quad array if it does not exist.
    */
    unsigned int getNumberOfQuads() const;

    /** \brief Returns the 4 edges adjacent to a given quad.
    *
    */
    const QuadEdges &getQuadEdge(const unsigned int i) ;

    /** returns the index (either 0, 1, 2, 3) of the vertex whose global index is vertexIndex. Returns -1 if none */
    int getVertexIndexInQuad(Quad &t,unsigned int vertexIndex) const;

    /** returns the index (either 0, 1, 2, 3) of the edge whose global index is edgeIndex. Returns -1 if none */
    int getEdgeIndexInQuad(QuadEdges &t,unsigned int edheIndex) const;

protected:
    /** \brief Creates the QuadSet array.
    *
    * This function must be implemented by derived classes to create a list of quads from a set of hexahedra for instance
    */
    virtual void createQuadSetArray();

    /** \brief Creates the EdgeSet array.
    *
    * Create the set of edges when needed.
    */
    virtual void createEdgeSetArray();

    bool hasQuads() const;

    bool hasQuadEdges() const;

    bool hasQuadVertexShell() const;

    bool hasQuadEdgeShell() const;

    void clearQuads();

    void clearQuadEdges();

    void clearQuadVertexShell();

    void clearQuadEdgeShell();

private:
    /** \brief Creates the array of edge indices for each quad
    *
    * This function is only called if the QuadEdge array is required.
    * m_quadEdge[i] contains the 4 indices of the 4 edges opposite to the ith vertex
    */
    void createQuadEdgeArray();

    /** \brief Creates the Quad Vertex Shell Array
    *
    * This function is only called if the QuadVertexShell array is required.
    * m_quadVertexShell[i] contains the indices of all quads adjacent to the ith vertex
    */
    void createQuadVertexShellArray();

    /** \brief Creates the Quad Edge Shell Array
    *
    * This function is only called if the QuadVertexShell array is required.
    * m_quadEdgeShell[i] contains the indices of all quads adjacent to the ith edge
    */
    void createQuadEdgeShellArray();

    /** \brief Returns a non-const quad vertex shell given a vertex index for subsequent modification
    *
    */
    sofa::helper::vector< unsigned int > &getQuadVertexShellForModification(const unsigned int vertexIndex);

    /** \brief Returns a non-const quad edge shell given the index of an edge for subsequent modification
    *
    */
    sofa::helper::vector< unsigned int > &getQuadEdgeShellForModification(const unsigned int edgeIndex);

protected:
    /// provides the set of quads
    sofa::helper::vector<Quad> m_quad;
    DataPtr< sofa::helper::vector<Quad> > d_quad;
    /// provides the 4 edges in each quad
    sofa::helper::vector<QuadEdges> m_quadEdge;
    /// for each vertex provides the set of quads adjacent to that vertex
    sofa::helper::vector< sofa::helper::vector< unsigned int > > m_quadVertexShell;
    /// for each edge provides the set of quads adjacent to that edge
    sofa::helper::vector< sofa::helper::vector< unsigned int > > m_quadEdgeShell;

    virtual void loadFromMeshLoader(sofa::component::container::MeshLoader* loader);
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
