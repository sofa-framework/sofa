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

// CONVENTION : indices ordering for the vertices of an hexahedron :
//
// 	   Y  3---------2
//     ^ /	       /|
//     |/	      / |
//     7---------6  |
//     |    	 |  |
//     |  0------|--1
//     | / 	     | /
//     |/	     |/
//     4---------5-->X
//    /
//   /
//  Z

#ifndef SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETTOPOLOGYCONTAINER_H
#define SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETTOPOLOGYCONTAINER_H

#include <sofa/component/topology/QuadSetTopologyContainer.h>

namespace sofa
{
namespace component
{
namespace topology
{
template <class DataTypes>
class HexahedronSetTopology;

template <class DataTypes>
class HexahedronSetTopologyModifier;

using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::HexaID HexaID;

typedef BaseMeshTopology::Hexa Hexa;
typedef BaseMeshTopology::SeqHexas SeqHexas;
typedef BaseMeshTopology::VertexHexas VertexHexas;
typedef BaseMeshTopology::EdgeHexas EdgeHexas;
typedef BaseMeshTopology::QuadHexas QuadHexas;
typedef BaseMeshTopology::HexaEdges HexaEdges;
typedef BaseMeshTopology::HexaQuads HexaQuads;

typedef Hexa Hexahedron;
typedef HexaEdges HexahedronEdges;
typedef HexaQuads HexahedronQuads;

/** a class that stores a set of hexahedra and provides access with adjacent quads, edges and vertices */
class HexahedronSetTopologyContainer : public QuadSetTopologyContainer
{
    template< typename DataTypes >
    friend class HexahedronSetTopologyModifier;

public:
    HexahedronSetTopologyContainer(core::componentmodel::topology::BaseTopology *top = NULL);

    HexahedronSetTopologyContainer(core::componentmodel::topology::BaseTopology *top,
            const sofa::helper::vector< Hexahedron > &hexahedra);

    virtual ~HexahedronSetTopologyContainer() {}

    virtual void init();

    template< typename DataTypes >
    HexahedronSetTopology< DataTypes >* getHexahedronSetTopology() const
    {
        return static_cast<HexahedronSetTopology< DataTypes >* > (this->m_basicTopology);
    }

    /** \brief Checks if the Hexahedron Set Topology is coherent
    *
    */
    virtual bool checkTopology() const;

    /** \brief Returns the Hexahedron array.
    *
    */
    const sofa::helper::vector<Hexahedron> &getHexahedronArray();

    /** \brief Returns the ith Hexahedron.
    *
    */
    const Hexahedron &getHexahedron(const unsigned int i);

    /** \brief Returns the number of hexahedra in this topology.
    *
    */
    unsigned int getNumberOfHexahedra() ;

    /** \brief Returns the Hexahedron Vertex Shells array.
    *
    */
    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &getHexahedronVertexShellArray() ;

    /** \brief Returns the set of hexahedra adjacent to a given vertex.
    *
    */
    const sofa::helper::vector< unsigned int > &getHexahedronVertexShell(const unsigned int i) ;

    /** \brief Returns the Hexahedron Edges  array.
    *
    */
    const sofa::helper::vector< HexahedronEdges > &getHexahedronEdgeArray() ;

    /** \brief Returns the 12 edges adjacent to a given hexahedron.
    *
    */
    const HexahedronEdges &getHexahedronEdges(const unsigned int i) ;

    /** \brief Returns for each index (between 0 and 12) the two vertex indices that are adjacent to that edge
    *
    */
    Edge getLocalHexahedronEdges (const unsigned int i) const;

    /** \brief Returns the Hexahedron Quads  array.
    *
    */
    const sofa::helper::vector< HexahedronQuads > &getHexahedronQuadArray() ;

    /** \brief Returns the 6 quads adjacent to a given hexahedron.
    *
    */
    const HexahedronQuads &getHexahedronQuads(const unsigned int i) ;

    /** \brief Returns the Hexahedron Edge Shells array.
    *
    */
    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &getHexahedronEdgeShellArray() ;

    /** \brief Returns the set of hexahedra adjacent to a given edge.
    *
    */
    const sofa::helper::vector< unsigned int > &getHexahedronEdgeShell(const unsigned int i) ;


    /** \brief Returns the Hexahedron Quad Shells array.
    *
    */
    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &getHexahedronQuadShellArray() ;

    /** \brief Returns the set of hexahedra adjacent to a given quad.
    *
    */
    const sofa::helper::vector< unsigned int > &getHexahedronQuadShell(const unsigned int i) ;

    /** Returns the indices of a hexahedron given 8 vertex indices : returns -1 if none */
    int getHexahedronIndex(const unsigned int v1, const unsigned int v2,
            const unsigned int v3, const unsigned int v4,
            const unsigned int v5, const unsigned int v6,
            const unsigned int v7, const unsigned int v8);

    /** returns the index of the vertex whose global index is vertexIndex. Returns -1 if none */
    int getVertexIndexInHexahedron(Hexahedron &t,unsigned int vertexIndex) const;

    /** returns the index of the edge whose global index is edgeIndex. Returns -1 if none */
    int getEdgeIndexInHexahedron(const HexahedronEdges &t,unsigned int edgeIndex) const;

    /** returns the index of the quad whose global index is quadIndex. Returns -1 if none */
    int getQuadIndexInHexahedron(const HexahedronQuads &t,unsigned int quadIndex) const;

    inline friend std::ostream& operator<< (std::ostream& out, const HexahedronSetTopologyContainer& t)
    {
        out  << t.m_hexahedron<< " "
                << t.m_hexahedronEdge<< " "
                << t.m_hexahedronQuad;

        out << " "<< t.m_hexahedronVertexShell.size();
        for (unsigned int i=0; i<t.m_hexahedronVertexShell.size(); i++)
        {
            out << " " << t.m_hexahedronVertexShell[i];
        }
        out <<" "<< t.m_hexahedronEdgeShell.size();
        for (unsigned int i=0; i<t.m_hexahedronEdgeShell.size(); i++)
        {
            out << " " << t.m_hexahedronEdgeShell[i];
        }
        out <<" "<< t.m_hexahedronQuadShell.size();
        for (unsigned int i=0; i<t.m_hexahedronQuadShell.size(); i++)
        {
            out << " " << t.m_hexahedronQuadShell[i];
        }
        return out;
    }

    inline friend std::istream& operator>>(std::istream& in, HexahedronSetTopologyContainer& t)
    {
        unsigned int s;
        sofa::helper::vector< unsigned int > value;


        in >> t.m_hexahedron >> t.m_hexahedronEdge >> t.m_hexahedronQuad;


        in >> s;
        for (unsigned int i=0; i<s; i++)
        {
            in >> value;
            t.m_hexahedronVertexShell.push_back(value);
        }
        in >> s;
        for (unsigned int i=0; i<s; i++)
        {
            in >> value;
            t.m_hexahedronEdgeShell.push_back(value);
        }
        in >> s;
        for (unsigned int i=0; i<s; i++)
        {
            in >> value;
            t.m_hexahedronQuadShell.push_back(value);
        }
        return in;
    }

protected:
    /** \brief Creates the EdgeSet array.
    *
    * Create the set of edges when needed.
    */
    virtual void createEdgeSetArray();

    /** \brief Creates the QuadSet array.
    *
    * Create the array of quads
    */
    virtual void createQuadSetArray();

    /** \brief Creates the HexahedronSet array.
    *
    * This function must be implemented by a derived classes
    */
    virtual void createHexahedronSetArray();

    bool hasHexahedra() const;

    bool hasHexahedronEdges() const;

    bool hasHexahedronQuads() const;

    bool hasHexahedronVertexShell() const;

    bool hasHexahedronEdgeShell() const;

    bool hasHexahedronQuadShell() const;

    void clearHexahedra();

    void clearHexahedronEdges();

    void clearHexahedronQuads();

    void clearHexahedronVertexShell();

    void clearHexahedronEdgeShell();

    void clearHexahedronQuadShell();

private:
    /** \brief Creates the array of edge indices for each hexahedron
    *
    * This function is only called if the HexahedronEdge array is required.
    * m_hexahedronEdge[i] contains the 12 indices of the 12 edges of each hexahedron
    */
    void createHexahedronEdgeArray();

    /** \brief Creates the array of quad indices for each hexahedron
    *
    * This function is only called if the HexahedronQuad array is required.
    * m_hexahedronQuad[i] contains the 6 indices of the 6 quads opposite to the ith vertex
    */
    void createHexahedronQuadArray();

    /** \brief Creates the Hexahedron Vertex Shell Array
    *
    * This function is only called if the HexahedronVertexShell array is required.
    * m_hexahedronVertexShell[i] contains the indices of all hexahedra adjacent to the ith vertex
    */
    void createHexahedronVertexShellArray();

    /** \brief Creates the Hexahedron Edge Shell Array
    *
    * This function is only called if the HexahedronEdheShell array is required.
    * m_hexahedronEdgeShell[i] contains the indices of all hexahedra adjacent to the ith edge
    */
    void createHexahedronEdgeShellArray();

    /** \brief Creates the Hexahedron Quad Shell Array
    *
    * This function is only called if the HexahedronQuadShell array is required.
    * m_hexahedronQuadShell[i] contains the indices of all hexahedra adjacent to the ith edge
    */
    void createHexahedronQuadShellArray();

    /** \brief Returns a non-const hexahedron vertex shell given a vertex index for subsequent modification
    *
    */
    sofa::helper::vector< unsigned int > &getHexahedronVertexShellForModification(const unsigned int vertexIndex);

    /** \brief Returns a non-const hexahedron edge shell given the index of an edge for subsequent modification
    *
    */
    sofa::helper::vector< unsigned int > &getHexahedronEdgeShellForModification(const unsigned int edgeIndex);


    /// BaseMeshTopology API
    /// @{

    const SeqHexas& getHexas()
    {
        return getHexahedronArray();
    }

    /// Returns the set of edges adjacent to a given hexahedron.
    const HexaEdges& getEdgeHexaShell(HexaID i)
    {
        return getHexahedronEdges(i);
    }

    /// Returns the set of quads adjacent to a given hexahedron.
    const HexaQuads& getQuadHexaShell(HexaID i)
    {
        return getHexahedronQuads(i);
    }

    /// Returns the set of hexahedra adjacent to a given vertex.
    const VertexHexas& getHexaVertexShell(PointID i)
    {
        return getHexahedronVertexShell(i);
    }

    /// Returns the set of hexahedra adjacent to a given edge.
    const EdgeHexas& getHexaEdgeShell(EdgeID i)
    {
        return getHexahedronEdgeShell(i);
    }

    /// Returns the set of hexahedra adjacent to a given quad.
    const QuadHexas& getHexaQuadShell(QuadID i)
    {
        return getHexahedronQuadShell(i);
    }

    /// @}

protected:
    /// provides the set of hexahedra
    sofa::helper::vector<Hexahedron> m_hexahedron;
    /// provides the set of edges for each hexahedron
    sofa::helper::vector<HexahedronEdges> m_hexahedronEdge;
    /// provides the set of quads for each hexahedron
    sofa::helper::vector<HexahedronQuads> m_hexahedronQuad;

    /// for each vertex provides the set of hexahedra adjacent to that vertex
    sofa::helper::vector< sofa::helper::vector< unsigned int > > m_hexahedronVertexShell;
    /// for each edge provides the set of hexahedra adjacent to that edge
    sofa::helper::vector< sofa::helper::vector< unsigned int > > m_hexahedronEdgeShell;
    /// for each quad provides the set of hexahedra adjacent to that edge
    sofa::helper::vector< sofa::helper::vector< unsigned int > > m_hexahedronQuadShell;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
