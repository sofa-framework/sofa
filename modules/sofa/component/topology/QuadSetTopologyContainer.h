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
#ifndef SOFA_COMPONENT_TOPOLOGY_QUADSETTOPOLOGYCONTAINER_H
#define SOFA_COMPONENT_TOPOLOGY_QUADSETTOPOLOGYCONTAINER_H

#include <sofa/component/topology/EdgeSetTopologyContainer.h>

namespace sofa
{
namespace component
{
namespace topology
{
template <class DataTypes>
class QuadSetTopology;

template <class DataTypes>
class QuadSetTopologyModifier;

using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::QuadID QuadID;
typedef BaseMeshTopology::Quad Quad;
typedef BaseMeshTopology::SeqQuads SeqQuads;
typedef BaseMeshTopology::VertexQuads VertexQuads;
typedef BaseMeshTopology::EdgeQuads EdgeQuads;
typedef BaseMeshTopology::QuadEdges QuadEdges;

/** Object that stores a set of quads and provides access
to each quad and its edges and vertices */
class QuadSetTopologyContainer : public EdgeSetTopologyContainer
{
    template< typename DataTypes >
    friend class QuadSetTopologyModifier;

public:
    QuadSetTopologyContainer();

    QuadSetTopologyContainer(const sofa::helper::vector< Quad >& quads );

    virtual ~QuadSetTopologyContainer() {}

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

    /** \brief Returns the ith Quad.
    *
    */
    const Quad &getQuad(const unsigned int i);

    /** \brief Returns the number of quads in this topology.
    *
    */
    unsigned int getNumberOfQuads() ;

    /** \brief Returns the set of quads adjacent to a given vertex.
    *
    */
    const sofa::helper::vector< unsigned int > &getQuadVertexShell(const unsigned int i) ;


    /** \brief Returns the 4 edges adjacent to a given quad.
    *
    */
    const QuadEdges &getQuadEdge(const unsigned int i) ;


    /** \brief Returns the set of quads adjacent to a given edge.
    *
    */
    const sofa::helper::vector< unsigned int > &getQuadEdgeShell(const unsigned int i) ;


    /** Returns the indices of a quad given four vertex indices : returns -1 if none */
    int getQuadIndex(const unsigned int v1, const unsigned int v2, const unsigned int v3, const unsigned int v4);

    /** returns the index (either 0, 1, 2, 3) of the vertex whose global index is vertexIndex. Returns -1 if none */
    int getVertexIndexInQuad(Quad &t,unsigned int vertexIndex) const;

    /** returns the index (either 0, 1, 2, 3) of the edge whose global index is edgeIndex. Returns -1 if none */
    int getEdgeIndexInQuad(QuadEdges &t,unsigned int edheIndex) const;

    inline friend std::ostream& operator<< (std::ostream& out, const QuadSetTopologyContainer& t)
    {
        out << t.m_quad.size() << " " << t.m_quad << " "
            << t.m_quadEdge.size() << " " << t.m_quadEdge << " "
            << t.m_quadVertexShell.size();
        for (unsigned int i=0; i<t.m_quadVertexShell.size(); i++)
        {
            out << " " << t.m_quadVertexShell[i].size();
            out << " " <<t.m_quadVertexShell[i] ;
        }
        out  << " " << t.m_quadEdgeShell.size();
        for (unsigned int i=0; i<t.m_quadEdgeShell.size(); i++)
        {
            out  << " " << t.m_quadEdgeShell[i].size();
            out  << " " << t.m_quadEdgeShell[i];
        }

        return out;
    }

    /// Needed to be compliant with Datas.
    inline friend std::istream& operator>>(std::istream& in, QuadSetTopologyContainer& t)
    {
        unsigned int s;
        in >> s;
        for (unsigned int i=0; i<s; i++)
        {
            Quad T; in >> T;
            t.m_quad.push_back(T);
        }
        in >> s;
        for (unsigned int i=0; i<s; i++)
        {
            QuadEdges T; in >> T;
            t.m_quadEdge.push_back(T);
        }

        unsigned int sub;
        in >> s;
        for (unsigned int i=0; i<s; i++)
        {
            in >> sub;
            sofa::helper::vector< unsigned int > v;
            for (unsigned int j=0; j<sub; j++)
            {
                unsigned int value;
                in >> value;
                v.push_back(value);
            }
            t.m_quadVertexShell.push_back(v);
        }

        in >> s;
        for (unsigned int i=0; i<s; i++)
        {
            in >> sub;
            sofa::helper::vector< unsigned int > v;
            for (unsigned int j=0; j<sub; j++)
            {
                unsigned int value;
                in >> value;
                v.push_back(value);
            }
            t.m_quadEdgeShell.push_back(v);
        }

        return in;
    }

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
    /// provides the 4 edges in each quad
    sofa::helper::vector<QuadEdges> m_quadEdge;
    /// for each vertex provides the set of quads adjacent to that vertex
    sofa::helper::vector< sofa::helper::vector< unsigned int > > m_quadVertexShell;
    /// for each edge provides the set of quads adjacent to that edge
    sofa::helper::vector< sofa::helper::vector< unsigned int > > m_quadEdgeShell;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
