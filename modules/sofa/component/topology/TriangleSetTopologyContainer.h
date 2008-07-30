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
#ifndef SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGYCONTAINER_H
#define SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGYCONTAINER_H

#include <sofa/component/topology/EdgeSetTopologyContainer.h>

namespace sofa
{

namespace component
{

namespace topology
{
template<class DataTypes>
class TriangleSetTopology;

template<class DataTypes>
class TriangleSetTopologyModifier;

using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::TriangleID TriangleID;
typedef BaseMeshTopology::Triangle Triangle;
typedef BaseMeshTopology::SeqTriangles SeqTriangles;
typedef BaseMeshTopology::VertexTriangles VertexTriangles;
typedef BaseMeshTopology::EdgeTriangles EdgeTriangles;
typedef BaseMeshTopology::TriangleEdges TriangleEdges;

/** Object that stores a set of triangles and provides access
to each triangle and its edges and vertices */
class TriangleSetTopologyContainer : public EdgeSetTopologyContainer
{
    template< typename DataTypes >
    friend class TriangleSetTopologyModifier;

public:
    TriangleSetTopologyContainer();

    TriangleSetTopologyContainer(const sofa::helper::vector< Triangle > &triangles );

    virtual ~TriangleSetTopologyContainer() {}

    virtual void init();

    /// BaseMeshTopology API
    /// @{

    const SeqTriangles& getTriangles()
    {
        return getTriangleArray();
    }

    /// Returns the set of edges adjacent to a given triangle.
    const TriangleEdges& getEdgeTriangleShell(TriangleID i)
    {
        return getTriangleEdge(i);
    }

    /// @}

    /** \brief Checks if the topology is coherent
    *
    * Check if the shell arrays are coherent
    */
    virtual bool checkTopology() const;

    /** \brief Returns the Triangle array.
    *
    */
    const sofa::helper::vector<Triangle> &getTriangleArray();

    /** \brief Returns the Triangle Vertex Shells array.
    *
    */
    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &getTriangleVertexShellArray();

    /** \brief Returns the TriangleEdges array (ie provide the 3 edge indices for each triangle)
    *
    */
    const sofa::helper::vector< TriangleEdges > &getTriangleEdgeArray() ;

    /** \brief Returns the Triangle Edge Shells array (ie provides the triangles adjacent to each edge)
    *
    */
    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &getTriangleEdgeShellArray() ;

    /** \brief Returns the ith Triangle.
    *
    */
    const Triangle &getTriangle(const unsigned int i);

    /** \brief Returns the number of triangles in this topology.
    *
    */
    unsigned int getNumberOfTriangles() ;

    /** \brief Returns the set of triangles adjacent to a given vertex.
    *
    */
    const sofa::helper::vector< unsigned int > &getTriangleVertexShell(const unsigned int i) ;


    /** \brief Returns the 3 edges adjacent to a given triangle.
    *
    */
    const TriangleEdges &getTriangleEdge(const unsigned int i) ;


    /** \brief Returns the set of triangles adjacent to a given edge.
    *
    */
    const sofa::helper::vector< unsigned int > &getTriangleEdgeShell(const unsigned int i) ;

    /** Returns the indices of a triangle given three vertex indices : returns -1 if none */
    int getTriangleIndex(const unsigned int v1, const unsigned int v2, const unsigned int v3);


    /** returns the index (either 0, 1 ,2) of the vertex whose global index is vertexIndex. Returns -1 if none */
    int getVertexIndexInTriangle(const Triangle &t,const unsigned int vertexIndex) const;

    /** returns the index (either 0, 1 ,2) of the edge whose global index is edgeIndex. Returns -1 if none */
    int getEdgeIndexInTriangle(const TriangleEdges &t,const unsigned int edgeIndex) const;

    inline friend std::ostream& operator<< (std::ostream& out, const TriangleSetTopologyContainer& t)
    {
        out << t.m_triangle.size() << " " << t.m_triangle << " "
            << t.m_triangleEdge.size() << " " << t.m_triangleEdge << " "
            << t.m_triangleVertexShell.size();
        for (unsigned int i=0; i<t.m_triangleVertexShell.size(); i++)
        {
            out << " " << t.m_triangleVertexShell[i].size();
            out << " " <<t.m_triangleVertexShell[i] ;
        }
        out  << " " << t.m_triangleEdgeShell.size();
        for (unsigned int i=0; i<t.m_triangleEdgeShell.size(); i++)
        {
            out  << " " << t.m_triangleEdgeShell[i].size();
            out  << " " << t.m_triangleEdgeShell[i];
        }

        return out;
    }

    /// Needed to be compliant with Datas.
    inline friend std::istream& operator>>(std::istream& in, TriangleSetTopologyContainer& t)
    {
        unsigned int s;
        in >> s;
        for (unsigned int i=0; i<s; i++)
        {
            Triangle T; in >> T;
            t.m_triangle.push_back(T);
        }
        in >> s;
        for (unsigned int i=0; i<s; i++)
        {
            TriangleEdges T; in >> T;
            t.m_triangleEdge.push_back(T);
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
            t.m_triangleVertexShell.push_back(v);
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
            t.m_triangleEdgeShell.push_back(v);
        }

        return in;
    }

protected:
    /** \brief Creates the TriangleSet array.
    *
    * This function must be implemented by derived classes to create a list of triangles from a set of tetrahedra for instance
    */
    virtual void createTriangleSetArray();

    /** \brief Creates the EdgeSet array.
    *
    * Create the set of edges when needed.
    */
    virtual void createEdgeSetArray();

    bool hasTriangles() const;

    bool hasTriangleEdges() const;

    bool hasTriangleVertexShell() const;

    bool hasTriangleEdgeShell() const;

    void clearTriangles();

    void clearTriangleEdges();

    void clearTriangleVertexShell();

    void clearTriangleEdgeShell();

private:
    /** \brief Creates the array of edge indices for each triangle
    *
    * This function is only called if the TriangleEdge array is required.
    * m_triangleEdge[i] contains the 3 indices of the 3 edges opposite to the ith vertex
    */
    void createTriangleEdgeArray();
    /** \brief Creates the Triangle Vertex Shell Array
    *
    * This function is only called if the TriangleVertexShell array is required.
    * m_triangleVertexShell[i] contains the indices of all triangles adjacent to the ith vertex
    */
    void createTriangleVertexShellArray();

    /** \brief Creates the Triangle Edge Shell Array
    *
    * This function is only called if the TriangleVertexShell array is required.
    * m_triangleEdgeShell[i] contains the indices of all triangles adjacent to the ith edge
    */
    void createTriangleEdgeShellArray();

    /** \brief Returns a non-const triangle vertex shell given a vertex index for subsequent modification
    *
    */
    sofa::helper::vector< unsigned int > &getTriangleVertexShellForModification(const unsigned int vertexIndex);
    /** \brief Returns a non-const triangle edge shell given the index of an edge for subsequent modification
    *
    */
    sofa::helper::vector< unsigned int > &getTriangleEdgeShellForModification(const unsigned int edgeIndex);

protected:
    /// provides the set of triangles
    sofa::helper::vector<Triangle> m_triangle;
    /// provides the 3 edges in each triangle
    sofa::helper::vector<TriangleEdges> m_triangleEdge;
    /// for each vertex provides the set of triangles adjacent to that vertex
    sofa::helper::vector< sofa::helper::vector< unsigned int > > m_triangleVertexShell;
    /// for each edge provides the set of triangles adjacent to that edge
    sofa::helper::vector< sofa::helper::vector< unsigned int > > m_triangleEdgeShell;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
