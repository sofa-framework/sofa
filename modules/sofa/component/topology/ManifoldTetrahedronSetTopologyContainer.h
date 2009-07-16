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
#ifndef SOFA_COMPONENT_TOPOLOGY_MANIFOLDTETRAHEDRONSETTOPOLOGYCONTAINER_H
#define SOFA_COMPONENT_TOPOLOGY_MANIFOLDTETRAHEDRONSETTOPOLOGYCONTAINER_H

#include <sofa/component/topology/TetrahedronSetTopologyContainer.h>

namespace sofa
{

namespace component
{

namespace topology
{
class TetrahedronSetTopologyModifier; //has to be change to Manifold one

using core::componentmodel::topology::BaseMeshTopology;

typedef BaseMeshTopology::PointID		PointID;
typedef BaseMeshTopology::EdgeID		EdgeID;
typedef BaseMeshTopology::TriangleID	TriangleID;
typedef BaseMeshTopology::TetraID		TetraID;
typedef BaseMeshTopology::Edge		Edge;
typedef BaseMeshTopology::Triangle	Triangle;
typedef BaseMeshTopology::Tetra		Tetra;
typedef BaseMeshTopology::SeqTetrahedra	SeqTetrahedra;
typedef BaseMeshTopology::TetrahedraAroundVertex	TetrahedraAroundVertex;
typedef BaseMeshTopology::TetrahedraAroundEdge	TetrahedraAroundEdge;
typedef BaseMeshTopology::TetrahedraAroundTriangle	TetrahedraAroundTriangle;
typedef BaseMeshTopology::EdgesInTetrahedron	EdgesInTetrahedron;
typedef BaseMeshTopology::TrianglesInTetrahedron	TrianglesInTetrahedron;

typedef Tetra		Tetrahedron;
typedef EdgesInTetrahedron	EdgesInTetrahedron;
typedef TrianglesInTetrahedron	TrianglesInTetrahedron;


/** a class that stores a set of tetrahedra and provides access with adjacent triangles, edges and vertices */
class SOFA_COMPONENT_CONTAINER_API ManifoldTetrahedronSetTopologyContainer : public TetrahedronSetTopologyContainer
{
    friend class TetrahedronSetTopologyModifier; // To be change to manifold one

public:
    typedef Tetra		Tetrahedron;
    typedef EdgesInTetrahedron	EdgesInTetrahedron;
    typedef TrianglesInTetrahedron	TrianglesInTetrahedron;

    ManifoldTetrahedronSetTopologyContainer();

    ManifoldTetrahedronSetTopologyContainer(const sofa::helper::vector< Tetrahedron >& tetrahedra );

    virtual ~ManifoldTetrahedronSetTopologyContainer() {}

    /// Procedural creation methods
    /// @{
    virtual void clear();
    /// @}

    virtual void init();

    virtual void reinit();

    /// BaseMeshTopology API
    /// @{

    /// @}

    /** \brief Checks if the topology is coherent
     *
     * Check if the shell arrays are coherent
     */
    virtual bool checkTopology() const;
    /** \brief Returns the Tetrahedron array.
     *
     */

    virtual void draw();


protected:

    Data<bool> debugViewIndices;
    Data<bool> debugViewIndicesTetra;
    Data<bool> shellDisplay;

    /** \brief Creates the Tetrahedron Vertex Shell Array
     *
     * This function is only called if the TetrahedraAroundVertex array is required.
     * m_tetrahedraAroundVertex[i] contains the indices of all tetrahedra adjacent to the ith vertex
     */
    virtual void createTetrahedraAroundVertexArray();

    /** \brief Creates the Tetrahedron Edge Shell Array
     *
     * This function is only called if the TetrahedronEdheShell array is required.
     * m_tetrahedraAroundEdge[i] contains the indices of all tetrahedra adjacent to the ith edge
     */
    virtual void createTetrahedraAroundEdgeArray();

    /** \brief Creates the Tetrahedron Triangle Shell Array
     *
     * This function is only called if the TetrahedraAroundTriangle array is required.
     * m_tetrahedraAroundTriangle[i] contains the indices of all tetrahedra adjacent to the ith edge
     */
    virtual void createTetrahedraAroundTriangleArray();

    /** \brief return if the tetrahedron is ine the same orientation as the one of reference
     * 1 if tetrahedrons have same orientation
     * 0 if tetrahedrons don't have same orientation
     * -1 if tetrahedrons don't share the same 4 vertices
     */
    int getTetrahedronOrientation(const Tetrahedron &t, const Tetrahedron &t_test );

    /** \brief return the orientation of a triangle relatively to one tetrahedron
     * 1 if good orientation
     * 0 if other orientation
     * -1 if triangle does'nt belongs to this tetrahedron
     */
    int getTriangleTetrahedronOrientation(const Tetrahedron &t, const Triangle &tri );

private:




protected:
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
