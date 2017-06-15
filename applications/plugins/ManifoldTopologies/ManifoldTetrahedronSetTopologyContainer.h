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
#ifndef SOFA_COMPONENT_TOPOLOGY_MANIFOLDTETRAHEDRONSETTOPOLOGYCONTAINER_H
#define SOFA_COMPONENT_TOPOLOGY_MANIFOLDTETRAHEDRONSETTOPOLOGYCONTAINER_H
#include <ManifoldTopologies/config.h>

#include <ManifoldTopologies/config.h>

#include <SofaBaseTopology/TetrahedronSetTopologyContainer.h>

namespace sofa
{

namespace component
{

namespace topology
{
class TetrahedronSetTopologyModifier; //has to be change to Manifold one

using core::topology::BaseMeshTopology;

typedef BaseMeshTopology::PointID	                     	PointID;
typedef BaseMeshTopology::EdgeID		                EdgeID;
typedef BaseMeshTopology::TriangleID               	TriangleID;
typedef BaseMeshTopology::TetraID	                  	TetraID;
typedef BaseMeshTopology::Edge                    	Edge;
typedef BaseMeshTopology::Triangle                 	Triangle;
typedef BaseMeshTopology::Tetra        	        	Tetra;
typedef BaseMeshTopology::SeqTetrahedra           	SeqTetrahedra;
typedef BaseMeshTopology::TetrahedraAroundVertex          TetrahedraAroundVertex;
typedef BaseMeshTopology::TetrahedraAroundEdge     	TetrahedraAroundEdge;
typedef BaseMeshTopology::TetrahedraAroundTriangle	TetrahedraAroundTriangle;
typedef BaseMeshTopology::EdgesInTetrahedron       	EdgesInTetrahedron;
typedef BaseMeshTopology::TrianglesInTetrahedron      	TrianglesInTetrahedron;

typedef Tetra		Tetrahedron;
typedef EdgesInTetrahedron	EdgesInTetrahedron;
typedef TrianglesInTetrahedron	TrianglesInTetrahedron;


/** a class that stores a set of tetrahedra and provides access with adjacent triangles, edges and vertices */
class SOFA_MANIFOLD_TOPOLOGIES_API ManifoldTetrahedronSetTopologyContainer : public TetrahedronSetTopologyContainer
{
    friend class TetrahedronSetTopologyModifier; // To be change to manifold one

public:
    SOFA_CLASS(ManifoldTetrahedronSetTopologyContainer,TetrahedronSetTopologyContainer);

    ManifoldTetrahedronSetTopologyContainer();

    virtual ~ManifoldTetrahedronSetTopologyContainer() {}

    virtual void init();
    virtual void reinit();


    /// Procedural creation methods
    /// @{
    virtual void clear();
    /// @}


    /** \brief Checks if the topology is coherent
     *
     * TODO: like in ManifoldTriangles, test the topology
     */
    virtual bool checkTopology() const;
    /** \brief Returns the Tetrahedron array.
     *
     */

    /** \brief return if the tetrahedron is ine the same orientation as the one of reference
     *
     * @param Ref to the tetrahedron of reference
     * @parem Ref to the tetrahedron to test.
     * @return 1 if tetrahedrons have same orientation
     * @return 0 if tetrahedrons don't have same orientation
     * @return -1 if tetrahedrons don't share the same 4 vertices
     */
    int getTetrahedronOrientation(const Tetrahedron &t, const Tetrahedron &t_test );

    /** \brief return the orientation of a triangle relatively to one tetrahedron
     *
     * @param Ref to the tetrahedron of reference
     * @parem Ref to the triangle to test.
     * @return 1 if good orientation
     * @return 0 if other orientation
     * @return -1 if triangle does'nt belongs to this tetrahedron
     */
    int getEdgeTriangleOrientation(const Triangle& f, const Edge& e);
    int getTriangleTetrahedronOrientation(const Tetrahedron &t, const Triangle &tri );



protected:

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



protected:

    Data<bool> debugViewIndices;
    Data<bool> debugViewIndicesTetra;
    Data<bool> shellDisplay;


};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
