/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_MANIFOLDTRIANGLESETTOPOLOGYMODIFIER_H
#define SOFA_COMPONENT_TOPOLOGY_MANIFOLDTRIANGLESETTOPOLOGYMODIFIER_H
#include <ManifoldTopologies/config.h>

#include <ManifoldTopologies/config.h>

#include <SofaBaseTopology/TriangleSetTopologyModifier.h>

namespace sofa
{
namespace component
{
namespace topology
{
class ManifoldTriangleSetTopologyContainer;

using core::topology::BaseMeshTopology;
typedef BaseMeshTopology::TriangleID TriangleID;
typedef BaseMeshTopology::Triangle Triangle;
typedef BaseMeshTopology::SeqTriangles SeqTriangles;
typedef BaseMeshTopology::TrianglesAroundVertex TrianglesAroundVertex;
typedef BaseMeshTopology::TrianglesAroundEdge TrianglesAroundEdge;
typedef BaseMeshTopology::EdgesInTriangle EdgesInTriangle;

/**
 * A class that modifies the topology by adding and removing triangles
 */
class SOFA_MANIFOLD_TOPOLOGIES_API ManifoldTriangleSetTopologyModifier : public TriangleSetTopologyModifier
{
public:
    SOFA_CLASS(ManifoldTriangleSetTopologyModifier,TriangleSetTopologyModifier);

    ManifoldTriangleSetTopologyModifier()
        : TriangleSetTopologyModifier()
        , m_triSwap(initData(&m_triSwap,  "swap 2 triangles by their index", "Debug : Test swap function (only while animate)."))
        , m_swapMesh(initData (&m_swapMesh, false, "Mesh Optimization", "If true, optimize the mesh only by swapping edges"))
    {
    }

    virtual ~ManifoldTriangleSetTopologyModifier() {}

    virtual void init();

    virtual void reinit();

    virtual void Debug(); // TO BE REMOVED WHEN CLASS IS SURE.


    /** \brief Add and remove a subset of triangles. Eventually remove isolated edges and vertices
     *
     * This function is a complete workflow using differents methods of this class:
     * \sa TriangleSetTopologyModifier::removeTrianglesWarning
     * \sa TriangleSetTopologyModifier::removeTrianglesProcess
     * \sa TriangleSetTopologyModifier::addTrianglesProcess
     * \sa TriangleSetTopologyModifier::addTrianglesWarning
     * \sa reorderingTopologyOnROI
     * And not the Manifold ones. The interest is to allow intermediate non manifold topology.
     * WARNING: There are no test incusring the modification will keep the topology manifold.
     *
     * @param nTri2Add - number of triangles to add.
     * @param triangles2Add - list of Triangle to add.
     * @param trianglesIndexList - List of their index.
     * @param ancestors - list of ancestors to these new triangles.
     * @param baryCoefs - their barycoefs related to these ancestors.
     * @param trianglesIndex2remove - List of triangle indices to remove.
     */
    virtual void addRemoveTriangles (const unsigned int nTri2Add,
            const sofa::helper::vector< Triangle >& triangles2Add,
            const sofa::helper::vector< unsigned int >& trianglesIndex2Add,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs,
            sofa::helper::vector< unsigned int >& trianglesIndex2remove);


    /** \brief: Reorder the vertex in the array of a given edge. In order to be in the oriented in the right direction
     * regarding the first triangle of m_trianglesAroundEdgeArray[ edgeIndex ].
     *
     */
    void reorderingEdge(const unsigned int edgeIndex);


    /** \brief: Reorder the triangle vertex array around a given vertex.
     *
     */
    void reorderingTrianglesAroundVertex (const unsigned int vertexIndex);


    /** \brief: Reorder the edge vertex array around a given vertex.
     *
     */
    void reorderingEdgesAroundVertex (const unsigned int vertexIndex);


    /** \brief: Reorder the three shell arrays around a list of given vertices.
     *
     */
    void reorderingTopologyOnROI (const sofa::helper::vector <unsigned int>& listVertex);


protected:

    Data< sofa::helper::vector< unsigned int> > m_triSwap;
    Data< bool > m_swapMesh;

    /**\brief Preconditions to fulfill before removing triangles. In this class topology should stay manifold.
    * This function call private functions to test the topology:
    * @see createRemovingTrianglesFutureModifications()
    * @see createRemovingEdgesFutureModifications()
    * @see testRemovingModifications().
    */
    virtual bool removeTrianglesPreconditions(const sofa::helper::vector< unsigned int >& items);

    /**\brief Postprocessing to apply to the triangle topology. In this class topology should stay manifold.
    * These functions reorder the triangles around each vertex where triangles have been deleted.
    * In order that only one connexe composante stay. It call the internal functions:
    * @see internalRemovingPostProcessingEdges()
    * @see internalRemovingPostProcessingTriangles()
    * @see updateRemovingModifications()
    * @see reorderEdgeForRemoving()
    */
    virtual void removeTrianglesPostProcessing(const sofa::helper::vector< unsigned int >& edgeToBeRemoved, const sofa::helper::vector< unsigned int >& vertexToBeRemoved );


    /**\brief Preconditions to fulfill before adding triangles. In this class topology should stay manifold.
     * Test if triangles could be added and stock the informations of where triangles are added in the map:
     * @see m_modificationsEdge
     */
    virtual bool addTrianglesPreconditions (const sofa::helper::vector <Triangle>& triangles);


    /**\brief Postprocessing to apply to the triangle topology. In this class topology should stay manifold.
     * Using the map @see m_modificationsEdge, reorder the different shells.
     */
    virtual void addTrianglesPostProcessing(const sofa::helper::vector <Triangle>& triangles);


private:

    ManifoldTriangleSetTopologyContainer* m_container;

    /** \brief iterator for the map.
     */
    std::map< unsigned int, sofa::helper::vector<unsigned int> >::iterator it_modif;
    std::map< unsigned int, sofa::helper::vector<int> >::iterator it_add;


    /** \brief This map store all the modifications (for the triangles) to apply to the topology.
     */
    std::map< unsigned int, sofa::helper::vector <unsigned int> > m_modifications;


    /** \brief This vector store all the modifications (for the edges) to apply to the topology.
     */
    sofa::helper::vector< unsigned int> m_modificationsEdge;


    /** \brief This map store all the modifications (for the triangles) to apply to the topology.
     */
    std::map< unsigned int, sofa::helper::vector <int> > m_Addmodifications;


    /** Fill the vector m_modificationEdge with the 3 edges of each triangle to be removed (without duplications).
     * This is needed,if orientations of edges have to be changed (always oriented in the right direction regarding the
     * first or the only one triangle of m_TrianglesAroundEdgeArray[ the_edge ]);
     */
    void createRemovingEdgesFutureModifications (const sofa::helper::vector <unsigned int> items);


    /** Create the vector m_modifications which store the modifications to apply to the topology.
     * Thus, tests can be done before any triangle(s) removal, in order to keep the topology Manifold.
     * m_modifications[0] = vertex index number.
     * m_modifications[i>0] = 0 (no change) or 1 (remove m_trianglesAroundVertex[ m_modifications[0] ][i+1])
     */
    void createRemovingTrianglesFutureModifications(const sofa::helper::vector< unsigned int >& items);


    /** Test the modifications to apply around one vertex. After removing triangles, only one connexe composante
     * should stay.
     */
    bool testRemovingModifications();


    /** According to m_modification map, reorder the m_trianglesAroundVertexArray.
     *
     */
    void internalRemovingPostProcessingTriangles();


    /** According to m_modificationEdge vector, reorder the m_EdgesAroundVertexArray.
     *
     */
    void internalRemovingPostProcessingEdges();


    /** If isolate edges or vertices have to be removed during the operation. This function update the information in the container:
     * m_modification and m_modificationEdge
     *
     */
    void updateRemovingModifications (const sofa::helper::vector< unsigned int >& edgeToBeRemoved, const sofa::helper::vector< unsigned int >& vertexToBeRemoved);


    /** For each edge of m_modificationEdge, this function call ManifoldTriangleSetTopologyContainer::reorderingEdge() to
     * change the orientation of the edge if needed.
     *
     */
    void reorderEdgeForRemoving();

    /**\brief Function swaping edge between two adjacents triangles. Create two new triangles and remove the two old one.
    * This function call private functions of the container reordering the different shells.
    * Different from the others used in adding and removing triangles which are faster but need informations of
    * the state's topology before modifications.
    * @see ManifoldTriangleSetTopologyContainer::reorderingTopologyOnROI()
    * @param index of first triangle.
    * @param index of second triangle adjacent to the first one.
    */
    void edgeSwap (const EdgeID& indexEdge);

};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
