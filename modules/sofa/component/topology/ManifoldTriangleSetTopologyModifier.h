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
#ifndef SOFA_COMPONENT_TOPOLOGY_MANIFOLDTRIANGLESETTOPOLOGYMODIFIER_H
#define SOFA_COMPONENT_TOPOLOGY_MANIFOLDTRIANGLESETTOPOLOGYMODIFIER_H

#include <sofa/component/topology/TriangleSetTopologyModifier.h>

namespace sofa
{
namespace component
{
namespace topology
{
class ManifoldTriangleSetTopologyContainer;

using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::TriangleID TriangleID;
typedef BaseMeshTopology::Triangle Triangle;
typedef BaseMeshTopology::SeqTriangles SeqTriangles;
typedef BaseMeshTopology::VertexTriangles VertexTriangles;
typedef BaseMeshTopology::EdgeTriangles EdgeTriangles;
typedef BaseMeshTopology::TriangleEdges TriangleEdges;

/**
 * A class that modifies the topology by adding and removing triangles
 */
class SOFA_COMPONENT_CONTAINER_API ManifoldTriangleSetTopologyModifier : public TriangleSetTopologyModifier
{
public:
    ManifoldTriangleSetTopologyModifier()
        : TriangleSetTopologyModifier()
    {
        m_triSwap=this->initData(&m_triSwap,  "swap 2 triangles by their index", "Debug : Test swap function (only while animate).");
    }

    virtual ~ManifoldTriangleSetTopologyModifier() {}

    virtual void init();

    virtual void reinit();

    /**\brief Function swaping edge between two adjacents triangles. Create two new triangles and remove the two old one.
    * This function call private functions of the container reordering the different shells.
    * Different from the others used in adding and removing triangles which are faster but need informations of
    * the state's topology before modifications.
    * @see ManifoldTriangleSetTopologyContainer::reorderingTopologyOnROI()
    * @param index of first triangle.
    * @param index of second triangle adjacent to the first one.
    */
    bool edgeSwap (const TriangleID& indexTri1, const TriangleID& indexTri2);

    virtual void Debug(); // TO BE REMOVED WHEN CLASS IS SURE.

protected:

    Data< sofa::helper::vector< unsigned int> > m_triSwap;

    /**\brief Preconditions to fulfill before removing triangles. In this class topology should stay manifold.
    * This function call private functions to test the topology:
    * @see createRemovingTrianglesFutureModifications()
    * @see createRemovingEdgesFutureModifications()
    * @see testRemovingModifications().
    */
    virtual bool removeTrianglesPreconditions(sofa::helper::vector< unsigned int >& items);

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
     * first or the only one triangle of m_TriangleEdgeShellArray[ the_edge ]);
     */
    void createRemovingEdgesFutureModifications (const sofa::helper::vector <unsigned int> items);


    /** Create the vector m_modifications which store the modifications to apply to the topology.
     * Thus, tests can be done before any triangle(s) removal, in order to keep the topology Manifold.
     * m_modifications[0] = vertex index number.
     * m_modifications[i>0] = 0 (no change) or 1 (remove m_triangleVertexShell[ m_modifications[0] ][i+1])
     */
    void createRemovingTrianglesFutureModifications(sofa::helper::vector< unsigned int >& items);


    /** Test the modifications to apply around one vertex. After removing triangles, only one connexe composante
     * should stay.
     */
    bool testRemovingModifications();


    /** According to m_modification map, reorder the m_triangleVertexShellArray.
     *
     */
    void internalRemovingPostProcessingTriangles();


    /** According to m_modificationEdge vector, reorder the m_EdgeVertexShellArray.
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


};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
