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
    { }

    virtual ~ManifoldTriangleSetTopologyModifier() {}

    virtual void init();

    virtual bool removePrecondition(sofa::helper::vector< unsigned int >& items);

    virtual void removePostProcessing();

    virtual void removePointsProcess(sofa::helper::vector<unsigned int> &indices, const bool removeDOF = true);

private:

    ManifoldTriangleSetTopologyContainer* m_container;

    /** \brief This vector store all the modifications to apply to the topology.
     */
    std::map< unsigned int, sofa::helper::vector <unsigned int> > m_modifications;

    /** Create the vector m_modifications which store the modifications to apply to the topology.
     * Thus, tests can be done before any triangle(s) removal, in order to keep the topology Manifold.
     * m_modifications[0] = vertex index number.
     * m_modifications[i>0] = 0 (no change) or 1 (remove m_triangleVertexShell[ m_modifications[0] ][i+1])
     */
    void createFutureModifications(sofa::helper::vector< unsigned int >& items);

    /** Test the modifications to apply around one vertex. After removing triangles, only one connexe composante
     * should stay.
     */
    bool testRemoveModifications();

};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
