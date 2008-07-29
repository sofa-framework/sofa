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
#ifndef SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGYMODIFIER_H
#define SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGYMODIFIER_H

#include <sofa/component/topology/EdgeSetTopologyModifier.h>

namespace sofa
{
namespace component
{
namespace topology
{
template<class DataTypes>
class TriangleSetTopology;

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
template<class DataTypes>
class TriangleSetTopologyModifier : public EdgeSetTopologyModifier <DataTypes>
{

public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;

    TriangleSetTopologyModifier()
        : EdgeSetTopologyModifier<DataTypes>()
    { }
    TriangleSetTopologyModifier(core::componentmodel::topology::BaseTopology *top)
        : EdgeSetTopologyModifier<DataTypes>(top)
    { }

    virtual ~TriangleSetTopologyModifier() {}

    TriangleSetTopology< DataTypes >* getTriangleSetTopology() const;

    /** \brief Write the current mesh into a msh file
     *
     */
    virtual void writeMSHfile(const char *filename);

    /** \brief Sends a message to warn that some triangles were added in this topology.
     *
     * \sa addTrianglesProcess
     */
    void addTrianglesWarning(const unsigned int nTriangles,
            const sofa::helper::vector< Triangle >& trianglesList,
            const sofa::helper::vector< unsigned int >& trianglesIndexList) ;

    /** \brief Sends a message to warn that some triangles were added in this topology.
     *
     * \sa addTrianglesProcess
     */
    void addTrianglesWarning(const unsigned int nTriangles,
            const sofa::helper::vector< Triangle >& trianglesList,
            const sofa::helper::vector< unsigned int >& trianglesIndexList,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs) ;

    /** \brief Sends a message to warn that some edges were added in this topology.
    *
    * \sa addEdgesProcess
    */
    void addEdgesWarning(const unsigned int nEdges,
            const sofa::helper::vector< Edge >& edgesList,
            const sofa::helper::vector< unsigned int >& edgesIndexList)
    {
        EdgeSetTopologyModifier<DataTypes>::addEdgesWarning( nEdges, edgesList, edgesIndexList);
    }

    /** \brief Sends a message to warn that some edges were added in this topology.
    *
    * \sa addEdgesProcess
    */
    void addEdgesWarning(const unsigned int nEdges,
            const sofa::helper::vector< Edge >& edgesList,
            const sofa::helper::vector< unsigned int >& edgesIndexList,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
    {
        EdgeSetTopologyModifier<DataTypes>::addEdgesWarning( nEdges, edgesList, edgesIndexList, ancestors, baryCoefs);
    }

    /** \brief Actually Add some triangles to this topology.
    *
    * \sa addTrianglesWarning
    */
    virtual void addTrianglesProcess(const sofa::helper::vector< Triangle > &triangles);

    /** \brief Sends a message to warn that some triangles are about to be deleted.
    *
    * \sa removeTrianglesProcess
    *
    * Important : parameter indices is not const because it is actually sorted from the highest index to the lowest one.
    */
    virtual void removeTrianglesWarning( sofa::helper::vector<unsigned int> &triangles);

    /** \brief Remove a subset of  triangles. Eventually remove isolated edges and vertices
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
    * \sa removeTrianglesWarning
    *
    * @param removeIsolatedEdges if true isolated edges are also removed
    * @param removeIsolatedPoints if true isolated vertices are also removed
    */
    virtual void removeTrianglesProcess( const sofa::helper::vector<unsigned int> &indices,
            const bool removeIsolatedEdges=false,
            const bool removeIsolatedPoints=false);

    /** \brief Add some edges to this topology.
     *
     * \sa addEdgesWarning
     */
    void addEdgesProcess(const sofa::helper::vector< Edge > &edges);

    /** \brief Remove a subset of edges
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
    * \sa removeEdgesWarning
    *
    * @param removeIsolatedItems if true isolated vertices are also removed
    * Important : parameter indices is not const because it is actually sorted from the highest index to the lowest one.
    */
    virtual void removeEdgesProcess( const sofa::helper::vector<unsigned int> &indices,
            const bool removeIsolatedItems=false);

    /** \brief Add some points to this topology.
    *
    * Use a list of ancestors to create the new points.
    * Last parameter baryCoefs defines the coefficient used for the creation of the new points.
    * Default value for these coefficient (when none is defined) is 1/n with n being the number of ancestors
    * for the point being created.
    * Important : the points are actually added to the mechanical object's state vectors iff (addDOF == true)
    *
    * \sa addPointsWarning
    */
    virtual void addPointsProcess(const unsigned int nPoints, const bool addDOF = true);

    /** \brief Add some points to this topology.
     *
     * Use a list of ancestors to create the new points.
     * Last parameter baryCoefs defines the coefficient used for the creation of the new points.
     * Default value for these coefficient (when none is defined) is 1/n with n being the number of ancestors
     * for the point being created.
     * Important : the points are actually added to the mechanical object's state vectors iff (addDOF == true)
     *
     * \sa addPointsWarning
     */
    virtual void addPointsProcess(const unsigned int nPoints,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs,
            const bool addDOF = true);

    /** \brief Remove a subset of points
     *
     * Elements corresponding to these points are removed from the mechanical object's state vectors.
     *
     * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removePointsWarning before calling removePointsProcess.
     * \sa removePointsWarning
     * Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)
     */
    virtual void removePointsProcess(sofa::helper::vector<unsigned int> &indices,
            const bool removeDOF = true);

    /** \brief Reorder this topology.
     *
     * Important : the points are actually renumbered in the mechanical object's state vectors iff (renumberDOF == true)
     * \see MechanicalObject::renumberValues
     */
    virtual void renumberPointsProcess( const sofa::helper::vector<unsigned int> &index,
            const sofa::helper::vector<unsigned int> &inv_index,
            const bool renumberDOF = true);

    //protected:
    /** \brief Load a triangle.
     */
    void addTriangle(Triangle e);

};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
