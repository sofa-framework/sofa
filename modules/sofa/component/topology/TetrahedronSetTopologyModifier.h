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
#ifndef SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETTOPOLOGYMODIFIER_H
#define SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETTOPOLOGYMODIFIER_H

#include <sofa/component/topology/TriangleSetTopologyModifier.h>

namespace sofa
{

namespace component
{

namespace topology
{
template <class DataTypes>
class TetrahedronSetTopology;

template <class DataTypes>
class TetrahedronSetTopologyLoader;

using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::TetraID TetraID;
typedef BaseMeshTopology::Tetra Tetra;
typedef BaseMeshTopology::SeqTetras SeqTetras;
typedef BaseMeshTopology::VertexTetras VertexTetras;
typedef BaseMeshTopology::EdgeTetras EdgeTetras;
typedef BaseMeshTopology::TriangleTetras TriangleTetras;
typedef BaseMeshTopology::TetraEdges TetraEdges;
typedef BaseMeshTopology::TetraTriangles TetraTriangles;

typedef Tetra Tetrahedron;
typedef TetraEdges TetrahedronEdges;
typedef TetraTriangles TetrahedronTriangles;

/**
* A class that modifies the topology by adding and removing tetrahedra
*/
template<class DataTypes>
class TetrahedronSetTopologyModifier : public TriangleSetTopologyModifier <DataTypes>
{
    friend class TetrahedronSetTopologyLoader<DataTypes>;
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;

    TetrahedronSetTopologyModifier(core::componentmodel::topology::BaseTopology *top)
        : TriangleSetTopologyModifier<DataTypes>(top)
    {}

    virtual ~TetrahedronSetTopologyModifier() {}

    TetrahedronSetTopology< DataTypes >* getTetrahedronSetTopology() const;

    /** \brief Build  a tetrahedron set topology from a file : also modifies the MechanicalObject
    *
    */
    virtual bool load(const char *filename);

    /** \brief Write the current mesh into a msh file
    *
    */
    virtual void writeMSHfile(const char *filename);

    /** \brief Sends a message to warn that some tetrahedra were added in this topology.
    *
    * \sa addTetrahedraProcess
    */
    void addTetrahedraWarning(const unsigned int nTetrahedra,
            const sofa::helper::vector< Tetrahedron >& tetrahedraList,
            const sofa::helper::vector< unsigned int >& tetrahedraIndexList);

    /** \brief Sends a message to warn that some tetrahedra were added in this topology.
    *
    * \sa addTetrahedraProcess
    */
    void addTetrahedraWarning(const unsigned int nTetrahedra,
            const sofa::helper::vector< Tetrahedron >& tetrahedraList,
            const sofa::helper::vector< unsigned int >& tetrahedraIndexList,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs);

    /** \brief Actually Add some tetrahedra to this topology.
    *
    * \sa addTetrahedraWarning
    */
    virtual void addTetrahedraProcess(const sofa::helper::vector< Tetrahedron > &tetrahedra);

    /** \brief Sends a message to warn that some tetrahedra are about to be deleted.
    *
    * \sa removeTetrahedraProcess
    *
    * Important : parameter indices is not const because it is actually sorted from the highest index to the lowest one.
    */
    void removeTetrahedraWarning( sofa::helper::vector<unsigned int> &tetrahedra);

    /** \brief Remove a subset of tetrahedra
    *
    * Elements corresponding to these points are removed form the mechanical object's state vectors.
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
    * \sa removeTetrahedraWarning
    * @param removeIsolatedItems if true remove isolated triangles, edges and vertices
    */
    virtual void removeTetrahedraProcess( const sofa::helper::vector<unsigned int> &indices,
            const bool removeIsolatedItems=false);

    /** \brief Actually Add some triangles to this topology.
    *
    * \sa addTrianglesWarning
    */
    virtual void addTrianglesProcess(const sofa::helper::vector< Triangle > &triangles);

    /** \brief Remove a subset of triangles
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
    * @param removeIsolatedEdges if true isolated edges are also removed
    * @param removeIsolatedPoints if true isolated vertices are also removed
    */
    virtual void removeTrianglesProcess(const sofa::helper::vector<unsigned int> &indices,
            const bool removeIsolatedEdges=false,
            const bool removeIsolatedPoints=false);

    /** \brief Add some edges to this topology.
    *
    * \sa addEdgesWarning
    */
    virtual void addEdgesProcess(const sofa::helper::vector< Edge > &edges);

    /** \brief Remove a subset of edges
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
    * \sa removeEdgesWarning
    *
    * Important : parameter indices is not const because it is actually sorted from the highest index to the lowest one.
    * @param removeIsolatedItems if true remove isolated vertices
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
    virtual void addPointsProcess(const unsigned int nPoints,
            const bool addDOF = true);

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

    /** \brief Add a new point (who has no ancestors) to this topology.
    *
    * \sa addPointsWarning
    */
    virtual void addNewPoint(unsigned int i,  const sofa::helper::vector< double >& x);

    /** \brief Remove a subset of points
    *
    * Elements corresponding to these points are removed form the mechanical object's state vectors.
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removePointsWarning before calling removePointsProcess.
    * \sa removePointsWarning
    * Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)
    */
    virtual void removePointsProcess( sofa::helper::vector<unsigned int> &indices, const bool removeDOF = true);

    /** \brief Reorder this topology.
    *
    * Important : the points are actually renumbered in the mechanical object's state vectors iff (renumberDOF == true)
    * \see MechanicalObject::renumberValues
    */
    virtual void renumberPointsProcess( const sofa::helper::vector<unsigned int> &index,
            const sofa::helper::vector<unsigned int> &/*inv_index*/,
            const bool renumberDOF = true);

protected:
    /** \brief Load a tetrahedron.
    */
    void addTetrahedron(Tetrahedron e);
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
