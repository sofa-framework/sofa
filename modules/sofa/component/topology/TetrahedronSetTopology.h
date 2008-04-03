/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETTOPOLOGY_H

#include <sofa/component/topology/TriangleSetTopology.h>
#include <vector>
#include <map>

namespace sofa
{

namespace component
{

namespace topology
{

using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::TetraID TetraID;

/// defining Tetrahedra as 4 DOFs indices
//typedef helper::fixed_array<unsigned int,4> Tetrahedron;
/// defining TetrahedronTriangles as 4 Triangles indices
//typedef helper::fixed_array<unsigned int,4> TetrahedronTriangles;
/// defining TetrahedronEdges as 6 Edge indices
//typedef helper::fixed_array<unsigned int,6> TetrahedronEdges;
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

/////////////////////////////////////////////////////////
/// TopologyChange subclasses
/////////////////////////////////////////////////////////



/** indicates that some tetrahedra were added */
class TetrahedraAdded : public core::componentmodel::topology::TopologyChange
{

public:
    unsigned int nTetrahedra;

    sofa::helper::vector< Tetrahedron > tetrahedronArray;

    sofa::helper::vector< unsigned int > tetrahedronIndexArray;

    sofa::helper::vector< sofa::helper::vector< unsigned int > > ancestorsList;

    sofa::helper::vector< sofa::helper::vector< double > > coefs;

    TetrahedraAdded(const unsigned int nT,
            const sofa::helper::vector< Tetrahedron >& _tetrahedronArray = (const sofa::helper::vector< Tetrahedron >)0,
            const sofa::helper::vector< unsigned int >& tetrahedraIndex = (const sofa::helper::vector< unsigned int >)0,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors = (const sofa::helper::vector< sofa::helper::vector< unsigned int > >)0,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs = (const sofa::helper::vector< sofa::helper::vector< double > >)0)
        : core::componentmodel::topology::TopologyChange(core::componentmodel::topology::TETRAHEDRAADDED), nTetrahedra(nT), tetrahedronArray(_tetrahedronArray), tetrahedronIndexArray(tetrahedraIndex),ancestorsList(ancestors), coefs(baryCoefs)
    {   }

    unsigned int getNbAddedTetrahedra() const
    {
        return nTetrahedra;
    }

};



/** indicates that some tetrahedra are about to be removed */
class TetrahedraRemoved : public core::componentmodel::topology::TopologyChange
{

public:
    sofa::helper::vector<unsigned int> removedTetrahedraArray;

public:
    TetrahedraRemoved(const sofa::helper::vector<unsigned int> _tArray) : core::componentmodel::topology::TopologyChange(core::componentmodel::topology::TETRAHEDRAREMOVED), removedTetrahedraArray(_tArray)
    {
    }

    const sofa::helper::vector<unsigned int> &getArray() const
    {
        return removedTetrahedraArray;
    }
    unsigned int getNbRemovedTetrahedra() const
    {
        return removedTetrahedraArray.size();
    }

};



/////////////////////////////////////////////////////////
/// TetrahedronSetTopology objects
/////////////////////////////////////////////////////////


/** a class that stores a set of tetrahedra and provides access with adjacent triangles, edges and vertices */
class TetrahedronSetTopologyContainer : public TriangleSetTopologyContainer
{
private:
    /** \brief Creates the array of edge indices for each tetrahedron
     *
     * This function is only called if the TetrahedronEdge array is required.
     * m_tetrahedronEdge[i] contains the 6 indices of the 6 edges of each tetrahedron
     The number of each edge is the following : edge 0 links vertex 0 and 1, edge 1 links vertex 0 and 2,
     edge 2 links vertex 0 and 3, edge 3 links vertex 1 and 2, edge 4 links vertex 1 and 3,
     edge 5 links vertex 2 and 3
    */
    void createTetrahedronEdgeArray();
    /** \brief Creates the array of triangle indices for each tetrahedron
     *
     * This function is only called if the TetrahedronTriangle array is required.
     * m_tetrahedronTriangle[i] contains the 4 indices of the 4 triangles opposite to the ith vertex
     */
    void createTetrahedronTriangleArray();
    /** \brief Creates the Tetrahedron Vertex Shell Array
     *
     * This function is only called if the TetrahedronVertexShell array is required.
     * m_tetrahedronVertexShell[i] contains the indices of all tetrahedra adjacent to the ith vertex
     */
    void createTetrahedronVertexShellArray();

    /** \brief Creates the Tetrahedron Edge Shell Array
     *
     * This function is only called if the TetrahedronEdheShell array is required.
     * m_tetrahedronEdgeShell[i] contains the indices of all tetrahedra adjacent to the ith edge
     */
    void createTetrahedronEdgeShellArray();
    /** \brief Creates the Tetrahedron Triangle Shell Array
     *
     * This function is only called if the TetrahedronTriangleShell array is required.
     * m_tetrahedronTriangleShell[i] contains the indices of all tetrahedra adjacent to the ith edge
     */
    void createTetrahedronTriangleShellArray();
protected:
    /// provides the set of tetrahedra
    sofa::helper::vector<Tetrahedron> m_tetrahedron;
    /// provides the set of edges for each tetrahedron
    sofa::helper::vector<TetrahedronEdges> m_tetrahedronEdge;
    /// provides the set of triangles for each tetrahedron
    sofa::helper::vector<TetrahedronTriangles> m_tetrahedronTriangle;

    /// for each vertex provides the set of tetrahedra adjacent to that vertex
    sofa::helper::vector< sofa::helper::vector< unsigned int > > m_tetrahedronVertexShell;
    /// for each edge provides the set of tetrahedra adjacent to that edge
    sofa::helper::vector< sofa::helper::vector< unsigned int > > m_tetrahedronEdgeShell;
    /// for each triangle provides the set of tetrahedra adjacent to that edge
    sofa::helper::vector< sofa::helper::vector< unsigned int > > m_tetrahedronTriangleShell;


    /** \brief Creates the EdgeSet array.
     *
     * Create the set of edges when needed.
     */
    virtual void createEdgeSetArray() {createTetrahedronEdgeArray();}

    /** \brief Creates the TriangleSet array.
     *
     * Create the array of triangles
     */
    virtual void createTriangleSetArray() {createTetrahedronTriangleArray();}

    /** \brief Creates the TetrahedronSet array.
     *
     * This function is only called by derived classes to create a list of edges from a set of tetrahedra or tetrahedra
     */
    virtual void createTetrahedronSetArray() {}

public:

    inline friend std::ostream& operator<< (std::ostream& out, const TetrahedronSetTopologyContainer& t)
    {
        out  << t.m_tetrahedron<< " "
                << t.m_tetrahedronEdge<< " "
                << t.m_tetrahedronTriangle;

        out << " "<< t.m_tetrahedronVertexShell.size();
        for (unsigned int i=0; i<t.m_tetrahedronVertexShell.size(); i++)
        {
            out << " " << t.m_tetrahedronVertexShell[i];
        }
        out <<" "<< t.m_tetrahedronEdgeShell.size();
        for (unsigned int i=0; i<t.m_tetrahedronEdgeShell.size(); i++)
        {
            out << " " << t.m_tetrahedronEdgeShell[i];
        }
        out <<" "<< t.m_tetrahedronTriangleShell.size();
        for (unsigned int i=0; i<t.m_tetrahedronTriangleShell.size(); i++)
        {
            out << " " << t.m_tetrahedronTriangleShell[i];
        }
        return out;
    }

    inline friend std::istream& operator>>(std::istream& in, TetrahedronSetTopologyContainer& t)
    {
        unsigned int s;
        sofa::helper::vector< unsigned int > value;


        in >> t.m_tetrahedron >> t.m_tetrahedronEdge >> t.m_tetrahedronTriangle;


        in >> s;
        for (unsigned int i=0; i<s; i++)
        {
            in >> value;
            t.m_tetrahedronVertexShell.push_back(value);
        }
        in >> s;
        for (unsigned int i=0; i<s; i++)
        {
            in >> value;
            t.m_tetrahedronEdgeShell.push_back(value);
        }
        in >> s;
        for (unsigned int i=0; i<s; i++)
        {
            in >> value;
            t.m_tetrahedronTriangleShell.push_back(value);
        }
        return in;
    }
    /** \brief Returns the Tetrahedron array.
     *
     */
    const sofa::helper::vector<Tetrahedron> &getTetrahedronArray();

    /** \brief Returns the ith Tetrahedron.
     *
     */
    const Tetrahedron &getTetrahedron(const unsigned int i);

    /** \brief Returns the number of tetrahedra in this topology.
     *
     */
    unsigned int getNumberOfTetrahedra() ;

    /** \brief Returns the Tetrahedron Vertex Shells array.
     *
     */
    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &getTetrahedronVertexShellArray() ;

    /** \brief Returns the set of tetrahedra adjacent to a given vertex.
     *
     */
    const sofa::helper::vector< unsigned int > &getTetrahedronVertexShell(const unsigned int i) ;

    /** \brief Returns the index of the edge joining vertex v1 and vertex v2; returns -1 if no edge exists
     *
     */

    /** \brief Returns the Tetrahedron Edges  array.
     *
     */
    const sofa::helper::vector< TetrahedronEdges > &getTetrahedronEdgeArray() ;

    /** \brief Returns the 6 edges adjacent to a given tetrahedron.
     *
     */
    const TetrahedronEdges &getTetrahedronEdges(const unsigned int i) ;

    /** \brief Returns for each index (between 0 and 5) the two vertex indices that are adjacent to that edge
     *
     */
    Edge getLocalTetrahedronEdges (const unsigned int i) const;



    /** \brief Returns the Tetrahedron Triangles  array.
     *
     */
    const sofa::helper::vector< TetrahedronTriangles > &getTetrahedronTriangleArray() ;

    /** \brief Returns the 4 triangles adjacent to a given tetrahedron.
     *
     */
    const TetrahedronTriangles &getTetrahedronTriangles(const unsigned int i) ;


    /** \brief Returns the Tetrahedron Edge Shells array.
     *
     */
    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &getTetrahedronEdgeShellArray() ;

    /** \brief Returns the set of tetrahedra adjacent to a given edge.
     *
     */
    const sofa::helper::vector< unsigned int > &getTetrahedronEdgeShell(const unsigned int i) ;

    /** \brief Returns the Tetrahedron Triangle Shells array.
     *
     */
    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &getTetrahedronTriangleShellArray() ;

    /** \brief Returns the set of tetrahedra adjacent to a given triangle.
     *
     */
    const sofa::helper::vector< unsigned int > &getTetrahedronTriangleShell(const unsigned int i) ;


    /** Returns the indices of a tetrahedron given four vertex indices : returns -1 if none */
    int getTetrahedronIndex(const unsigned int v1, const unsigned int v2, const unsigned int v3, const unsigned int v4);


    /** returns the index (either 0, 1 ,2 or 3) of the vertex whose global index is vertexIndex. Returns -1 if none */
    int getVertexIndexInTetrahedron(const Tetrahedron &t,unsigned int vertexIndex) const;

    /** returns the index (either 0, 1 ,2, 3, 4 or 5) of the edge whose global index is edgeIndex. Returns -1 if none */
    int getEdgeIndexInTetrahedron(const TetrahedronEdges &t,unsigned int edgeIndex) const;

    /** returns the index (either 0, 1 ,2 or 3) of the triangle whose global index is triangleIndex. Returns -1 if none */
    int getTriangleIndexInTetrahedron(const TetrahedronTriangles &t,unsigned int triangleIndex) const;


    /** \brief Checks if the Tetrahedron Set Topology is coherent
     *
     */
    virtual bool checkTopology() const;

    TetrahedronSetTopologyContainer(core::componentmodel::topology::BaseTopology *top=NULL,
            /* const sofa::helper::vector< unsigned int > &DOFIndex = (const sofa::helper::vector< unsigned int >)0,  */
            const sofa::helper::vector< Tetrahedron >         &tetrahedra    = (const sofa::helper::vector< Tetrahedron >)        0 );



    template< typename DataTypes >
    friend class TetrahedronSetTopologyModifier;
protected:
    /** \brief Returns a non-const tetrahedron vertex shell given a vertex index for subsequent modification
     *
     */
    sofa::helper::vector< unsigned int > &getTetrahedronVertexShellForModification(const unsigned int vertexIndex);
    /** \brief Returns a non-const tetrahedron edge shell given the index of an edge for subsequent modification
     *
     */
    sofa::helper::vector< unsigned int > &getTetrahedronEdgeShellForModification(const unsigned int edgeIndex);


};


template <class DataTypes>
class TetrahedronSetTopologyLoader;

/**
 * A class that modifies the topology by adding and removing tetrahedra
 */
template<class DataTypes>
class TetrahedronSetTopologyModifier : public TriangleSetTopologyModifier <DataTypes>
{

public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;

    TetrahedronSetTopologyModifier(core::componentmodel::topology::BaseTopology *top) : TriangleSetTopologyModifier<DataTypes>(top)
    {
    }
    /** \brief Build  a tetrahedron set topology from a file : also modifies the MechanicalObject
     *
     */
    virtual bool load(const char *filename);

    /*
    template< typename DataTypes >
      friend class TetrahedronSetTopologyAlgorithms;
    */

    //protected:
    /** \brief Sends a message to warn that some tetrahedra were added in this topology.
     *
     * \sa addTetrahedraProcess
     */
    void addTetrahedraWarning(const unsigned int nTetrahedra,
            const sofa::helper::vector< Tetrahedron >& tetrahedraList,
            const sofa::helper::vector< unsigned int >& tetrahedraIndexList,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors= (const sofa::helper::vector< sofa::helper::vector<unsigned int > >) 0 ,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs= (const sofa::helper::vector< sofa::helper::vector< double > >)0) ;



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
    virtual void removeTetrahedraProcess( const sofa::helper::vector<unsigned int> &indices,const bool removeIsolatedItems=false);

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
    virtual void removeTrianglesProcess(const sofa::helper::vector<unsigned int> &indices, const bool removeIsolatedEdges=false, const bool removeIsolatedPoints=false);

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
    virtual void removeEdgesProcess( const sofa::helper::vector<unsigned int> &indices,const bool removeIsolatedItems=false);



    /** \brief Add some points to this topology.
     *
     * Use a list of ancestors to create the new points.
     * Last parameter baryCoefs defines the coefficient used for the creation of the new points.
     * Default value for these coefficient (when none is defined) is 1/n with n being the number of ancestors
     * for the point being created.
     *
     * \sa addPointsWarning
     */
    virtual void addPointsProcess(const unsigned int nPoints,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors = (const sofa::helper::vector< sofa::helper::vector< unsigned int > >)0,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs = (const sofa::helper::vector< sofa::helper::vector< double > >)0 );

    virtual void addNewPoint( const sofa::helper::vector< double >& x) {TriangleSetTopologyModifier< DataTypes >::addNewPoint(x);};


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
     * \see MechanicalObject::renumberValues
     */
    virtual void renumberPointsProcess( const sofa::helper::vector<unsigned int> &index );


protected:
    void addTetrahedron(Tetrahedron e);

public:
    //template <class DataTypes>
    friend class TetrahedronSetTopologyLoader<DataTypes>;

};



/**
 * A class that performs topology algorithms on an TetrahedronSet.
 */
template < class DataTypes >
class TetrahedronSetTopologyAlgorithms : public TriangleSetTopologyAlgorithms<DataTypes>
{

public:

    typedef typename DataTypes::Real Real;

    TetrahedronSetTopologyAlgorithms(sofa::core::componentmodel::topology::BaseTopology *top) : TriangleSetTopologyAlgorithms<DataTypes>(top)
    {
    }

    /** \brief Remove a set  of tetrahedra
      @param tetrahedra an array of tetrahedron indices to be removed (note that the array is not const since it needs to be sorted)
      *
      */
    virtual void removeTetrahedra(sofa::helper::vector< unsigned int >& tetrahedra);

    virtual void removeItems(sofa::helper::vector< unsigned int >& items);

};

/**
 * A class that provides geometry information on an TetrahedronSet.
 */
template < class DataTypes >
class TetrahedronSetGeometryAlgorithms : public EdgeSetGeometryAlgorithms<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;


    TetrahedronSetGeometryAlgorithms(sofa::core::componentmodel::topology::BaseTopology *top) : EdgeSetGeometryAlgorithms<DataTypes>(top)
    {
    }

    /// computes the volume of tetrahedron no i and returns it
    Real computeTetrahedronVolume(const unsigned int i) const;
    /// computes the tetrahedron volume of all tetrahedra are store in the array interface
    void computeTetrahedronVolume( BasicArrayInterface<Real> &ai) const;
    /// computes the tetrahedron volume  of tetrahedron no i and returns it
    Real computeRestTetrahedronVolume(const unsigned int i) const;
};


/** Describes a topological object that only consists as a set of points :
it is a base class for all topological objects */
template<class DataTypes>
class TetrahedronSetTopology : public TriangleSetTopology <DataTypes>
{

public:
    TetrahedronSetTopology(component::MechanicalObject<DataTypes> *obj);
    DataPtr<TetrahedronSetTopologyContainer > *f_m_topologyContainer;

    virtual void init();
    /** \brief Returns the TetrahedronSetTopologyContainer object of this TetrahedronSetTopology.
     */
    TetrahedronSetTopologyContainer *getTetrahedronSetTopologyContainer() const
    {
        return (TetrahedronSetTopologyContainer *)this->m_topologyContainer;
    }
    /** \brief Returns the TetrahedronSetTopologyAlgorithms object of this TetrahedronSetTopology.
     */
    TetrahedronSetTopologyAlgorithms<DataTypes> *getTetrahedronSetTopologyAlgorithms() const
    {
        return (TetrahedronSetTopologyAlgorithms<DataTypes> *)this->m_topologyAlgorithms;
    }

    virtual core::componentmodel::topology::TopologyAlgorithms *getTopologyAlgorithms() const
    {
        return getTetrahedronSetTopologyAlgorithms();
    }

    /** \brief Returns the TetrahedronSetTopologyAlgorithms object of this TetrahedronSetTopology.
     */
    TetrahedronSetGeometryAlgorithms<DataTypes> *getTetrahedronSetGeometryAlgorithms() const
    {
        return (TetrahedronSetGeometryAlgorithms<DataTypes> *)this->m_geometryAlgorithms;
    }

    /// BaseMeshTopology API
    /// @{

    const SeqTetras& getTetras()   { return getTetrahedronSetTopologyContainer()->getTetrahedronArray(); }
    /// Returns the set of edges adjacent to a given tetrahedron.
    const TetraEdges& getEdgeTetraShell(TetraID i) { return getTetrahedronSetTopologyContainer()->getTetrahedronEdges(i); }
    /// Returns the set of triangles adjacent to a given tetrahedron.
    const TetraTriangles& getTriangleTetraShell(TetraID i) { return getTetrahedronSetTopologyContainer()->getTetrahedronTriangles(i); }
    /// Returns the set of tetrahedra adjacent to a given vertex.
    const VertexTetras& getTetraVertexShell(PointID i) { return getTetrahedronSetTopologyContainer()->getTetrahedronVertexShell(i); }
    /// Returns the set of tetrahedra adjacent to a given edge.
    const EdgeTetras& getTetraEdgeShell(EdgeID i) { return getTetrahedronSetTopologyContainer()->getTetrahedronEdgeShell(i); }
    /// Returns the set of tetrahedra adjacent to a given triangle.
    const TriangleTetras& getTetraTriangleShell(TriangleID i) { return getTetrahedronSetTopologyContainer()->getTetrahedronTriangleShell(i); }

    /// @}

};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
