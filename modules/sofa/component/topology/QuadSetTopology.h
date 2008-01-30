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
#ifndef SOFA_COMPONENT_TOPOLOGY_QUADSETTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_QUADSETTOPOLOGY_H

#include <sofa/component/topology/EdgeSetTopology.h>
#include <vector>
#include <map>

#include <sofa/defaulttype/Vec.h> // typing "Vec"

namespace sofa
{

namespace component
{

namespace topology
{
using namespace sofa::defaulttype; // typing "Vec"

/// defining Quads as 4 DOFs indices
typedef helper::fixed_array<unsigned int,4> Quad;
/// defining QuadEdges as 4 Edge indices
typedef helper::fixed_array<unsigned int,4> QuadEdges;

/////////////////////////////////////////////////////////
/// TopologyChange subclasses
/////////////////////////////////////////////////////////


template< class Real>
bool is_point_in_quad(const Vec<3,Real>& p, const Vec<3,Real>& a, const Vec<3,Real>& b, const Vec<3,Real>& c, const Vec<3,Real>& d);

void snapping_test_quad(double epsilon, double alpha0, double alpha1, double alpha2, double alpha3, bool& is_snap_0, bool& is_snap_1, bool& is_snap_2, bool& is_snap_3);

template< class Real>
inline Real areaProduct(const Vec<3,Real>& a, const Vec<3,Real>& b);

template< class Real>
inline Real areaProduct(const defaulttype::Vec<2,Real>& a, const defaulttype::Vec<2,Real>& b );

template< class Real>
inline Real areaProduct(const defaulttype::Vec<1,Real>& , const defaulttype::Vec<1,Real>&  );

/** indicates that some quads were added */
class QuadsAdded : public core::componentmodel::topology::TopologyChange
{

public:
    unsigned int nQuads;

protected:
    sofa::helper::vector< Quad > quadArray;

    sofa::helper::vector< unsigned int > quadIndexArray;

public:
    sofa::helper::vector< sofa::helper::vector< unsigned int > > ancestorsList;

    sofa::helper::vector< sofa::helper::vector< double > > coefs;

public:

    QuadsAdded(const unsigned int nT,
            const sofa::helper::vector< Quad >& _quadArray = (const sofa::helper::vector< Quad >)0,
            const sofa::helper::vector< unsigned int >& quadsIndex = (const sofa::helper::vector< unsigned int >)0,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors = (const sofa::helper::vector< sofa::helper::vector< unsigned int > >)0,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs = (const sofa::helper::vector< sofa::helper::vector< double > >)0)
        : core::componentmodel::topology::TopologyChange(core::componentmodel::topology::QUADSADDED), nQuads(nT), quadArray(_quadArray), quadIndexArray(quadsIndex),ancestorsList(ancestors), coefs(baryCoefs)
    {   }

    unsigned int getNbAddedQuads() const
    {
        return nQuads;
    }

    const Quad &getQuad(const unsigned int i)
    {
        return quadArray[i];
    }

};



/** indicates that some quads are about to be removed */
class QuadsRemoved : public core::componentmodel::topology::TopologyChange
{

protected:
    sofa::helper::vector<unsigned int> removedQuadsArray;

    const sofa::helper::vector<unsigned int> &getArray() const
    {
        return removedQuadsArray;
    }

public:
    QuadsRemoved(const sofa::helper::vector<unsigned int> _qArray) : core::componentmodel::topology::TopologyChange(core::componentmodel::topology::QUADSREMOVED), removedQuadsArray(_qArray)
    {
    }

    unsigned int getNbRemovedQuads() const
    {
        return removedQuadsArray.size();
    }

    unsigned int &getQuadIndices(const unsigned int i)
    {
        return removedQuadsArray[i];
    }

};



/////////////////////////////////////////////////////////
/// QuadSetTopology objects
/////////////////////////////////////////////////////////


/** Object that stores a set of quads and provides access
to each quad and its edges and vertices */
class QuadSetTopologyContainer : public EdgeSetTopologyContainer
{
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
protected:
    /// provides the set of quads
    sofa::helper::vector<Quad> m_quad;
    /// provides the 4 edges in each quad
    sofa::helper::vector<QuadEdges> m_quadEdge;
    /// for each vertex provides the set of quads adjacent to that vertex
    sofa::helper::vector< sofa::helper::vector< unsigned int > > m_quadVertexShell;
    /// for each edge provides the set of quads adjacent to that edge
    sofa::helper::vector< sofa::helper::vector< unsigned int > > m_quadEdgeShell;


    /** \brief Creates the QuadSet array.
     *
     * This function is only called by derived classes to create a list of quads from a set of tetrahedra for instance
     */
    virtual void createQuadSetArray() {}

    /** \brief Creates the EdgeSet array.
     *
     * Create the set of edges when needed.
     */
    virtual void createEdgeSetArray() {createQuadEdgeArray();}

    /** \brief Returns the Quad array.
     *
     */
    const sofa::helper::vector<Quad> &getQuadArray();

    /** \brief Returns the Quad Vertex Shells array.
     *
     */
    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &getQuadVertexShellArray() ;

    /** \brief Returns the QuadEdges array (ie provide the 4 edge indices for each quad)
     *
     */
    const sofa::helper::vector< QuadEdges > &getQuadEdgeArray() ;

    /** \brief Returns the Quad Edge Shells array (ie provides the quads adjacent to each edge)
     *
     */
    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &getQuadEdgeShellArray() ;

public:
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

    /** \brief Checks if the Quad Set Topology is coherent
     *
     * Check if the Edge and the Edhe Shell arrays are coherent
     */
    virtual bool checkTopology() const;

    QuadSetTopologyContainer(core::componentmodel::topology::BaseTopology *top,
            const sofa::helper::vector< unsigned int > &DOFIndex = (const sofa::helper::vector< unsigned int >)0,
            const sofa::helper::vector< Quad >         &quads    = (const sofa::helper::vector< Quad >)        0 );

    template< typename DataTypes >
    friend class QuadSetTopologyModifier;
private:
    /** \brief Returns a non-const quad vertex shell given a vertex index for subsequent modification
     *
     */
    sofa::helper::vector< unsigned int > &getQuadVertexShellForModification(const unsigned int vertexIndex);
    /** \brief Returns a non-const quad edge shell given the index of an edge for subsequent modification
     *
     */
    sofa::helper::vector< unsigned int > &getQuadEdgeShellForModification(const unsigned int edgeIndex);


};


template <class DataTypes>
class QuadSetTopologyLoader;

/**
 * A class that modifies the topology by adding and removing quads
 */
template<class DataTypes>
class QuadSetTopologyModifier : public EdgeSetTopologyModifier <DataTypes>
{

public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;

    QuadSetTopologyModifier(core::componentmodel::topology::BaseTopology *top) : EdgeSetTopologyModifier<DataTypes>(top)
    {
    }
    /** \brief Build a quad set topology from a file : also modifies the MechanicalObject
     *
     */
    virtual bool load(const char *filename);


    /** \brief Sends a message to warn that some quads were added in this topology.
     *
     * \sa addQuadsProcess
     */
    void addQuadsWarning(const unsigned int nQuads,
            const sofa::helper::vector< Quad >& quadsList,
            const sofa::helper::vector< unsigned int >& quadsIndexList,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors= (const sofa::helper::vector< sofa::helper::vector<unsigned int > >) 0 ,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs= (const sofa::helper::vector< sofa::helper::vector< double > >)0) ;

    void addEdgesWarning(const unsigned int nEdges,
            const sofa::helper::vector< Edge >& edgesList,
            const sofa::helper::vector< unsigned int >& edgesIndexList,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors= (const sofa::helper::vector< sofa::helper::vector<unsigned int > >) 0 ,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs= (const sofa::helper::vector< sofa::helper::vector< double > >)0)
    {
        EdgeSetTopologyModifier<DataTypes>::addEdgesWarning( nEdges,edgesList,edgesIndexList,ancestors,baryCoefs);
    }


    /** \brief Actually Add some quads to this topology.
     *
     * \sa addQuadsWarning
     */
    virtual void addQuadsProcess(const sofa::helper::vector< Quad > &quads);

    /** \brief Sends a message to warn that some quads are about to be deleted.
     *
     * \sa removeQuadsProcess
     *
     * Important : parameter indices is not const because it is actually sorted from the highest index to the lowest one.
     */
    virtual void removeQuadsWarning( sofa::helper::vector<unsigned int> &quads);

    /** \brief Remove a subset of  quads. Eventually remove isolated edges and vertices
     *
     * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
     * \sa removeQuadsWarning
     *
     * @param removeIsolatedItems if true isolated vertices are also removed
     */
    virtual void removeQuadsProcess( const sofa::helper::vector<unsigned int> &indices,const bool removeIsolatedItems=false);

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



    /** \brief Remove a subset of points
     *
     * Elements corresponding to these points are removed from the mechanical object's state vectors.
     *
     * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removePointsWarning before calling removePointsProcess.
     * \sa removePointsWarning
     * Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)
     */
    virtual void removePointsProcess(sofa::helper::vector<unsigned int> &indices, const bool removeDOF = true);



    /** \brief Reorder this topology.
     *
     * \see MechanicalObject::renumberValues
     */
    virtual void renumberPointsProcess( const sofa::helper::vector<unsigned int> &index );


protected:
    void addQuad(Quad e);

public:
    //template <class DataTypes>
    friend class QuadSetTopologyLoader<DataTypes>;

};



/**
 * A class that performs topology algorithms on an QuadSet.
 */
template < class DataTypes >
class QuadSetTopologyAlgorithms : public PointSetTopologyAlgorithms<DataTypes>
{

public:

    typedef typename DataTypes::Real Real;

    QuadSetTopologyAlgorithms(sofa::core::componentmodel::topology::BaseTopology *top) : PointSetTopologyAlgorithms<DataTypes>(top)
    {
    }

    /** \brief Remove a set  of quads
        @param quads an array of quad indices to be removed (note that the array is not const since it needs to be sorted)
        *
        */
    virtual void removeQuads(sofa::helper::vector< unsigned int >& quads);

};

/**
 * A class that provides geometry information on an QuadSet.
 */
template < class DataTypes >
class QuadSetGeometryAlgorithms : public EdgeSetGeometryAlgorithms<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;


    QuadSetGeometryAlgorithms(sofa::core::componentmodel::topology::BaseTopology *top) : EdgeSetGeometryAlgorithms<DataTypes>(top)
    {
    }
    /// computes the area of quad no i and returns it
    Real computeQuadArea(const unsigned int i) const;
    /// computes the quad area of all quads are store in the array interface
    void computeQuadArea( BasicArrayInterface<Real> &ai) const;
    /// computes the initial area  of quad no i and returns it
    Real computeRestQuadArea(const unsigned int i) const;

    // Computes the normal vector of a quad indexed by ind_q (not normed)
    Vec<3,double> computeQuadNormal(const unsigned int ind_q);

    // Test if a quad indexed by ind_q (and incident to the vertex indexed by ind_p) is included or not in the plane defined by (ind_p, plane_vect)
    bool is_quad_in_plane(const unsigned int ind_q, const unsigned int ind_p, const Vec<3,Real>& plane_vect);

};


/** Describes a topological object that only consists as a set of points :
it is a base class for all topological objects */
template<class DataTypes>
class QuadSetTopology : public EdgeSetTopology <DataTypes>
{

public:
    QuadSetTopology(component::MechanicalObject<DataTypes> *obj);


    virtual void init();
    /** \brief Returns the QuadSetTopologyContainer object of this QuadSetTopology.
     */
    QuadSetTopologyContainer *getQuadSetTopologyContainer() const
    {
        return (QuadSetTopologyContainer *)this->m_topologyContainer;
    }
    /** \brief Returns the QuadSetTopologyAlgorithms object of this QuadSetTopology.
     */
    QuadSetTopologyAlgorithms<DataTypes> *getQuadSetTopologyAlgorithms() const
    {
        return (QuadSetTopologyAlgorithms<DataTypes> *)this->m_topologyAlgorithms;
    }
    /** \brief Returns the QuadSetTopologyAlgorithms object of this QuadSetTopology.
     */
    QuadSetGeometryAlgorithms<DataTypes> *getQuadSetGeometryAlgorithms() const
    {
        return (QuadSetGeometryAlgorithms<DataTypes> *)this->m_geometryAlgorithms;
    }

};


} // namespace topology

} // namespace component

} // namespace sofa

#endif
