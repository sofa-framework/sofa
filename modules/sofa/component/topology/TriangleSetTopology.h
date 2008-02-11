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
#ifndef SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGY_H

#include <sofa/component/topology/EdgeSetTopology.h>
#include <vector>
#include <map>
#include <sofa/defaulttype/Vec.h> // typing "Vec"

//#include <sofa/component/mapping/Tetra2TriangleTopologicalMapping.h>

namespace sofa
{
namespace component
{

namespace topology
{
using namespace sofa::defaulttype; // typing "Vec"

/// defining Triangles as 3 DOFs indices
typedef helper::fixed_array<unsigned int,3> Triangle;
/// defining TriangleEdges as 3 Edge indices
typedef helper::fixed_array<unsigned int,3> TriangleEdges;

/////////////////////////////////////////////////////////
/// TopologyChange subclasses
/////////////////////////////////////////////////////////


template< class Real>
bool is_point_in_triangle(const Vec<3,Real>& p, const Vec<3,Real>& a, const Vec<3,Real>& b, const Vec<3,Real>& c);

template< class Real>
bool is_point_in_halfplane(const Vec<3,Real>& p, unsigned int e0, unsigned int e1, const Vec<3,Real>& a, const Vec<3,Real>& b, const Vec<3,Real>& c, unsigned int ind_p0, unsigned int ind_p1, unsigned int ind_p2);

void snapping_test_triangle(double epsilon, double alpha0, double alpha1, double alpha2, bool& is_snap_0, bool& is_snap_1, bool& is_snap_2);
void snapping_test_edge(double epsilon, double alpha0, double alpha1, bool& is_snap_0, bool& is_snap_1);

template< class Real>
inline Real areaProduct(const Vec<3,Real>& a, const Vec<3,Real>& b);

template< class Real>
inline Real areaProduct(const defaulttype::Vec<2,Real>& a, const defaulttype::Vec<2,Real>& b );

template< class Real>
inline Real areaProduct(const defaulttype::Vec<1,Real>& , const defaulttype::Vec<1,Real>&  );

/** indicates that some triangles were added */
class TrianglesAdded : public core::componentmodel::topology::TopologyChange
{

public:
    unsigned int nTriangles;

    sofa::helper::vector< Triangle > triangleArray;

    sofa::helper::vector< unsigned int > triangleIndexArray;

    sofa::helper::vector< sofa::helper::vector< unsigned int > > ancestorsList;

    sofa::helper::vector< sofa::helper::vector< double > > coefs;

    TrianglesAdded(const unsigned int nT,
            const sofa::helper::vector< Triangle >& _triangleArray = (const sofa::helper::vector< Triangle >)0,
            const sofa::helper::vector< unsigned int >& trianglesIndex = (const sofa::helper::vector< unsigned int >)0,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors = (const sofa::helper::vector< sofa::helper::vector< unsigned int > >)0,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs = (const sofa::helper::vector< sofa::helper::vector< double > >)0)
        : core::componentmodel::topology::TopologyChange(core::componentmodel::topology::TRIANGLESADDED), nTriangles(nT), triangleArray(_triangleArray), triangleIndexArray(trianglesIndex),ancestorsList(ancestors), coefs(baryCoefs)
    {   }

    unsigned int getNbAddedTriangles() const
    {
        return nTriangles;
    }

};



/** indicates that some triangles are about to be removed */
class TrianglesRemoved : public core::componentmodel::topology::TopologyChange
{

public:
    sofa::helper::vector<unsigned int> removedTrianglesArray;

public:
    TrianglesRemoved(const sofa::helper::vector<unsigned int> _tArray) : core::componentmodel::topology::TopologyChange(core::componentmodel::topology::TRIANGLESREMOVED), removedTrianglesArray(_tArray)
    {
    }

    const sofa::helper::vector<unsigned int> &getArray() const
    {
        return removedTrianglesArray;
    }
    unsigned int getNbRemovedTriangles() const
    {
        return removedTrianglesArray.size();
    }

};



/////////////////////////////////////////////////////////
/// TriangleSetTopology objects
/////////////////////////////////////////////////////////


/** Object that stores a set of triangles and provides access
to each triangle and its edges and vertices */
class TriangleSetTopologyContainer : public EdgeSetTopologyContainer
{
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
protected:
    /// provides the set of triangles
    sofa::helper::vector<Triangle> m_triangle;
    /// provides the 3 edges in each triangle
    sofa::helper::vector<TriangleEdges> m_triangleEdge;
    /// for each vertex provides the set of triangles adjacent to that vertex
    sofa::helper::vector< sofa::helper::vector< unsigned int > > m_triangleVertexShell;
    /// for each edge provides the set of triangles adjacent to that edge
    sofa::helper::vector< sofa::helper::vector< unsigned int > > m_triangleEdgeShell;


    /** \brief Creates the TriangleSet array.
     *
     * This function is only called by derived classes to create a list of triangles from a set of tetrahedra for instance
     */
    virtual void createTriangleSetArray() {}

    /** \brief Creates the EdgeSet array.
     *
     * Create the set of edges when needed.
     */
    virtual void createEdgeSetArray() {createTriangleEdgeArray();}

public:
    /** \brief Returns the Triangle array.
     *
     */
    const sofa::helper::vector<Triangle> &getTriangleArray();

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
    /** \brief Returns the ith Triangle.
     *
     */
    const Triangle &getTriangle(const unsigned int i);

    /** \brief Returns the number of triangles in this topology.
     *
     */
    unsigned int getNumberOfTriangles() ;

    /** \brief Returns the Triangle Vertex Shells array.
     *
     */
    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &getTriangleVertexShellArray() ;

    /** \brief Returns the set of triangles adjacent to a given vertex.
     *
     */
    const sofa::helper::vector< unsigned int > &getTriangleVertexShell(const unsigned int i) ;


    /** \brief Returns the TriangleEdges array (ie provide the 3 edge indices for each triangle)
     *
     */
    const sofa::helper::vector< TriangleEdges > &getTriangleEdgeArray() ;

    /** \brief Returns the 3 edges adjacent to a given triangle.
     *
     */
    const TriangleEdges &getTriangleEdge(const unsigned int i) ;

    /** \brief Returns the Triangle Edge Shells array (ie provides the triangles adjacent to each edge)
     *
     */
    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &getTriangleEdgeShellArray() ;

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

    /** \brief Checks if the Triangle Set Topology is coherent
     *
     * Check if the Edge and the Edhe Shell arrays are coherent
     */
    virtual bool checkTopology() const;

    TriangleSetTopologyContainer(core::componentmodel::topology::BaseTopology *top=NULL,
            const sofa::helper::vector< unsigned int > &DOFIndex = (const sofa::helper::vector< unsigned int >)0,
            const sofa::helper::vector< Triangle >         &triangles    = (const sofa::helper::vector< Triangle >)        0 );

    template< typename DataTypes >
    friend class TriangleSetTopologyModifier;

private:
    /** \brief Returns a non-const triangle vertex shell given a vertex index for subsequent modification
     *
     */
    sofa::helper::vector< unsigned int > &getTriangleVertexShellForModification(const unsigned int vertexIndex);
    /** \brief Returns a non-const triangle edge shell given the index of an edge for subsequent modification
     *
     */
    sofa::helper::vector< unsigned int > &getTriangleEdgeShellForModification(const unsigned int edgeIndex);


};


template <class DataTypes>
class TriangleSetTopologyLoader;

/**
 * A class that modifies the topology by adding and removing triangles
 */
template<class DataTypes>
class TriangleSetTopologyModifier : public EdgeSetTopologyModifier <DataTypes>
{

public:

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;

    TriangleSetTopologyModifier(core::componentmodel::topology::BaseTopology *top) : EdgeSetTopologyModifier<DataTypes>(top)
    {
    }
    /** \brief Build a triangle set topology from a file : also modifies the MechanicalObject
     *
     */
    virtual bool load(const char *filename);

    /*
    template< typename DataTypes >
      friend class TriangleSetTopologyAlgorithms;

    friend class sofa::core::componentmodel::topology::TopologicalMapping;

    template< typename In, typename Out >
      friend class Tetra2TriangleTopologicalMapping;
      */

    //protected:
    /** \brief Sends a message to warn that some triangles were added in this topology.
     *
     * \sa addTrianglesProcess
     */
    void addTrianglesWarning(const unsigned int nTriangles,
            const sofa::helper::vector< Triangle >& trianglesList,
            const sofa::helper::vector< unsigned int >& trianglesIndexList,
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
    virtual void removeTrianglesProcess( const sofa::helper::vector<unsigned int> &indices, const bool removeIsolatedEdges=false, const bool removeIsolatedPoints=false);

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


    virtual void addNewPoint( const sofa::helper::vector< double >& x) {EdgeSetTopologyModifier< DataTypes >::addNewPoint(x);};

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


    //protected:
    //public: // must actually be protected (has to be fixed)
    void addTriangle(Triangle e);

public:
    //template <class DataTypes>
    friend class TriangleSetTopologyLoader<DataTypes>;

};



/**
 * A class that performs topology algorithms on an TriangleSet.
 */
template < class DataTypes >
class TriangleSetTopologyAlgorithms : public EdgeSetTopologyAlgorithms<DataTypes>
{

public:

    typedef typename DataTypes::Real Real;

    TriangleSetTopologyAlgorithms(sofa::core::componentmodel::topology::BaseTopology *top) : EdgeSetTopologyAlgorithms<DataTypes>(top)
    {
    }

    /** \brief Remove a set  of triangles
        @param triangles an array of triangle indices to be removed (note that the array is not const since it needs to be sorted)
        *
    	@param removeIsolatedEdges if true isolated edges are also removed
        @param removeIsolatedPoints if true isolated vertices are also removed
    	*
        */
    virtual void removeTriangles(sofa::helper::vector< unsigned int >& triangles, const bool removeIsolatedEdges, const bool removeIsolatedPoints);

    // Prepares the incision along the list of points (ind_edge,coord) intersected by the vector from point a to point b
    // and the triangular mesh
    double Prepare_InciseAlongPointsList(const Vec<3,double>& a, const Vec<3,double>& b, const unsigned int ind_ta, const unsigned int ind_tb, unsigned int new_ind_ta, unsigned int newind_tb);

    // Incises along the list of points (ind_edge,coord) intersected by the vector from point a to point b
    // and the triangular mesh
    bool InciseAlongPointsList(bool is_first_cut, const Vec<3,double>& a, const Vec<3,double>& b, const unsigned int ind_ta, const unsigned int ind_tb,
            unsigned int& a_last, sofa::helper::vector< unsigned int > &a_p12_last, sofa::helper::vector< unsigned int > &a_i123_last,
            unsigned int& b_last, sofa::helper::vector< unsigned int > &b_p12_last, sofa::helper::vector< unsigned int > &b_i123_last,
            sofa::helper::vector< sofa::helper::vector<unsigned int> > &new_points, sofa::helper::vector< sofa::helper::vector<unsigned int> > &closest_vertices);

    // Removes triangles along the list of points (ind_edge,coord) intersected by the vector from point a to point b
    // and the triangular mesh
    void RemoveAlongTrianglesList(const Vec<3,double>& a, const Vec<3,double>& b, const unsigned int ind_ta, const unsigned int ind_tb);

    // Incises along the list of points (ind_edge,coord) intersected by the sequence of input segments (list of input points) and the triangular mesh
    void InciseAlongLinesList(const sofa::helper::vector< Vec<3,double> >& input_points, const sofa::helper::vector< unsigned int > &input_triangles);

    /** \brief Duplicate the given edge. Only works of at least one of its points is adjacent to a border.
     * @returns the number of newly created points, or -1 if the incision failed.
     */
    virtual int InciseAlongEdge(unsigned int edge);

};

/**
 * A class that provides geometry information on an TriangleSet.
 */
template < class DataTypes >
class TriangleSetGeometryAlgorithms : public EdgeSetGeometryAlgorithms<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;


    TriangleSetGeometryAlgorithms(sofa::core::componentmodel::topology::BaseTopology *top) : EdgeSetGeometryAlgorithms<DataTypes>(top)
    {
    }

    const Coord& getPositionPoint(unsigned int i)
    {
        TriangleSetTopology< DataTypes > *topology = static_cast<TriangleSetTopology< DataTypes >* >(this->m_basicTopology);
        const VecCoord& vect_c = *topology->getDOF()->getX();
        return vect_c[i];
    };

    /// computes the area of triangle no i and returns it
    Real computeTriangleArea(const unsigned int i) const;
    /// computes the triangle area of all triangles are store in the array interface
    void computeTriangleArea( BasicArrayInterface<Real> &ai) const;
    /// computes the initial area  of triangle no i and returns it
    Real computeRestTriangleArea(const unsigned int i) const;

    // barycentric coefficients of point p in triangle (a,b,c) indexed by ind_t
    sofa::helper::vector< double > computeTriangleBarycoefs( const Vec<3,double> &p, unsigned int ind_t);

    // barycentric coefficients of point p in triangle whose vertices are indexed by (ind_p1,ind_p2,ind_p3)
    sofa::helper::vector< double > compute3PointsBarycoefs( const Vec<3,double> &p, unsigned int ind_p1, unsigned int ind_p2, unsigned int ind_p3);

    // test if a point is included in the triangle indexed by ind_t
    bool is_PointinTriangle(bool is_tested, const Vec<3,Real>& p, unsigned int ind_t, unsigned int &ind_t_test);

    // Computes the point defined by 2 indices of vertex and 1 barycentric coordinate
    Vec<3,double> computeBaryEdgePoint(sofa::helper::vector< unsigned int>& indices, const double &coord_p);

    // Computes the normal vector of a triangle indexed by ind_t (not normed)
    Vec<3,double> computeTriangleNormal(const unsigned int ind_t);

    // Tests how to triangularize a quad whose vertices are defined by (p_q1, p_q2, ind_q3, ind_q4) according to the Delaunay criterion
    bool isQuadDeulaunayOriented(const Vec<3,double>& p_q1, const Vec<3,double>& p_q2, unsigned int ind_q3, unsigned int ind_q4);

    // Computes the opposite point to ind_p
    Vec<3,double> getOppositePoint(unsigned int ind_p, sofa::helper::vector< unsigned int>& indices, const double &coord_p);

    // Test if a triangle indexed by ind_t (and incident to the vertex indexed by ind_p) is included or not in the plane defined by (ind_p, plane_vect)
    bool is_triangle_in_plane(const unsigned int ind_t, const unsigned int ind_p, const Vec<3,Real>& plane_vect);


    // Prepares the duplication of a vertex
    void Prepare_VertexDuplication(const unsigned int ind_p, const unsigned int ind_t_from, const unsigned int ind_t_to,
            const sofa::helper::vector< unsigned int>& indices_from, const double &coord_from, const sofa::helper::vector< unsigned int>& indices_to, const double &coord_to,
            sofa::helper::vector< unsigned int > &triangles_list_1, sofa::helper::vector< unsigned int > &triangles_list_2);



    // Computes the intersection of the vector from point a to point b and the triangle indexed by t
    bool computeSegmentTriangleIntersection(bool is_entered, const Vec<3,double>& a, const Vec<3,double>& b, const unsigned int ind_t,
            sofa::helper::vector<unsigned int> &indices,
            double &baryCoef, double& coord_kmin);

    // Computes the list of points (ind_edge,coord) intersected by the segment from point a to point b
    // and the triangular mesh
    bool computeIntersectedPointsList(const Vec<3,double>& a, const Vec<3,double>& b, const unsigned int ind_ta, unsigned int& ind_tb,
            sofa::helper::vector< unsigned int > &triangles_list, sofa::helper::vector< sofa::helper::vector< unsigned int> > &indices_list, sofa::helper::vector< double >& coords_list, bool& is_on_boundary);
};


/** Describes a topological object that only consists as a set of points :
it is a base class for all topological objects */
template<class DataTypes>
class TriangleSetTopology : public EdgeSetTopology <DataTypes>
{

public:
    TriangleSetTopology(component::MechanicalObject<DataTypes> *obj);
    DataPtr< TriangleSetTopologyContainer > *f_m_topologyContainer;

    virtual void init();
    /** \brief Returns the TriangleSetTopologyContainer object of this TriangleSetTopology.
     */
    TriangleSetTopologyContainer *getTriangleSetTopologyContainer() const
    {
        return (TriangleSetTopologyContainer *)this->m_topologyContainer;
    }
    /** \brief Returns the TriangleSetTopologyAlgorithms object of this TriangleSetTopology.
     */
    TriangleSetTopologyAlgorithms<DataTypes> *getTriangleSetTopologyAlgorithms() const
    {
        return (TriangleSetTopologyAlgorithms<DataTypes> *)this->m_topologyAlgorithms;
    }
    /** \brief Returns the TriangleSetTopologyAlgorithms object of this TriangleSetTopology.
     */
    TriangleSetGeometryAlgorithms<DataTypes> *getTriangleSetGeometryAlgorithms() const
    {
        return (TriangleSetGeometryAlgorithms<DataTypes> *)this->m_geometryAlgorithms;
    }

};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
