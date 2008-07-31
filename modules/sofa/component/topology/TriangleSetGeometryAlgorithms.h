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
#ifndef SOFA_COMPONENT_TOPOLOGY_TRIANGLESETGEOMETRYALGORITHMS_H
#define SOFA_COMPONENT_TOPOLOGY_TRIANGLESETGEOMETRYALGORITHMS_H

#include <sofa/component/topology/EdgeSetGeometryAlgorithms.h>

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
* A class that provides geometry information on an TriangleSet.
*/
template < class DataTypes >
class TriangleSetGeometryAlgorithms : public EdgeSetGeometryAlgorithms<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;

    TriangleSetGeometryAlgorithms()
        : EdgeSetGeometryAlgorithms<DataTypes>()
    {}

    TriangleSetGeometryAlgorithms(sofa::core::componentmodel::topology::BaseTopology *top)
        : EdgeSetGeometryAlgorithms<DataTypes>(top)
    {}

    virtual ~TriangleSetGeometryAlgorithms() {}

    TriangleSetTopology< DataTypes >* getTriangleSetTopology() const;

    /** \brief Returns spatial position of point indexed by i
    *
    */
    const Coord& getPositionPoint(unsigned int i);

    /** \brief Computes the area of triangle no i and returns it
    *
    */
    Real computeTriangleArea(const unsigned int i) const;

    /** \brief Computes the triangle area of all triangles are store in the array interface
    *
    */
    void computeTriangleArea( BasicArrayInterface<Real> &ai) const;

    /** \brief Computes the initial area  of triangle no i and returns it
    *
    */
    Real computeRestTriangleArea(const unsigned int i) const;

    /** \brief Computes barycentric coefficients of point p in triangle (a,b,c) indexed by ind_t
    *
    */
    sofa::helper::vector< double > computeTriangleBarycoefs( const sofa::defaulttype::Vec<3,double> &p, unsigned int ind_t);

    /** \brief Computes barycentric coefficients of point p in triangle whose vertices are indexed by (ind_p1,ind_p2,ind_p3)
    *
    */
    sofa::helper::vector< double > compute3PointsBarycoefs( const sofa::defaulttype::Vec<3,double> &p,
            unsigned int ind_p1,
            unsigned int ind_p2,
            unsigned int ind_p3);

    /** \brief Finds the two closest points from two triangles (each of the point belonging to one triangle)
    *
    */
    void closestIndexPair(unsigned int ind_ta, unsigned int ind_tb, unsigned int &ind1, unsigned int &ind2);

    /** \brief Tests if a point is included in the triangle indexed by ind_t
    *
    */
    bool is_PointinTriangle(bool is_tested, const sofa::defaulttype::Vec<3,Real>& p, unsigned int ind_t, unsigned int &ind_t_test);

    /** \brief Computes the point defined by 2 indices of vertex and 1 barycentric coordinate
    *
    */
    sofa::defaulttype::Vec<3,double> computeBaryEdgePoint(sofa::helper::vector< unsigned int>& indices, const double &coord_p);

    /** \brief Computes the normal vector of a triangle indexed by ind_t (not normed)
    *
    */
    sofa::defaulttype::Vec<3,double> computeTriangleNormal(const unsigned int ind_t);

    /** \brief Tests how to triangularize a quad whose vertices are defined by (p_q1, p_q2, ind_q3, ind_q4) according to the Delaunay criterion
    *
    */
    bool isQuadDeulaunayOriented(const sofa::defaulttype::Vec<3,double>& p_q1,
            const sofa::defaulttype::Vec<3,double>& p_q2,
            unsigned int ind_q3, unsigned int ind_q4);

    /** \brief Computes the opposite point to ind_p
    *
    */
    sofa::defaulttype::Vec<3,double> getOppositePoint(unsigned int ind_p, sofa::helper::vector< unsigned int>& indices, const double &coord_p);

    /** \brief Tests if a triangle indexed by ind_t (and incident to the vertex indexed by ind_p) is included or not in the plane defined by (ind_p, plane_vect)
    *
    */
    bool is_triangle_in_plane(const unsigned int ind_t, const unsigned int ind_p,
            const sofa::defaulttype::Vec<3,Real>& plane_vect);

    /** \brief Prepares the duplication of a vertex
    *
    */
    void Prepare_VertexDuplication(const unsigned int ind_p,
            const unsigned int ind_t_from, const unsigned int ind_t_to,
            const sofa::helper::vector< unsigned int>& indices_from, const double &coord_from,
            const sofa::helper::vector< unsigned int>& indices_to, const double &coord_to,
            sofa::helper::vector< unsigned int > &triangles_list_1,
            sofa::helper::vector< unsigned int > &triangles_list_2);

    /** \brief Computes the intersection of the vector from point a to point b and the triangle indexed by t
    *
    */
    bool computeSegmentTriangleIntersection(bool is_entered,
            const sofa::defaulttype::Vec<3,double>& a,
            const sofa::defaulttype::Vec<3,double>& b,
            const unsigned int ind_t,
            sofa::helper::vector<unsigned int> &indices,
            double &baryCoef, double& coord_kmin);

    /** \brief Computes the list of points (ind_edge,coord) intersected by the segment from point a to point b and the triangular mesh
    *
    */
    bool computeIntersectedPointsList(const sofa::defaulttype::Vec<3,double>& a,
            const sofa::defaulttype::Vec<3,double>& b,
            const unsigned int ind_ta, unsigned int& ind_tb,
            sofa::helper::vector< unsigned int > &triangles_list,
            sofa::helper::vector< sofa::helper::vector< unsigned int> > &indices_list,
            sofa::helper::vector< double >& coords_list,
            bool& is_on_boundary);

    /** \brief Write the current mesh into a msh file
    */
    void writeMSHfile(const char *filename);
};


template< class Real>
bool is_point_in_triangle(const sofa::defaulttype::Vec<3,Real>& p,
        const sofa::defaulttype::Vec<3,Real>& a,
        const sofa::defaulttype::Vec<3,Real>& b,
        const sofa::defaulttype::Vec<3,Real>& c);

template< class Real>
bool is_point_in_halfplane(const sofa::defaulttype::Vec<3,Real>& p,
        unsigned int e0, unsigned int e1,
        const sofa::defaulttype::Vec<3,Real>& a,
        const sofa::defaulttype::Vec<3,Real>& b,
        const sofa::defaulttype::Vec<3,Real>& c,
        unsigned int ind_p0, unsigned int ind_p1, unsigned int ind_p2);

void snapping_test_triangle(double epsilon,
        double alpha0, double alpha1, double alpha2,
        bool& is_snap_0, bool& is_snap_1, bool& is_snap_2);

void snapping_test_edge(double epsilon,
        double alpha0, double alpha1,
        bool& is_snap_0, bool& is_snap_1);

template< class Real>
inline Real areaProduct(const sofa::defaulttype::Vec<3,Real>& a, const sofa::defaulttype::Vec<3,Real>& b);

template< class Real>
inline Real areaProduct(const defaulttype::Vec<2,Real>& a, const defaulttype::Vec<2,Real>& b );

template< class Real>
inline Real areaProduct(const defaulttype::Vec<1,Real>& , const defaulttype::Vec<1,Real>&  );

} // namespace topology

} // namespace component

} // namespace sofa

#endif
