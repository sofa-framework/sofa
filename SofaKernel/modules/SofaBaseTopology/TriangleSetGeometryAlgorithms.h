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
#ifndef SOFA_COMPONENT_TOPOLOGY_TRIANGLESETGEOMETRYALGORITHMS_H
#define SOFA_COMPONENT_TOPOLOGY_TRIANGLESETGEOMETRYALGORITHMS_H
#include "config.h"

#include <SofaBaseTopology/EdgeSetGeometryAlgorithms.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace component
{

namespace topology
{

/**
* A class that provides geometry information on an TriangleSet.
*/
template < class DataTypes >
class TriangleSetGeometryAlgorithms : public EdgeSetGeometryAlgorithms<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TriangleSetGeometryAlgorithms,DataTypes), SOFA_TEMPLATE(EdgeSetGeometryAlgorithms,DataTypes));


    typedef sofa::core::topology::BaseMeshTopology::PointID PointID;
    typedef sofa::core::topology::BaseMeshTopology::EdgeID EdgeID;
    typedef sofa::core::topology::BaseMeshTopology::Edge Edge;
    typedef sofa::core::topology::BaseMeshTopology::SeqEdges SeqEdges;
    typedef sofa::core::topology::BaseMeshTopology::EdgesAroundVertex EdgesAroundVertex;

    typedef core::topology::BaseMeshTopology::TriangleID TriangleID;
    typedef core::topology::BaseMeshTopology::Triangle Triangle;
    typedef core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
    typedef core::topology::BaseMeshTopology::TrianglesAroundVertex TrianglesAroundVertex;
    typedef core::topology::BaseMeshTopology::TrianglesAroundEdge TrianglesAroundEdge;
    typedef core::topology::BaseMeshTopology::EdgesInTriangle EdgesInTriangle;

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;

protected:
    bool initializedCubatureTables;
    void defineTetrahedronCubaturePoints();
    TriangleSetGeometryAlgorithms()
        : EdgeSetGeometryAlgorithms<DataTypes>()
        ,initializedCubatureTables(false)
        ,showTriangleIndices (initData(&showTriangleIndices, (bool) false, "showTriangleIndices", "Debug : view Triangle indices"))
        , _draw(initData(&_draw, false, "drawTriangles","if true, draw the triangles in the topology"))
        , _drawColor(initData(&_drawColor, sofa::defaulttype::Vec4f(0.2f,1.0f,1.0f,1.0f), "drawColorTriangles", "RGBA code color used to draw edges."))
        , _drawNormals(initData(&_drawNormals, false, "drawNormals","if true, draw the triangles in the topology"))
        , _drawNormalLength (initData(&_drawNormalLength, (SReal)10, "drawNormalLength", "Fiber length visualisation."))
        , p_recomputeTrianglesOrientation(initData(&p_recomputeTrianglesOrientation, false, "recomputeTrianglesOrientation","if true, will recompute triangles orientation according to normals."))
        , p_flipNormals(initData(&p_flipNormals, false, "flipNormals","if true, will flip normal of the first triangle used to recompute triangle orientation."))
    {
    }

    virtual ~TriangleSetGeometryAlgorithms() {}
public:
    virtual void draw(const core::visual::VisualParams* vparams) override;

    virtual void init() override;

    virtual void reinit() override;

    void computeTriangleAABB(const TriangleID i, Coord& minCoord, Coord& maxCoord) const;

    Coord computeTriangleCenter(const TriangleID i) const;

    Coord computeRestTriangleCenter(const TriangleID i) const;

    void computeTriangleCircumcenterBaryCoefs(sofa::defaulttype::Vec<3,Real> &baryCoord, const TriangleID i) const;

    Coord computeTriangleCircumcenter(const TriangleID i) const;

    void getTriangleVertexCoordinates(const TriangleID i, Coord[3]) const;

    void getRestTriangleVertexCoordinates(const TriangleID i, Coord[3]) const;

    /** \brief Computes the area of triangle no i and returns it
     *
     */
    Real computeTriangleArea(const TriangleID i) const;

    /** \brief Computes the triangle area of all triangles are store in the array interface
    *
    */
    void computeTriangleArea( BasicArrayInterface<Real> &ai) const;

    /** \brief Computes the initial area  of triangle no i and returns it
     *
     */
    Real computeRestTriangleArea(const TriangleID i) const;

    /** \brief Computes barycentric coefficients of point p in triangle (a,b,c) indexed by ind_t
    *
    */
    sofa::helper::vector< double > computeTriangleBarycoefs(const TriangleID ind_t, const sofa::defaulttype::Vec<3,double> &p) const;

    /** \brief Computes barycentric coefficients of point p in triangle whose vertices are indexed by (ind_p1,ind_p2,ind_p3)
     *
     */
    sofa::helper::vector< double > compute3PointsBarycoefs( const sofa::defaulttype::Vec<3,double> &p,
            unsigned int ind_p1,
            unsigned int ind_p2,
            unsigned int ind_p3,
            bool bRest=false) const;

    /** \brief Finds the two closest points from two triangles (each of the point belonging to one triangle)
     *
     */
    void computeClosestIndexPair(const TriangleID ind_ta, const TriangleID ind_tb,
            unsigned int &ind1, unsigned int &ind2) const;

    /** \brief Tests if a point is included in the triangle indexed by ind_t
     *
     */
    bool isPointInsideTriangle(const TriangleID ind_t, bool is_tested, const sofa::defaulttype::Vec<3,Real>& p, unsigned int &ind_t_test, bool bRest=false) const;

    bool isPointInTriangle(const TriangleID ind_t, bool is_tested, const sofa::defaulttype::Vec<3,Real>& p, unsigned int &ind_t_test) const;



    /** \brief Computes the point defined by 2 indices of vertex and 1 barycentric coordinate
     *
     */
    sofa::defaulttype::Vec<3,double> computeBaryEdgePoint(unsigned int p0, unsigned int p1, double coord_p) const;
    sofa::defaulttype::Vec<3,double> computeBaryEdgePoint(Edge e, double coord_p) const
    {
        return computeBaryEdgePoint(e[0], e[1], coord_p);
    }

    /** \brief Computes the point defined by 3 indices of vertex and 1 barycentric coordinate
     *
     */
    sofa::defaulttype::Vec<3,double> computeBaryTrianglePoint(unsigned int p0, unsigned int p1, unsigned int p2, sofa::defaulttype::Vec<3,double>& coord_p) const;
    sofa::defaulttype::Vec<3,double> computeBaryTrianglePoint(Triangle& t, sofa::defaulttype::Vec<3,double>& coord_p) const
    {
        return computeBaryTrianglePoint(t[0], t[1], t[2], coord_p);
    }


    /** \brief Computes the normal vector of a triangle indexed by ind_t (not normed)
     *
     */
    sofa::defaulttype::Vec<3,double> computeTriangleNormal(const TriangleID ind_t) const;

    /** \brief Tests how to triangularize a quad whose vertices are defined by (p_q1, p_q2, ind_q3, ind_q4) according to the Delaunay criterion
    *
    */
    bool isQuadDeulaunayOriented(const typename DataTypes::Coord& p_q1,
            const typename DataTypes::Coord& p_q2,
            unsigned int ind_q3, unsigned int ind_q4);

    /** \brief Tests how to triangularize a quad whose vertices are defined by (p1, p2, p3, p4) according to the Delaunay criterion
     *
     */
    bool isQuadDeulaunayOriented(const typename DataTypes::Coord& p1,
            const typename DataTypes::Coord& p2,
            const typename DataTypes::Coord& p3,
            const typename DataTypes::Coord& p4);


    /** \brief Given two triangles, test if the intersection of the diagonals of the quad composed
     * by the triangles is inside the quad or not. (test if triangles form a quad)
     * @param triangle1 Coord tab of the 3 vertices composing the first triangle.
     * @param triangle2 same for second triangle.
     * @return bool, true if the intersection point is inside. Else (not inside or no intersection) false.
     */
    bool isDiagonalsIntersectionInQuad (const typename DataTypes::Coord triangle1[3], const typename DataTypes::Coord triangle2[3]);


    /** \brief Computes the opposite point to ind_p
     *
     */
    sofa::defaulttype::Vec<3,double> getOppositePoint(unsigned int ind_p, const Edge& indices, double coord_p) const;

    /** \brief Tests if a triangle indexed by ind_t (and incident to the vertex indexed by ind_p) is included or not in the plane defined by (ind_p, plane_vect)
     *
    */
    bool isTriangleInPlane(const TriangleID ind_t, const unsigned int ind_p,
            const sofa::defaulttype::Vec<3,Real>& plane_vect) const;

    /** \brief Prepares the duplication of a vertex
     *
     */
    void prepareVertexDuplication(const unsigned int ind_p,
            const TriangleID ind_t_from, const TriangleID ind_t_to,
            const Edge& indices_from, const double &coord_from,
            const Edge& indices_to, const double &coord_to,
            sofa::helper::vector< unsigned int > &triangles_list_1,
            sofa::helper::vector< unsigned int > &triangles_list_2) const;

    /** \brief Computes the intersection of the vector from point a to point b and the triangle indexed by t
     *
     * @param a : first input point
     * @param b : last input point
     * @param ind_t : triangle indice
     * @param indices : vertex indices of edge (belonging to ind_t) crossed by the vecteur AB
     * @param baryCoef : barycoef of the intersection point on the edge
     * @param coord_kmin : barycoef of the intersection point on the vecteur AB.
     */
    bool computeSegmentTriangleIntersection(bool is_entered,
            const sofa::defaulttype::Vec<3,double>& a,
            const sofa::defaulttype::Vec<3,double>& b,
            const TriangleID ind_t,
            sofa::helper::vector<unsigned int> &indices,
            double &baryCoef, double& coord_kmin) const;

    /** \brief Computes the list of points (ind_edge,coord) intersected by the segment from point a to point b and the triangular mesh
     *
     */
    bool computeIntersectedPointsList(const PointID last_point,
            const sofa::defaulttype::Vec<3,double>& a,
            const sofa::defaulttype::Vec<3,double>& b,
            unsigned int& ind_ta, unsigned int& ind_tb,
            sofa::helper::vector< unsigned int > &triangles_list,
            sofa::helper::vector< unsigned int > &edges_list,
            sofa::helper::vector< double >& coords_list,
            bool& is_on_boundary) const;

    /** \brief Computes the list of objects (points, edges, triangles) intersected by the segment from point a to point b and the triangular mesh.
     *
     * @return List of object intersect (type enum @see core::topology::TopologyObjectType)
     * @return List of indices of these objetcs
     * @return List of barycentric coordinate defining the position of the intersection in each object
     * (i.e 0 coord for a point, 1 for and edge and 3 for a triangle).
     */
    bool computeIntersectedObjectsList (const PointID last_point,
            const sofa::defaulttype::Vec<3,double>& a, const sofa::defaulttype::Vec<3,double>& b,
            unsigned int& ind_ta, unsigned int& ind_tb,
            sofa::helper::vector< sofa::core::topology::TopologyObjectType>& topoPath_list,
            sofa::helper::vector<unsigned int>& indices_list,
            sofa::helper::vector< sofa::defaulttype::Vec<3, double> >& coords_list) const;


    /** \brief Get the triangle in a given direction from a point.
     */
    int getTriangleInDirection(PointID p, const sofa::defaulttype::Vec<3,double>& dir) const;


    /** \brief Write the current mesh into a msh file
     */
    void writeMSHfile(const char *filename) const;

    /** \brief This function will changed vertex index in triangles if normal from one to another triangle are in opposite direction.
      First triangle index is used as ground truth. Use option flipNormals if first triangle direction is wrong.
      */
    void reorderTrianglesOrientationFromNormals();

    /** \brief Process the added point initialization according to the topology and local coordinates.
    */
    virtual void initPointAdded(unsigned int indice, const core::topology::PointAncestorElem &ancestorElem
        , const helper::vector< VecCoord* >& coordVecs, const helper::vector< VecDeriv* >& derivVecs) override;

    /// return a pointer to the container of cubature points
    NumericalIntegrationDescriptor<Real,3> &getTriangleNumericalIntegrationDescriptor();
protected:
    Data<bool> showTriangleIndices;
    Data<bool> _draw;
    Data<sofa::defaulttype::Vec4f> _drawColor;
    Data<bool> _drawNormals;
    Data <SReal> _drawNormalLength;
    Data<bool> p_recomputeTrianglesOrientation;
    Data<bool> p_flipNormals;
    /// include cubature points
    NumericalIntegrationDescriptor<Real,3> triangleNumericalIntegration;
};


/*template< class Real>
  bool is_point_in_triangle(const sofa::defaulttype::Vec<3,Real>& p,
  const sofa::defaulttype::Vec<3,Real>& a,
  const sofa::defaulttype::Vec<3,Real>& b,
  const sofa::defaulttype::Vec<3,Real>& c);*/
template<class Real>
bool is_point_in_triangle(const sofa::defaulttype::Vec<3,Real>& p,
        const sofa::defaulttype::Vec<3,Real>& a, const sofa::defaulttype::Vec<3,Real>& b, const sofa::defaulttype::Vec<3,Real>& c);


template< class Real>
bool is_point_in_halfplane(const sofa::defaulttype::Vec<3,Real>& p,
        unsigned int e0, unsigned int e1,
        const sofa::defaulttype::Vec<3,Real>& a,
        const sofa::defaulttype::Vec<3,Real>& b,
        const sofa::defaulttype::Vec<3,Real>& c,
        unsigned int ind_p0, unsigned int ind_p1, unsigned int ind_p2);

void SOFA_BASE_TOPOLOGY_API snapping_test_triangle(double epsilon,
        double alpha0, double alpha1, double alpha2,
        bool& is_snap_0, bool& is_snap_1, bool& is_snap_2);

void SOFA_BASE_TOPOLOGY_API snapping_test_edge(double epsilon,
        double alpha0, double alpha1,
        bool& is_snap_0, bool& is_snap_1);

template< class Real>
inline Real areaProduct(const sofa::defaulttype::Vec<3,Real>& a, const sofa::defaulttype::Vec<3,Real>& b);

template< class Real>
inline Real areaProduct(const defaulttype::Vec<2,Real>& a, const defaulttype::Vec<2,Real>& b );

template< class Real>
inline Real areaProduct(const defaulttype::Vec<1,Real>& , const defaulttype::Vec<1,Real>&  );

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_TOPOLOGY_TRIANGLESETGEOMETRYALGORITHMS_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BASE_TOPOLOGY_API TriangleSetGeometryAlgorithms<defaulttype::Vec3dTypes>;
extern template class SOFA_BASE_TOPOLOGY_API TriangleSetGeometryAlgorithms<defaulttype::Vec2dTypes>;
extern template class SOFA_BASE_TOPOLOGY_API TriangleSetGeometryAlgorithms<defaulttype::Vec1dTypes>;
//extern template class SOFA_BASE_TOPOLOGY_API TriangleSetGeometryAlgorithms<defaulttype::Rigid3dTypes>;
//extern template class SOFA_BASE_TOPOLOGY_API TriangleSetGeometryAlgorithms<defaulttype::Rigid2dTypes>;
#endif

#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_TOPOLOGY_API TriangleSetGeometryAlgorithms<defaulttype::Vec3fTypes>;
extern template class SOFA_BASE_TOPOLOGY_API TriangleSetGeometryAlgorithms<defaulttype::Vec2fTypes>;
extern template class SOFA_BASE_TOPOLOGY_API TriangleSetGeometryAlgorithms<defaulttype::Vec1fTypes>;
//extern template class SOFA_BASE_TOPOLOGY_API TriangleSetGeometryAlgorithms<defaulttype::Rigid3fTypes>;
//extern template class SOFA_BASE_TOPOLOGY_API TriangleSetGeometryAlgorithms<defaulttype::Rigid2fTypes>;
#endif
#endif

} // namespace topology

} // namespace component

} // namespace sofa

#endif
