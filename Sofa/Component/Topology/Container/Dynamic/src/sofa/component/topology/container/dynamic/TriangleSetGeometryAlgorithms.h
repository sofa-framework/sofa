/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/component/topology/container/dynamic/config.h>

#include <sofa/component/topology/container/dynamic/EdgeSetGeometryAlgorithms.h>
#include <sofa/type/Vec.h>

namespace sofa::component::topology::container::dynamic
{

class TriangleSetTopologyContainer;

class TriangleSetTopologyModifier;

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
    typedef sofa::core::topology::BaseMeshTopology::QuadID QuadID;
    typedef sofa::core::topology::BaseMeshTopology::ElemID ElemID;
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

    using Vec3 = sofa::type::Vec<3, Real>;

protected:
    bool initializedCubatureTables;
    void defineTetrahedronCubaturePoints();
    TriangleSetGeometryAlgorithms()
        : EdgeSetGeometryAlgorithms<DataTypes>()
        ,initializedCubatureTables(false)
        ,showTriangleIndices (initData(&showTriangleIndices, (bool) false, "showTriangleIndices", "Debug : view Triangle indices"))
        , _draw(initData(&_draw, false, "drawTriangles","if true, draw the triangles in the topology"))
        , _drawColor(initData(&_drawColor, sofa::type::RGBAColor(0.3f,0.5f,0.8f,1.0f), "drawColorTriangles", "RGBA code color used to draw triangles"))
        , _drawNormals(initData(&_drawNormals, false, "drawNormals","if true, draw the triangles in the topology"))
        , _drawNormalLength (initData(&_drawNormalLength, (SReal)10, "drawNormalLength", "Fiber length visualisation."))
        , p_recomputeTrianglesOrientation(initData(&p_recomputeTrianglesOrientation, false, "recomputeTrianglesOrientation","if true, will recompute triangles orientation according to normals"))
        , p_flipNormals(initData(&p_flipNormals, false, "flipNormals","if true, will flip normal of the first triangle used to recompute triangle orientation."))
    {
    }

    virtual ~TriangleSetGeometryAlgorithms() {}
public:
    void draw(const core::visual::VisualParams* vparams) override;

    void init() override;

    void reinit() override;

    void computeTriangleAABB(const TriangleID i, Coord& minCoord, Coord& maxCoord) const;

    Coord computeTriangleCenter(const TriangleID i) const;

    Coord computeRestTriangleCenter(const TriangleID i) const;

    void computeTriangleCircumcenterBaryCoefs(sofa::type::Vec<3,Real> &baryCoord, const TriangleID i) const;

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
    sofa::type::vector< SReal > computeTriangleBarycoefs(const TriangleID ind_t, const sofa::type::Vec<3,Real> &p) const;

    /** \brief Computes barycentric coefficients of point p in initial triangle (a,b,c) indexed by ind_t
    *
    */
    sofa::type::vector< SReal > computeRestTriangleBarycoefs(const TriangleID ind_t, const sofa::type::Vec<3, Real>& p) const;

    /** \brief Computes barycentric coefficients of point p in triangle whose vertices are indexed by (ind_p1,ind_p2,ind_p3)
     *
     */
    sofa::type::vector< SReal > compute3PointsBarycoefs( const sofa::type::Vec<3, Real> &p,
            PointID ind_p1,
            PointID ind_p2,
            PointID ind_p3,
            bool bRest=false) const;

    /** \brief Finds the two closest points from two triangles (each of the point belonging to one triangle)
     *
     */
    void computeClosestIndexPair(const TriangleID ind_ta, const TriangleID ind_tb,
        PointID &ind1, PointID &ind2) const;

    /** \brief Tests if a point is included in the triangle indexed by ind_t
     *
     */
    bool isPointInsideTriangle(const TriangleID ind_t, bool is_tested, const sofa::type::Vec<3,Real>& p, TriangleID &ind_t_test, bool bRest=false) const;

    bool isPointInTriangle(const TriangleID ind_t, bool is_tested, const sofa::type::Vec<3,Real>& p, TriangleID &ind_t_test) const;



    /** \brief Computes the point defined by 2 indices of vertex and 1 barycentric coordinate
     *
     */
    sofa::type::Vec<3, Real> computeBaryEdgePoint(PointID p0, PointID p1, Real coord_p) const;
    sofa::type::Vec<3, Real> computeBaryEdgePoint(Edge e, Real coord_p) const
    {
        return computeBaryEdgePoint(e[0], e[1], coord_p);
    }

    /** \brief Computes the point defined by 3 indices of vertex and 1 barycentric coordinate
     *
     */
    sofa::type::Vec<3, Real> computeBaryTrianglePoint(PointID p0, PointID p1, PointID p2, sofa::type::Vec<3, Real>& coord_p) const;
    sofa::type::Vec<3, Real> computeBaryTrianglePoint(Triangle& t, sofa::type::Vec<3, Real>& coord_p) const
    {
        return computeBaryTrianglePoint(t[0], t[1], t[2], coord_p);
    }


    /** \brief Computes the normal vector of a triangle indexed by ind_t (not normed)
     *
     */
    sofa::type::Vec<3, Real> computeTriangleNormal(const TriangleID ind_t) const;

    /** \brief Tests how to triangularize a quad whose vertices are defined by (p_q1, p_q2, ind_q3, ind_q4) according to the Delaunay criterion
    *
    */
    bool isQuadDeulaunayOriented(const typename DataTypes::Coord& p_q1,
            const typename DataTypes::Coord& p_q2,
            QuadID ind_q3, QuadID ind_q4);

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
    sofa::type::Vec<3, Real> getOppositePoint(PointID ind_p, const Edge& indices, Real coord_p) const;

    /** \brief Tests if a triangle indexed by ind_t (and incident to the vertex indexed by ind_p) is included or not in the plane defined by (ind_p, plane_vect)
     *
    */
    bool isTriangleInPlane(const TriangleID ind_t, const PointID ind_p,
            const sofa::type::Vec<3,Real>& plane_vect) const;

    /** \brief Prepares the duplication of a vertex
     *
     */
    void prepareVertexDuplication(const PointID ind_p,
            const TriangleID ind_t_from, const TriangleID ind_t_to,
            const Edge& indices_from, const Real& coord_from,
            const Edge& indices_to, const Real& coord_to,
            sofa::type::vector< TriangleID > &triangles_list_1,
            sofa::type::vector< TriangleID > &triangles_list_2) const;

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
        const sofa::type::Vec<3, Real>& a,
        const sofa::type::Vec<3, Real>& b,
        const TriangleID ind_t,
        sofa::type::vector<PointID>& indices,
        Real& baryCoef, Real& coord_kmin) const;

    /** \brief Computes the intersection between the edges of the Triangle triId and the vector [AB] projected into this Triangle frame.
     * @param ptA : first input point
     * @param ptB : last input point
     * @param triId : index of the triangle whose edges will be checked
     * @param intersectedEdges : output list of Edge global Ids that are intersected by vector AB (size could be 0, 1 or 2)
     * @param baryCoefs : output list of barycoef corresponding to the relative position of the intersection on the edge (same size and order as @sa intersectedEdges)
    */
    bool computeSegmentTriangleIntersectionInPlane(
        const sofa::type::Vec<3, Real>& ptA,
        const sofa::type::Vec<3, Real>& ptB,
        const TriangleID triId,
        sofa::type::vector<EdgeID>& intersectedEdges,
        sofa::type::vector<Real>& baryCoefs) const;

    /** \brief Computes the intersections of the vector from point a to point b and the triangle indexed by t
    *
    * @param a : first input point
    * @param b : last input point
    * @param ind_t : triangle indice
    * @param indices : indices of edges (belonging to ind_t) crossed by the vecteur AB
    * @param baryCoef : barycoef of intersections points on the edge
    */
    bool computeIntersectionsLineTriangle(bool is_entered,
        const sofa::type::Vec<3, Real>& a,
        const sofa::type::Vec<3, Real>& b,
        const TriangleID ind_t,
        sofa::type::vector<PointID>& indices,
        sofa::type::vector<Real>& vecBaryCoef,
        sofa::type::vector<Real>& vecCoordKmin) const;

    /** \brief Computes the list of points (ind_edge,coord) intersected by the segment from point a to point b and the triangular mesh
     *
     */
    bool computeIntersectedPointsList(const PointID last_point,
            const sofa::type::Vec<3,Real>& a,
            const sofa::type::Vec<3,Real>& b,
            TriangleID& ind_ta, TriangleID& ind_tb,
            sofa::type::vector< TriangleID > &triangles_list,
            sofa::type::vector< EdgeID > &edges_list,
            sofa::type::vector< Real >& coords_list,
            bool& is_on_boundary) const;

    /** \brief Computes the list of objects (points, edges, triangles) intersected by the segment from point a to point b and the triangular mesh.
     *
     * @return List of object intersect (type enum @see geometry::ElementType)
     * @return List of indices of these objects
     * @return List of barycentric coordinate defining the position of the intersection in each object
     * (i.e 0 coord for a point, 1 for and edge and 3 for a triangle).
     */
    bool computeIntersectedObjectsList(const PointID last_point, const Vec3& pointA, const Vec3& pointB,
        TriangleID& ind_triA, TriangleID& ind_triB,
        sofa::type::vector< sofa::geometry::ElementType >& intersected_topoElements,
        sofa::type::vector< ElemID >& intersected_indices,
        sofa::type::vector< Vec3 >& intersected_barycoefs) const;


    /** \brief Get the triangle in a given direction from a point.
     */
    int getTriangleInDirection(PointID p, const sofa::type::Vec<3,Real>& dir) const;


    /** \brief This function will changed vertex index in triangles if normal from one to another triangle are in opposite direction.
      First triangle index is used as ground truth. Use option flipNormals if first triangle direction is wrong.
      */
    void reorderTrianglesOrientationFromNormals();

    /** \brief Process the added point initialization according to the topology and local coordinates.
    */
    void initPointAdded(PointID indice, const core::topology::PointAncestorElem &ancestorElem
        , const type::vector< VecCoord* >& coordVecs, const type::vector< VecDeriv* >& derivVecs) override;

    /// return a pointer to the container of cubature points
    NumericalIntegrationDescriptor<Real,3> &getTriangleNumericalIntegrationDescriptor();


    /** \brief  Moves and fixes the two closest points of two triangles to their median point
    */
    bool Suture2Points(TriangleID ind_ta, TriangleID ind_tb, PointID &ind1, PointID &ind2);

    /** \brief Removes triangles along the list of points (ind_edge,coord) intersected by the vector from point a to point b and the triangular mesh
     */
    void RemoveAlongTrianglesList(const sofa::type::Vec<3, Real>& a,
        const sofa::type::Vec<3, Real>& b,
        const TriangleID ind_ta, const TriangleID ind_tb);

    /** \brief Incises along the list of points (ind_edge,coord) intersected by the sequence of input segments (list of input points) and the triangular mesh
     */
    void InciseAlongLinesList(const sofa::type::vector< sofa::type::Vec<3, Real> >& input_points,
        const sofa::type::vector< TriangleID > &input_triangles);



    /** \brief Split triangles to create edges along a path given as a the list of existing edges and triangles crossed by it.
     * Each end of the path is given either by an existing point or a point inside the first/last triangle. If the first/last triangle is (TriangleID)-1, it means that to path crosses the boundary of the surface.
     * @returns the indice of the end point, or -1 if the incision failed.
     */
    virtual int SplitAlongPath(PointID ind_A, Coord& pointA, PointID ind_B, Coord& pointB,
        sofa::type::vector< sofa::geometry::ElementType >& intersected_topoElements,
        sofa::type::vector< ElemID >& intersected_indices,
        sofa::type::vector< Vec3 >& intersected_barycoefs,
        sofa::type::vector< EdgeID >& new_edges, Real epsilonSnapPath = 0.0, Real epsilonSnapBorder = 0.0);



    /* void SnapAlongPath (sofa::type::vector<TriangleID>& triangles_list, sofa::type::vector<EdgeID>& edges_list,
      sofa::type::vector<Real>& intersected_barycoefs, sofa::type::vector<Real>& points2Snap);*/

    void SnapAlongPath(sofa::type::vector< sofa::geometry::ElementType>& intersected_topoElements,
        sofa::type::vector<ElemID>& intersected_indices, sofa::type::vector< Vec3 >& intersected_barycoefs,
        sofa::type::vector< sofa::type::vector<Real> >& points2Snap,
        Real epsilonSnapPath);

    void SnapBorderPath(PointID pa, Coord& a, PointID pb, Coord& b,
        sofa::type::vector< sofa::geometry::ElementType>& intersected_topoElements,
        sofa::type::vector<ElemID>& intersected_indices,
        sofa::type::vector< Vec3 >& intersected_barycoefs,
        sofa::type::vector< sofa::type::vector<Real> >& points2Snap,
        Real epsilonSnapBorder);



    /** \brief Duplicates the given edges. Only works if at least the first or last point is adjacent to a border.
     * @returns true if the incision succeeded.
     */
    virtual bool InciseAlongEdgeList(const sofa::type::vector<EdgeID>& edges, sofa::type::vector<PointID>& new_points, sofa::type::vector<PointID>& end_points, bool& reachBorder);


    SOFA_ATTRIBUTE_DISABLED("v23.12", "v23.12", "Method writeMSHfile has been disabled. To export the topology as .gmsh file, use the sofa::component::io::mesh::MeshExporter.")
    void writeMSHfile(const char *filename) const {msg_deprecated() << "Method writeMSHfile has been disabled. To export the topology as " << filename << " file, use the sofa::component::io::mesh::MeshExporter."; }

protected:
    Data<bool> showTriangleIndices; ///< Debug : view Triangle indices
    Data<bool> _draw; ///< if true, draw the triangles in the topology
    Data<sofa::type::RGBAColor> _drawColor; ///< RGBA code color used to draw triangles
    Data<bool> _drawNormals; ///< if true, draw the triangles in the topology
    Data <SReal> _drawNormalLength; ///< Fiber length visualisation.
    Data<bool> p_recomputeTrianglesOrientation; ///< if true, will recompute triangles orientation according to normals
    Data<bool> p_flipNormals; ///< if true, will flip normal of the first triangle used to recompute triangle orientation.
    /// include cubature points
    NumericalIntegrationDescriptor<Real,3> triangleNumericalIntegration;

    bool mustComputeBBox() const override;

private:
    TriangleSetTopologyContainer*				m_container;
    TriangleSetTopologyModifier*				m_modifier;
};


/*template< class Real>
  bool is_point_in_triangle(const sofa::type::Vec<3,Real>& p,
  const sofa::type::Vec<3,Real>& a,
  const sofa::type::Vec<3,Real>& b,
  const sofa::type::Vec<3,Real>& c);*/
template<class Real>
bool is_point_in_triangle(const sofa::type::Vec<3,Real>& p,
        const sofa::type::Vec<3,Real>& a, const sofa::type::Vec<3,Real>& b, const sofa::type::Vec<3,Real>& c);


template< class Real>
bool is_point_in_halfplane(const sofa::type::Vec<3,Real>& p,
        unsigned int e0, unsigned int e1,
        const sofa::type::Vec<3,Real>& a,
        const sofa::type::Vec<3,Real>& b,
        const sofa::type::Vec<3,Real>& c,
        unsigned int ind_p0, unsigned int ind_p1, unsigned int ind_p2);

void SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API snapping_test_triangle(SReal epsilon,
        SReal alpha0, SReal alpha1, SReal alpha2,
        bool& is_snap_0, bool& is_snap_1, bool& is_snap_2);

void SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API snapping_test_edge(SReal epsilon,
        SReal alpha0, SReal alpha1,
        bool& is_snap_0, bool& is_snap_1);

template< class Real>
inline Real areaProduct(const sofa::type::Vec<3,Real>& a, const sofa::type::Vec<3,Real>& b);

template< class Real>
inline Real areaProduct(const type::Vec<2,Real>& a, const type::Vec<2,Real>& b );

template< class Real>
inline Real areaProduct(const type::Vec<1,Real>& , const type::Vec<1,Real>&  );

#if !defined(SOFA_COMPONENT_TOPOLOGY_TRIANGLESETGEOMETRYALGORITHMS_CPP)
extern template class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API TriangleSetGeometryAlgorithms<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API TriangleSetGeometryAlgorithms<defaulttype::Vec2Types>;
#endif

} //namespace sofa::component::topology::container::dynamic
