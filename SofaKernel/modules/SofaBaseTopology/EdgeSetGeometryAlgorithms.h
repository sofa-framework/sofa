/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_EDGESETGEOMETRYALGORITHMS_H
#define SOFA_COMPONENT_TOPOLOGY_EDGESETGEOMETRYALGORITHMS_H
#include "config.h"

#include <SofaBaseTopology/PointSetGeometryAlgorithms.h>
#include <SofaBaseTopology/NumericalIntegrationDescriptor.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/helper/types/RGBAColor.h>
namespace sofa
{

namespace component
{

namespace topology
{
/** \brief A class used as an interface with an array : Useful to compute geometric information on each edge in an efficient way
 *
 */
template < class T>
class BasicArrayInterface
{
public:
    // Access to i-th element.
    virtual T & operator[](int i)=0;
    virtual ~BasicArrayInterface() {}

};


/**
 * A class that provides geometry information on an EdgeSet.
 */
template < class DataTypes >
class EdgeSetGeometryAlgorithms : public PointSetGeometryAlgorithms<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(EdgeSetGeometryAlgorithms,DataTypes),SOFA_TEMPLATE(PointSetGeometryAlgorithms,DataTypes));
    typedef sofa::core::topology::BaseMeshTopology::EdgeID EdgeID;
    typedef sofa::core::topology::BaseMeshTopology::Edge Edge;
    typedef sofa::core::topology::BaseMeshTopology::SeqEdges SeqEdges;
    typedef sofa::core::topology::BaseMeshTopology::EdgesAroundVertex EdgesAroundVertex;

    typedef sofa::helper::types::RGBAColor RGBAColor ;
    typedef sofa::defaulttype::Vec3d Vec3d;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::CPos CPos;
    enum { NC = CPos::static_size };

protected:
    bool initializedEdgeCubatureTables;
    EdgeSetGeometryAlgorithms()
        : PointSetGeometryAlgorithms<DataTypes>()
        ,initializedEdgeCubatureTables(false)
        , showEdgeIndices(core::objectmodel::Base::initData(&showEdgeIndices, (bool) false, "showEdgeIndices", "Debug : view Edge indices."))
        , _draw(core::objectmodel::Base::initData(&_draw, false, "drawEdges","if true, draw the edges in the topology."))
        , _drawColor(initData(&_drawColor, RGBAColor(0.4f,1.0f,0.3f, 1.0f), "drawColorEdges", "RGB code color used to draw edges."))
    {
    }
    virtual ~EdgeSetGeometryAlgorithms() {}

    void defineEdgeCubaturePoints();
public:
    //virtual void reinit();

    virtual void draw(const core::visual::VisualParams* vparams);

    /// computes the length of edge no i and returns it
    Real computeEdgeLength(const EdgeID i) const;

    /// computes the edge length of all edges and stores it in the array interface
    void computeEdgeLength( BasicArrayInterface<Real> &ai) const;

    /// computes the initial length of edge no i and returns it
    Real computeRestEdgeLength(const EdgeID i) const;

    /// computes the initial square length of edge no i and returns it
    Real computeRestSquareEdgeLength(const EdgeID i) const;

    void computeEdgeAABB(const EdgeID i, CPos& minCoord, CPos& maxCoord) const;

    Coord computeEdgeCenter(const EdgeID i) const;

    Coord computeEdgeDirection(const EdgeID i) const;
    Coord computeRestEdgeDirection(const EdgeID i) const;

    void getEdgeVertexCoordinates(const EdgeID i, Coord[2]) const;

    void getRestEdgeVertexCoordinates(const EdgeID i, Coord[2]) const;

    // test if a point is on the triangle indexed by ind_e
    bool isPointOnEdge(const sofa::defaulttype::Vec<3,double> &pt, const unsigned int ind_e) const;

    // compute barycentric coefficients
    sofa::helper::vector< double > compute2PointsBarycoefs(const sofa::defaulttype::Vec<3,double> &p, unsigned int ind_p1, unsigned int ind_p2) const;
    sofa::helper::vector< double > computeRest2PointsBarycoefs(const sofa::defaulttype::Vec<3,double> &p, unsigned int ind_p1, unsigned int ind_p2) const;

    /** \brief Compute the projection coordinate of a point C on the edge i. Using compute2EdgesIntersection().
    * @param i edgeID on which point is projected.
    * @param coord_x coordinate of point to project
    * @param intersected bool default value true, changed as false if no intersection is done.
    * @return barycentric coefficient of the projection in edgeID i.
    */
    sofa::helper::vector< double > computePointProjectionOnEdge (const EdgeID i, sofa::defaulttype::Vec<3,double> coord_x, bool& intersected);

    /** \brief Compute the intersection coordinate of the 2 input straight lines. Lines vector director are computed using coord given in input.
    * @param edge1 tab Coord[2] from the 2 vertices composing first edge
    * @param edge2 same for second edge
    * @param intersected bool default value true, changed as false if no intersection is done.
    * @return Coord of intersection point, 0 if no intersection.
    */
    Coord compute2EdgesIntersection (const Coord edge1[2], const Coord edge2[2], bool& intersected);

    bool computeEdgePlaneIntersection (EdgeID edgeID, sofa::defaulttype::Vec<3,Real> pointOnPlane, sofa::defaulttype::Vec<3,Real> normalOfPlane, sofa::defaulttype::Vec<3,Real>& intersection);
    bool computeRestEdgePlaneIntersection (EdgeID edgeID, sofa::defaulttype::Vec<3,Real> pointOnPlane, sofa::defaulttype::Vec<3,Real> normalOfPlane, sofa::defaulttype::Vec<3,Real>& intersection);

    void writeMSHfile(const char *filename) const;

    /** Computes weights allowing to compute the deformation gradient (deformed basis)  at each vertex during the simulation, for a volumetric object.
      For each vertex, computes the weights associated with each edge around the vertex, so that the weighted sum of the edges corresponds to the identity.
      The current configuration is taken as reference.
      During the simulation, the weights and edge indices can be used to compute rotated and deformed bases for each vertex

      The output vectors contain the concatenation of the values for each vertex.
      The weights are computed using a pseudo-inverse of the edge matrix: w_i = Vt_i.(V.Vt)^{-1}.Id3
      \param numEdges number of edges attached to a vertex
      \param edges attached to the vertices
      \param weights associated with the edges. Each Vec3d represents the contribution of the associated edge to x,y and z of the deformed basis.
      */
   void computeLocalFrameEdgeWeights( helper::vector<unsigned>& numEdges, helper::vector<Edge>& edges, helper::vector<Vec3d>& weights ) const;

    /** \brief Process the added point initialization according to the topology and local coordinates.
    */
    virtual void initPointAdded(unsigned int indice, const core::topology::PointAncestorElem &ancestorElem
        , const helper::vector< VecCoord* >& coordVecs, const helper::vector< VecDeriv* >& derivVecs);

    /** return a pointer to the container of cubature points */
    NumericalIntegrationDescriptor<Real,1> &getEdgeNumericalIntegrationDescriptor();

protected:
    Data<bool> showEdgeIndices;
    Data<bool>  _draw;
    Data<RGBAColor> _drawColor;
    /// include cubature points
    NumericalIntegrationDescriptor<Real,1> edgeNumericalIntegration;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_TOPOLOGY_EDGESETGEOMETRYALGORITHMS_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BASE_TOPOLOGY_API EdgeSetGeometryAlgorithms<defaulttype::Vec3dTypes>;
extern template class SOFA_BASE_TOPOLOGY_API EdgeSetGeometryAlgorithms<defaulttype::Vec2dTypes>;
extern template class SOFA_BASE_TOPOLOGY_API EdgeSetGeometryAlgorithms<defaulttype::Vec1dTypes>;
extern template class SOFA_BASE_TOPOLOGY_API EdgeSetGeometryAlgorithms<defaulttype::Rigid3dTypes>;
extern template class SOFA_BASE_TOPOLOGY_API EdgeSetGeometryAlgorithms<defaulttype::Rigid2dTypes>;
#endif

#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_TOPOLOGY_API EdgeSetGeometryAlgorithms<defaulttype::Vec3fTypes>;
extern template class SOFA_BASE_TOPOLOGY_API EdgeSetGeometryAlgorithms<defaulttype::Vec2fTypes>;
extern template class SOFA_BASE_TOPOLOGY_API EdgeSetGeometryAlgorithms<defaulttype::Vec1fTypes>;
extern template class SOFA_BASE_TOPOLOGY_API EdgeSetGeometryAlgorithms<defaulttype::Rigid3fTypes>;
extern template class SOFA_BASE_TOPOLOGY_API EdgeSetGeometryAlgorithms<defaulttype::Rigid2fTypes>;
#endif
#endif

} // namespace topology

} // namespace component

} // namespace sofa

#endif
