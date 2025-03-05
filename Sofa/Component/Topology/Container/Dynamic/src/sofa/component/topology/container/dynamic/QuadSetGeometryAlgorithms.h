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

namespace sofa::component::topology::container::dynamic
{
/**
* A class that provides geometry information on an QuadSet.
*/
template < class DataTypes >
class QuadSetGeometryAlgorithms : public EdgeSetGeometryAlgorithms<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(QuadSetGeometryAlgorithms,DataTypes),SOFA_TEMPLATE(EdgeSetGeometryAlgorithms,DataTypes));

    typedef sofa::core::topology::BaseMeshTopology::PointID PointID;
    typedef sofa::core::topology::BaseMeshTopology::EdgeID EdgeID;
    typedef sofa::core::topology::BaseMeshTopology::Edge Edge;
    typedef sofa::core::topology::BaseMeshTopology::SeqEdges SeqEdges;
    typedef sofa::core::topology::BaseMeshTopology::EdgesAroundVertex EdgesAroundVertex;

    typedef sofa::core::topology::BaseMeshTopology::QuadID QuadID;
    typedef sofa::core::topology::BaseMeshTopology::Quad Quad;
    typedef sofa::core::topology::BaseMeshTopology::SeqQuads SeqQuads;
    typedef sofa::core::topology::BaseMeshTopology::QuadsAroundVertex QuadsAroundVertex;
    typedef sofa::core::topology::BaseMeshTopology::QuadsAroundEdge QuadsAroundEdge;
    typedef sofa::core::topology::BaseMeshTopology::EdgesInQuad EdgesInQuad;


    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
protected:
    QuadSetGeometryAlgorithms()
        : EdgeSetGeometryAlgorithms<DataTypes>()
        , showQuadIndices(core::objectmodel::Base::initData(&showQuadIndices, (bool) false, "showQuadIndices", "Debug : view Quad indices"))
        , _drawQuads(core::objectmodel::Base::initData(&_drawQuads, false, "drawQuads","if true, draw the quads in the topology"))
        , _drawColor(initData(&_drawColor, sofa::type::RGBAColor(0.0f,0.4f,0.4f,1.0f), "drawColorQuads", "RGB code color used to draw quads."))
    { }

    virtual ~QuadSetGeometryAlgorithms() {}
public:
    void computeQuadAABB(const QuadID i, Coord& minCoord, Coord& maxCoord) const;

    Coord computeQuadCenter(const QuadID i) const;

    void getQuadVertexCoordinates(const QuadID i, Coord[4]) const;

    void getRestQuadVertexCoordinates(const QuadID i, Coord[4]) const;

    /** \brief Computes the area of quad no i and returns it
    *
    */
    Real computeQuadArea(const QuadID i) const;

    /** \brief Computes the quad area of all quads are store in the array interface
    *
    */
    void computeQuadArea( BasicArrayInterface<Real> &ai) const;

    /** \brief Computes the initial area  of quad no i and returns it
    *
    */
    Real computeRestQuadArea(const QuadID i) const;

    /** \brief Computes the normal vector of a quad indexed by ind_q (not normed)
    *
    */
    type::Vec<3,double> computeQuadNormal(const QuadID ind_q) const;

    /** \brief Tests if a quad indexed by ind_q (and incident to the vertex indexed by ind_p)
    * is included or not in the plane defined by (ind_p, plane_vect)
    *
    */
    bool isQuadInPlane(const QuadID ind_q, const PointID ind_p,
            const type::Vec<3,Real>& plane_vect) const;

    bool isPointInQuad(const QuadID ind_q, const sofa::type::Vec<3,Real>& p) const;

    void draw(const core::visual::VisualParams* vparams) override;

    SOFA_ATTRIBUTE_DISABLED("v23.12", "v23.12", "Method writeMSHfile has been disabled. To export the topology as .gmsh file, use the sofa::component::io::mesh::MeshExporter.")
    void writeMSHfile(const char *filename) const {msg_deprecated() << "Method writeMSHfile has been disabled. To export the topology as " << filename << " file, use the sofa::component::io::mesh::MeshExporter."; }

protected:
    Data<bool> showQuadIndices; ///< Debug : view Quad indices
    Data<bool> _drawQuads; ///< if true, draw the quads in the topology
    Data<sofa::type::RGBAColor> _drawColor; ///< RGB code color used to draw quads.

    bool mustComputeBBox() const override;
};

template<class Coord>
bool is_point_in_quad(const Coord& p,
        const Coord& a, const Coord& b,
        const Coord& c, const Coord& d);

void snapping_test_quad(double epsilon, double alpha0, double alpha1, double alpha2, double alpha3,
        bool& is_snap_0, bool& is_snap_1, bool& is_snap_2, bool& is_snap_3);

template< class Real>
inline Real areaProduct(const type::Vec<3,Real>& a, const type::Vec<3,Real>& b);

template< class Real>
inline Real areaProduct(const type::Vec<2,Real>& a, const type::Vec<2,Real>& b );

template< class Real>
inline Real areaProduct(const type::Vec<1,Real>& , const type::Vec<1,Real>&  );

#if !defined(SOFA_COMPONENT_TOPOLOGY_QUADSETGEOMETRYALGORITHMS_CPP)
extern template class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API QuadSetGeometryAlgorithms<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API QuadSetGeometryAlgorithms<defaulttype::Vec2Types>;

#endif

} //namespace sofa::component::topology::container::dynamic
