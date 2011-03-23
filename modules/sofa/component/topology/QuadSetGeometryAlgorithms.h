/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_QUADSETGEOMETRYALGORITHMS_H
#define SOFA_COMPONENT_TOPOLOGY_QUADSETGEOMETRYALGORITHMS_H

#include <sofa/component/topology/EdgeSetGeometryAlgorithms.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{
namespace component
{
namespace topology
{
using core::topology::BaseMeshTopology;
typedef BaseMeshTopology::QuadID QuadID;
typedef BaseMeshTopology::Quad Quad;
typedef BaseMeshTopology::SeqQuads SeqQuads;
typedef BaseMeshTopology::QuadsAroundVertex QuadsAroundVertex;
typedef BaseMeshTopology::QuadsAroundEdge QuadsAroundEdge;
typedef BaseMeshTopology::EdgesInQuad EdgesInQuad;

/**
* A class that provides geometry information on an QuadSet.
*/
template < class DataTypes >
class QuadSetGeometryAlgorithms : public EdgeSetGeometryAlgorithms<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(QuadSetGeometryAlgorithms,DataTypes),SOFA_TEMPLATE(EdgeSetGeometryAlgorithms,DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;

    QuadSetGeometryAlgorithms()
        : EdgeSetGeometryAlgorithms<DataTypes>()
        , showQuadIndices(core::objectmodel::Base::initData(&showQuadIndices, (bool) false, "showQuadIndices", "Debug : view Quad indices"))
        , _draw(core::objectmodel::Base::initData(&_draw, false, "drawQuads","if true, draw the quads in the topology"))
        , _drawColor(initData(&_drawColor, sofa::defaulttype::Vector3(0.0,0.4,0.4), "drawColorQuads", "RGB code color used to draw quads."))
    { }

    virtual ~QuadSetGeometryAlgorithms() {}

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
    defaulttype::Vec<3,double> computeQuadNormal(const QuadID ind_q) const;

    /** \brief Tests if a quad indexed by ind_q (and incident to the vertex indexed by ind_p)
    * is included or not in the plane defined by (ind_p, plane_vect)
    *
    */
    bool isQuadInPlane(const QuadID ind_q, const unsigned int ind_p,
            const defaulttype::Vec<3,Real>& plane_vect) const;

    bool isPointInQuad(const QuadID ind_q, const sofa::defaulttype::Vec<3,Real>& p) const;

    /** \brief Write the current mesh into a msh file
    */
    void writeMSHfile(const char *filename) const;

    virtual void draw();

protected:
    Data<bool> showQuadIndices;
    Data<bool> _draw;
    Data<sofa::defaulttype::Vector3> _drawColor;

};

template<class Coord>
bool is_point_in_quad(const Coord& p,
        const Coord& a, const Coord& b,
        const Coord& c, const Coord& d);

void snapping_test_quad(double epsilon, double alpha0, double alpha1, double alpha2, double alpha3,
        bool& is_snap_0, bool& is_snap_1, bool& is_snap_2, bool& is_snap_3);

template< class Real>
inline Real areaProduct(const defaulttype::Vec<3,Real>& a, const defaulttype::Vec<3,Real>& b);

template< class Real>
inline Real areaProduct(const defaulttype::Vec<2,Real>& a, const defaulttype::Vec<2,Real>& b );

template< class Real>
inline Real areaProduct(const defaulttype::Vec<1,Real>& , const defaulttype::Vec<1,Real>&  );

#if defined(WIN32) && !defined(SOFA_COMPONENT_TOPOLOGY_QUADSETGEOMETRYALGORITHMS_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_TOPOLOGY_API QuadSetGeometryAlgorithms<defaulttype::Vec3dTypes>;
extern template class SOFA_COMPONENT_TOPOLOGY_API QuadSetGeometryAlgorithms<defaulttype::Vec2dTypes>;
extern template class SOFA_COMPONENT_TOPOLOGY_API QuadSetGeometryAlgorithms<defaulttype::Vec1dTypes>;
//extern template class SOFA_COMPONENT_TOPOLOGY_API QuadSetGeometryAlgorithms<defaulttype::Rigid3dTypes>;
//extern template class SOFA_COMPONENT_TOPOLOGY_API QuadSetGeometryAlgorithms<defaulttype::Rigid2dTypes>;
#endif

#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_TOPOLOGY_API QuadSetGeometryAlgorithms<defaulttype::Vec3fTypes>;
extern template class SOFA_COMPONENT_TOPOLOGY_API QuadSetGeometryAlgorithms<defaulttype::Vec2fTypes>;
extern template class SOFA_COMPONENT_TOPOLOGY_API QuadSetGeometryAlgorithms<defaulttype::Vec1fTypes>;
//extern template class SOFA_COMPONENT_TOPOLOGY_API QuadSetGeometryAlgorithms<defaulttype::Rigid3fTypes>;
//extern template class SOFA_COMPONENT_TOPOLOGY_API QuadSetGeometryAlgorithms<defaulttype::Rigid2fTypes>;
#endif
#endif

} // namespace topology

} // namespace component

} // namespace sofa

#endif
