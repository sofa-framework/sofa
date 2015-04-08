/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETGEOMETRYALGORITHMS_H
#define SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETGEOMETRYALGORITHMS_H

#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.h>
#include <SofaBaseTopology/NumericalIntegrationDescriptor.h>

namespace sofa
{

namespace component
{

namespace topology
{
using core::topology::BaseMeshTopology;
typedef BaseMeshTopology::TetraID TetraID;
typedef BaseMeshTopology::Tetra Tetra;
typedef BaseMeshTopology::SeqTetrahedra SeqTetrahedra;
typedef BaseMeshTopology::SeqEdges SeqEdges;
typedef BaseMeshTopology::TetrahedraAroundVertex TetrahedraAroundVertex;
typedef BaseMeshTopology::TetrahedraAroundEdge TetrahedraAroundEdge;
typedef BaseMeshTopology::TetrahedraAroundTriangle TetrahedraAroundTriangle;
typedef BaseMeshTopology::EdgesInTetrahedron EdgesInTetrahedron;
typedef BaseMeshTopology::TrianglesInTetrahedron TrianglesInTetrahedron;

typedef Tetra Tetrahedron;
typedef EdgesInTetrahedron EdgesInTetrahedron;
typedef TrianglesInTetrahedron TrianglesInTetrahedron;

/**
* A class that provides geometry information on an TetrahedronSet.
*/
template < class DataTypes >
class TetrahedronSetGeometryAlgorithms : public TriangleSetGeometryAlgorithms<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TetrahedronSetGeometryAlgorithms,DataTypes),SOFA_TEMPLATE(TriangleSetGeometryAlgorithms,DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
protected:
	bool initializedCubatureTables;
	void defineTetrahedronCubaturePoints();

    TetrahedronSetGeometryAlgorithms()
        : TriangleSetGeometryAlgorithms<DataTypes>()
		,initializedCubatureTables(false)
        ,showTetrahedraIndices (core::objectmodel::Base::initData(&showTetrahedraIndices, (bool) false, "showTetrahedraIndices", "Debug : view Tetrahedrons indices"))
        , _draw(core::objectmodel::Base::initData(&_draw, false, "drawTetrahedra","if true, draw the tetrahedra in the topology"))
        , _drawColor(initData(&_drawColor, sofa::defaulttype::Vec3f(1.0f,1.0f,0.0f), "drawColorTetrahedra", "RGB code color used to draw tetrahedra."))
    {
        core::objectmodel::Base::addAlias(&showTetrahedraIndices, "showTetrasIndices");
    }

    virtual ~TetrahedronSetGeometryAlgorithms() {}
public:
    virtual void draw(const core::visual::VisualParams* vparams);

    void computeTetrahedronAABB(const TetraID i, Coord& minCoord, Coord& maxCoord) const;

    Coord computeTetrahedronCenter(const TetraID i) const;

    Coord computeTetrahedronCircumcenter(const TetraID i) const;

    bool isPointInTetrahedron(const TetraID i, const sofa::defaulttype::Vec<3,Real>& p) const;

    /// return (if the point is in the tetrahedron) the barycentric coordinates of the point in the tetrahedron
    bool isPointInTetrahedron(const TetraID ind_t, const sofa::defaulttype::Vec<3,Real>& pTest, sofa::defaulttype::Vec<4,Real>& barycentricCoordinates) const;

    void getTetrahedronVertexCoordinates(const TetraID i, Coord[4]) const;

    void getRestTetrahedronVertexCoordinates(const TetraID i, Coord[4]) const;

    /// computes the volume of tetrahedron no i and returns it
    Real computeTetrahedronVolume(const TetraID i) const;

    /// computes the tetrahedron volume of all tetrahedra are store in the array interface
    void computeTetrahedronVolume( BasicArrayInterface<Real> &ai) const;

    /// computes the tetrahedron volume  of tetrahedron no i and returns it
    Real computeRestTetrahedronVolume(const TetraID i) const;
    Real computeRestTetrahedronVolume(const Tetrahedron t) const;

    /// finds the indices of all tetrahedra in the ball of center ind_ta and of radius dist(ind_ta, ind_tb)
    void getTetraInBall(const TetraID ind_ta, const TetraID ind_tb,
            sofa::helper::vector<unsigned int> &indices) const;

    /// finds the indices of all tetrahedra in the ball of center ind_ta and of radius dist(ind_ta, ind_tb)
    void getTetraInBall(const TetraID ind_ta, Real r,
            sofa::helper::vector<unsigned int> &indices) const;
    void getTetraInBall(const Coord& c, Real r,
            sofa::helper::vector<unsigned int> &indices) const;
    /** \brief Write the current mesh into a msh file
    */
    void writeMSHfile(const char *filename) const;

    /// finds the intersection point with plane which is defined by c and normal
    void getIntersectionPointWithPlane(const TetraID ind_ta, sofa::defaulttype::Vec<3,Real>& c, sofa::defaulttype::Vec<3,Real>& normal, sofa::helper::vector< sofa::defaulttype::Vec<3,Real> >& intersectedPoint, SeqEdges& intersectedEdge);

    /// finds the intersection point between edge and plane
    bool computeIntersectionEdgeWithPlane(sofa::defaulttype::Vec<3,Real>& p1,
                                          sofa::defaulttype::Vec<3,Real>& p2,
                                          sofa::defaulttype::Vec<3,Real>& c,
                                          sofa::defaulttype::Vec<3,Real>& normal,
                                          sofa::defaulttype::Vec<3,Real>& intersection);

    bool checkNodeSequence(Tetra& tetra);

	/// return a pointer to the container of cubature points
	NumericalIntegrationDescriptor<Real,4> &getTetrahedronNumericalIntegrationDescriptor();

protected:
    Data<bool> showTetrahedraIndices;
    Data<bool> _draw;
    Data<sofa::defaulttype::Vec3f> _drawColor;
	/// include cubature points
	NumericalIntegrationDescriptor<Real,4> tetrahedronNumericalIntegration;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETGEOMETRYALGORITHMS_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BASE_TOPOLOGY_API TetrahedronSetGeometryAlgorithms<defaulttype::Vec3dTypes>;
extern template class SOFA_BASE_TOPOLOGY_API TetrahedronSetGeometryAlgorithms<defaulttype::Vec2dTypes>;
extern template class SOFA_BASE_TOPOLOGY_API TetrahedronSetGeometryAlgorithms<defaulttype::Vec1dTypes>;
//extern template class SOFA_BASE_TOPOLOGY_API TetrahedronSetGeometryAlgorithms<defaulttype::Rigid3dTypes>;
//extern template class SOFA_BASE_TOPOLOGY_API TetrahedronSetGeometryAlgorithms<defaulttype::Rigid2dTypes>;
#endif

#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_TOPOLOGY_API TetrahedronSetGeometryAlgorithms<defaulttype::Vec3fTypes>;
extern template class SOFA_BASE_TOPOLOGY_API TetrahedronSetGeometryAlgorithms<defaulttype::Vec2fTypes>;
extern template class SOFA_BASE_TOPOLOGY_API TetrahedronSetGeometryAlgorithms<defaulttype::Vec1fTypes>;
//extern template class SOFA_BASE_TOPOLOGY_API TetrahedronSetGeometryAlgorithms<defaulttype::Rigid3fTypes>;
//extern template class SOFA_BASE_TOPOLOGY_API TetrahedronSetGeometryAlgorithms<defaulttype::Rigid2fTypes>;
#endif
#endif

} // namespace topology

} // namespace component

} // namespace sofa

#endif
