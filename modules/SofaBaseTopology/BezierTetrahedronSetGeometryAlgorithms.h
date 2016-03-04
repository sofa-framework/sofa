/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_BEZIERTETRAHEDRONSETGEOMETRYALGORITHMS_H
#define SOFA_COMPONENT_TOPOLOGY_BEZIERTETRAHEDRONSETGEOMETRYALGORITHMS_H
#include "config.h"

#include <SofaBaseTopology/TetrahedronSetGeometryAlgorithms.h>
#include <SofaBaseTopology/BezierTetrahedronSetTopologyContainer.h>


namespace sofa
{

namespace component
{

namespace topology
{

/**
* A class that provides geometry information on an TetrahedronSet.
*/
template < class DataTypes >
class BezierTetrahedronSetGeometryAlgorithms : public TetrahedronSetGeometryAlgorithms<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(BezierTetrahedronSetGeometryAlgorithms,DataTypes),SOFA_TEMPLATE(TetrahedronSetGeometryAlgorithms,DataTypes));
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef sofa::defaulttype::Vec<4,Real> Vec4;
    typedef sofa::defaulttype::Mat<4,4,Real> Mat44;

    typedef core::topology::BaseMeshTopology::PointID PointID;
    typedef core::topology::BaseMeshTopology::Edge Edge;
    typedef core::topology::BaseMeshTopology::TetraID TetraID;
    typedef core::topology::BaseMeshTopology::Tetra Tetra;
    typedef core::topology::BaseMeshTopology::SeqTetrahedra SeqTetrahedra;
    typedef core::topology::BaseMeshTopology::SeqEdges SeqEdges;
    typedef core::topology::BaseMeshTopology::TetrahedraAroundVertex TetrahedraAroundVertex;
    typedef core::topology::BaseMeshTopology::TetrahedraAroundEdge TetrahedraAroundEdge;
    typedef core::topology::BaseMeshTopology::TetrahedraAroundTriangle TetrahedraAroundTriangle;
    typedef core::topology::BaseMeshTopology::EdgesInTetrahedron EdgesInTetrahedron;
    typedef core::topology::BaseMeshTopology::TrianglesInTetrahedron TrianglesInTetrahedron;
	typedef BezierTetrahedronSetTopologyContainer::VecPointID VecPointID;

	typedef Tetra Tetrahedron;
protected:
   
	/// container	
	BezierTetrahedronSetTopologyContainer *container; 
	/// degree of the polynomial
	BezierDegreeType degree; 
	// array of Tetrahedral Bezier indices
	sofa::helper::vector<TetrahedronBezierIndex> tbiArray;
	// array of Bernstein coefficient following the same order as tbiArray
	sofa::helper::vector<Real> bernsteinCoefficientArray;
	// map used to store the Bernstein coefficient given a Tetrahedron Bezier Index
	std::map<TetrahedronBezierIndex,Real> bernsteinCoeffMap;
	/// the list of edges of the Bezier Tetrahedron used in the draw function
    sofa::helper::set<std::pair<Edge,size_t> > bezierTetrahedronEdgeSet;



	/// constructor 
	BezierTetrahedronSetGeometryAlgorithms();
    virtual ~BezierTetrahedronSetGeometryAlgorithms() {}
public:
	virtual void init();
	virtual void reinit();
    virtual void draw(const core::visual::VisualParams* vparams);
	/// returns a pointer to the BezierTetrahedronSetTopologyContainer object
	BezierTetrahedronSetTopologyContainer *getTopologyContainer() const {
		return container;
	}
	/// computes the nodal value given the tetrahedron index, the barycentric coordinates and the vector of nodal values
	Coord computeNodalValue(const size_t tetrahedronIndex,const Vec4 barycentricCoordinate,const VecCoord& p); 
	/// computes the nodal value assuming that the position is the regular position in the mechanical state object
	Coord computeNodalValue(const size_t tetrahedronIndex,const Vec4 barycentricCoordinate); 
	/// computes the shape function 
	Real computeBernsteinPolynomial(const TetrahedronBezierIndex tbi, const Vec4 barycentricCoordinate);
	/// computes the shape function gradient
    Vec4 computeBernsteinPolynomialGradient(const TetrahedronBezierIndex tbi, const Vec4 barycentricCoordinate);
    /// computes the shape function hessian
    Mat44 computeBernsteinPolynomialHessian(const TetrahedronBezierIndex tbi, const Vec4 barycentricCoordinate);
	/// computes Jacobian i.e. determinant of dpos/dmu
	Real computeJacobian(const size_t tetrahedronIndex, const Vec4 barycentricCoordinate,const VecCoord& p);
	/// computes Jacobian
	Real computeJacobian(const size_t tetrahedronIndex, const Vec4 barycentricCoordinate);
	/// compute the 4 De Casteljeau  of degree d-1
	void computeDeCasteljeauPoints(const size_t tetrahedronIndex, const Vec4 barycentricCoordinate, const VecCoord& p,Coord point[4]);
	/// test if the Bezier tetrahedron is a simple affine tesselation of a regular tetrahedron
	bool isBezierTetrahedronAffine(const size_t tetrahedronIndex,const VecCoord& p, const Real tolerance=(Real)1e-5) const; 
protected:
    Data<bool> drawControlPointsEdges;
	Data<bool> drawSmoothEdges;
	Data<bool> drawControlPoints;
	Data<Real> d_referenceRadius; // radius to draw control points
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_TOPOLOGY_BEZIERTETRAHEDRONSETGEOMETRYALGORITHMS_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BASE_TOPOLOGY_API BezierTetrahedronSetGeometryAlgorithms<defaulttype::Vec3dTypes>;
extern template class SOFA_BASE_TOPOLOGY_API BezierTetrahedronSetGeometryAlgorithms<defaulttype::Vec2dTypes>;
extern template class SOFA_BASE_TOPOLOGY_API BezierTetrahedronSetGeometryAlgorithms<defaulttype::Vec1dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_TOPOLOGY_API BezierTetrahedronSetGeometryAlgorithms<defaulttype::Vec3fTypes>;
extern template class SOFA_BASE_TOPOLOGY_API BezierTetrahedronSetGeometryAlgorithms<defaulttype::Vec2fTypes>;
extern template class SOFA_BASE_TOPOLOGY_API BezierTetrahedronSetGeometryAlgorithms<defaulttype::Vec1fTypes>;
#endif
#endif

} // namespace topology

} // namespace component

} // namespace sofa

#endif
