/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_BEZIERTRIANGLESETGEOMETRYALGORITHMS_H
#define SOFA_COMPONENT_TOPOLOGY_BEZIERTRIANGLESETGEOMETRYALGORITHMS_H

#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.h>
#include <SofaGeneralTopology/BezierTriangleSetTopologyContainer.h>


namespace sofa
{

namespace component
{

namespace topology
{


/**
* A class that provides geometry information on an Bezier TriangleSet of any degree.
*/
template < class DataTypes >
class BezierTriangleSetGeometryAlgorithms : public TriangleSetGeometryAlgorithms<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(BezierTriangleSetGeometryAlgorithms,DataTypes),SOFA_TEMPLATE(TriangleSetGeometryAlgorithms,DataTypes));


    typedef core::topology::BaseMeshTopology::PointID PointID;
    typedef core::topology::BaseMeshTopology::Edge Edge;
    typedef core::topology::BaseMeshTopology::TriangleID TriangleID;
    typedef core::topology::BaseMeshTopology::Triangle Triangle;
    typedef core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
    typedef core::topology::BaseMeshTopology::TrianglesAroundVertex TrianglesAroundVertex;
    typedef core::topology::BaseMeshTopology::TrianglesAroundEdge TrianglesAroundEdge;
    typedef core::topology::BaseMeshTopology::EdgesInTriangle EdgesInTriangle;
	typedef BezierTriangleSetTopologyContainer::VecPointID VecPointID;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
	typedef typename DataTypes::Deriv Deriv;
    typedef sofa::defaulttype::Vec<3,Real> Vec3;
    typedef sofa::defaulttype::Mat<3,3,Real> Mat33;

    Data<bool> drawControlPointsEdges;
    Data<bool> drawSmoothEdges;
	Data<bool> drawControlPoints;
protected:
   
	/// container	
	BezierTriangleSetTopologyContainer *container; 
	/// degree of the polynomial
	BezierDegreeType degree; 
	// array of Triangle Bezier indices
	sofa::helper::vector<TriangleBezierIndex> tbiArray;
	// array of Bernstein coefficient following the same order as tbiArray
	sofa::helper::vector<Real> bernsteinCoefficientArray;
	// map used to store the Bernstein coefficient given a Triangle Bezier Index
	std::map<TriangleBezierIndex,Real> bernsteinCoeffMap;
	/// the list of edges of the Bezier Triangle used in the draw function
    std::set<std::pair<Edge,bool> > bezierTriangleEdgeSet;


	/// constructor 
	BezierTriangleSetGeometryAlgorithms();
    virtual ~BezierTriangleSetGeometryAlgorithms() {}
public:
	virtual void init();
	virtual void reinit();
    virtual void draw(const core::visual::VisualParams* vparams);
	/// returns a pointer to the BezierTriangleSetTopologyContainer object
	BezierTriangleSetTopologyContainer *getTopologyContainer() const {
		return container;
	}
	/// computes the nodal value given the Triangle index, the barycentric coordinates and the vector of nodal values
	Coord computeNodalValue(const size_t triangleIndex,const Vec3 barycentricCoordinate,const VecCoord& p); 
	/// computes the nodal value assuming that the position is the regular position in the mechanical state object
	Coord computeNodalValue(const size_t triangleIndex,const Vec3 barycentricCoordinate); 
	/// computes the shape function 
	Real computeBernsteinPolynomial(const TriangleBezierIndex tbi, const Vec3 barycentricCoordinate);
	/// computes the shape function gradient
    Vec3 computeBernsteinPolynomialGradient(const TriangleBezierIndex tbi, const Vec3 barycentricCoordinate);
    /// computes the shape function hessian
    Mat33 computeBernsteinPolynomialHessian(const TriangleBezierIndex tbi, const Vec3 barycentricCoordinate);
	/// computes Jacobian i.e. cross product  of dpos/du and dpos/dv
	Deriv computeJacobian(const size_t triangleIndex, const Vec3 barycentricCoordinate,const VecCoord& p);
	/// computes Jacobian
	Deriv computeJacobian(const size_t triangleIndex, const Vec3 barycentricCoordinate);
	/// compute the 4 De Casteljeau  of degree d-1
	void computeDeCasteljeauPoints(const size_t triangleIndex, const Vec3 barycentricCoordinate, const VecCoord& p,Coord point[3]);
	/// test if the Bezier Triangle is a simple affine tesselation of a regular Triangle
	bool isBezierTriangleAffine(const size_t triangleIndex,const VecCoord& p, const Real tolerance=(Real)1e-5) const; 

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_TOPOLOGY_BEZIERTRIANGLESETGEOMETRYALGORITHMS_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_TOPOLOGY_API BezierTriangleSetGeometryAlgorithms<defaulttype::Vec3dTypes>;
extern template class SOFA_GENERAL_TOPOLOGY_API BezierTriangleSetGeometryAlgorithms<defaulttype::Vec2dTypes>;
extern template class SOFA_GENERAL_TOPOLOGY_API BezierTriangleSetGeometryAlgorithms<defaulttype::Vec1dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_TOPOLOGY_API BezierTriangleSetGeometryAlgorithms<defaulttype::Vec3fTypes>;
extern template class SOFA_GENERAL_TOPOLOGY_API BezierTriangleSetGeometryAlgorithms<defaulttype::Vec2fTypes>;
extern template class SOFA_GENERAL_TOPOLOGY_API BezierTriangleSetGeometryAlgorithms<defaulttype::Vec1fTypes>;
#endif
#endif

} // namespace topology

} // namespace component

} // namespace sofa

#endif
