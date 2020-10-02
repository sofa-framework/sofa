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
#ifndef SOFA_COMPONENT_TOPOLOGY_BEZIERTRIANGLESETTOPOLOGYCONTAINER_H
#define SOFA_COMPONENT_TOPOLOGY_BEZIERTRIANGLESETTOPOLOGYCONTAINER_H

#include "config.h"

#include <SofaBaseTopology/TriangleSetTopologyContainer.h>

namespace sofa
{

namespace component
{

namespace topology
{
class BezierTriangleSetTopologyModifier;

//using core::topology::BaseMeshTopology;



typedef unsigned char BezierDegreeType;
typedef sofa::defaulttype::Vec<3,BezierDegreeType> TriangleBezierIndex;


/** a class that stores a set of Bezier tetrahedra and provides access with adjacent triangles, edges and vertices 
A Bezier Tetrahedron has exactly the same topology as a Tetrahedron but with additional (control) points on its edges, triangles and inside 
We use a Vec4D to number the control points inside  a Bezier tetrahedron */
class SOFA_GENERAL_TOPOLOGY_API BezierTriangleSetTopologyContainer : public TriangleSetTopologyContainer
{
public:
	 SOFA_CLASS(BezierTriangleSetTopologyContainer,TriangleSetTopologyContainer);

	typedef BaseMeshTopology::PointID		            	PointID;
	typedef BaseMeshTopology::EdgeID		               	EdgeID;
	typedef BaseMeshTopology::TriangleID	               TriangleID;
	typedef BaseMeshTopology::Edge		        	         Edge;
	typedef BaseMeshTopology::Triangle	        	         Triangle;
	typedef BaseMeshTopology::SeqTriangles	        	      SeqTriangles;
	typedef BaseMeshTopology::EdgesInTriangle	         	EdgesInTriangle;
	typedef BaseMeshTopology::TrianglesAroundVertex    	TrianglesAroundVertex;
	typedef BaseMeshTopology::TrianglesAroundEdge        	TrianglesAroundEdge;
	typedef sofa::helper::vector<TriangleID>                  VecTriangleID;
	typedef sofa::helper::vector<PointID>					  VecPointID;
	typedef sofa::defaulttype::Vec<2,BezierDegreeType>		EdgeBezierIndex;

	typedef sofa::defaulttype::Vec<3,int> ElementTriangleIndex;
	typedef sofa::defaulttype::Vec<3,size_t> LocalTriangleIndex;
	typedef std::pair<size_t,TriangleBezierIndex> ControlPointLocation;
	typedef sofa::helper::vector<SReal> SeqWeights;
	typedef sofa::helper::vector<bool> SeqBools;

    friend class BezierTriangleSetTopologyModifier;
	friend class Mesh2BezierTopologicalMapping;
	friend class BezierTetra2BezierTriangleTopologicalMapping;



protected:
    BezierTriangleSetTopologyContainer();

    virtual ~BezierTriangleSetTopologyContainer() {}
public:
    virtual void init();
	// build some maps specific of the degree of the tetrahedral elements.
	virtual void reinit();

    /// Bezier Specific Information Topology API
    /// @{
public :
	/// the degree of the Bezier Tetrahedron 1=linear, 2=quadratic...
	Data <size_t> d_degree;
	/// the number of control points corresponding to the vertices of the triangle mesh (different from the total number of points)
    Data<size_t> d_numberOfTriangularPoints;
	/// whether the Bezier triangles are integral (false = classical Bezier splines) or rational splines (true)
	Data <SeqBools> d_isRationalSpline;
	/// the array of weights for rational splines
	Data <SeqWeights > d_weightArray;
public :
	// specifies where a Bezier Point can lies with respect to the underlying tetrahedral mesh
	enum BezierTrianglePointLocation
    {
        POINT = 0,
        EDGE =1 ,
        TRIANGLE = 2,
		NONE = 3
    };
	/// get the Degree of the Bezier Tetrahedron 
	BezierDegreeType getDegree() const;
	/// get the number of control points corresponding to the vertices of the triangle mesh 
	size_t getNumberOfTriangularPoints() const;
	/// get the global index of the Bezier  point associated with a given tetrahedron index and given its 4D Index   
	size_t getGlobalIndexOfBezierPoint(const TriangleID tetrahedronIndex,const TriangleBezierIndex id) ;
	/// get the indices of all control points associated with a given triangle
	void getGlobalIndexArrayOfBezierPointsInTriangle(const TriangleID triangleIndex, VecPointID & indexArray) ;
	/// return the Bezier index given the local index in a triangle
	TriangleBezierIndex getTriangleBezierIndex(const size_t localIndex) const;
	/// get the Triangle Bezier Index Array of degree d
	sofa::helper::vector<TriangleBezierIndex> getTriangleBezierIndexArray() const;
	/// get the Triangle Bezier Index Array of a given degree 
	sofa::helper::vector<TriangleBezierIndex> getTriangleBezierIndexArrayOfGivenDegree(const BezierDegreeType deg) const;
	/** create an array which maps the local index of a Triangle Bezier Index of degree d-1
	into a local index of a TBI of degree d by adding respectively (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) **/
	sofa::helper::vector<LocalTriangleIndex> getMapOfTriangleBezierIndexArrayFromInferiorDegree() const;
	/** return the array describing each of the  (degree+1)*(degree+1) subtriangles with local indices ( i.e. indices between 0 and (degree+1)*(degree+2)/2  ) */ 
	sofa::helper::vector<LocalTriangleIndex> getLocalIndexSubtriangleArray() const;
	/** return the array describing each of the  (deg+1)*(deg+1) subtriangles with local indices ( i.e. indices between 0 and (deg+1)*(deg+2)/2  ) */ 
	sofa::helper::vector<LocalTriangleIndex> getLocalIndexSubtriangleArrayOfGivenDegree(const BezierDegreeType deg)  const;
	/// return the local index in a tetrahedron from a tetrahedron Bezier index (inverse of getTetrahedronBezierIndex())
	size_t getLocalIndexFromTriangleBezierIndex(const TriangleBezierIndex id) const;
	/// return the location, the element index and offset from the global index of a point
	void getLocationFromGlobalIndex(const size_t globalIndex, BezierTrianglePointLocation &location, 
		size_t &elementIndex, size_t &elementOffset) ;
	/// convert the edge offset into a EdgeBezierIndex
	void getEdgeBezierIndexFromEdgeOffset(size_t offset, EdgeBezierIndex &ebi);
	/// convert the triangle offset into a TriangleBezierIndex
	void getTriangleBezierIndexFromTriangleOffset(size_t offset, TriangleBezierIndex &tbi);
	/// check the Bezier Point Topology
	bool checkBezierPointTopology();
	 /// @}
	/** \brief Returns the weight coordinate of the ith DOF. */
	virtual SReal getWeight(int i) const;
	/// returns the array of weights
	const SeqWeights & getWeightArray() const;
	// if the Bezier triangle is rational or integral
	bool isRationalSpline(int i) const;

protected:
	/** Map which provides the global index of a control point knowing its location (i.e. triangle index and its TriangleBezierIndex).
	This is empty by default since there is a default layout of control points based on edge and triangles indices */
	std::map<ControlPointLocation,size_t> locationToGlobalIndexMap;
	/** Map which provides the  location (i.e. triangle index and its TriangleBezierIndex) of a control point knowing its  global index.
	Note that the location may not be unique.
	This is empty by default since there is a default layout of control points based on edge and triangles indices */
	std::multimap<size_t,ControlPointLocation> globalIndexToLocationMap;


	/// Map which provides the location (point, edge, Triangle) of a control point given its Triangle Bezier index
	std::map<TriangleBezierIndex,ElementTriangleIndex> elementMap;
	/// Map which provides the offset in the DOF vector for a control point lying on an edge 
	std::map<TriangleBezierIndex,size_t> edgeOffsetMap;
	/// Map which provides the offset in the DOF vector for a control point lying on a triangle 
	std::map<TriangleBezierIndex,size_t> triangleOffsetMap;

	/// Map which provides the rank in a control point from the array outputted by getGlobalIndexArrayOfBezierPointsInTriangle (consistent with bezierIndexArray) 
	std::map<TriangleBezierIndex,size_t> localIndexMap;
	/// array of the Triangle Bezier index outputed by the function getGlobalIndexArrayOfBezierPointsInTriangle()
	sofa::helper::vector<TriangleBezierIndex> bezierIndexArray;
	/// array of the Triangle Bezier index outputed by the function getGlobalIndexArrayOfBezierPointsInTriangle()
	sofa::helper::vector<TriangleBezierIndex> reducedDegreeBezierIndexArray;
	/// convert triangle offset into triangle bezier index
	sofa::helper::vector<TriangleBezierIndex> offsetToTriangleBezierIndexArray;



};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
