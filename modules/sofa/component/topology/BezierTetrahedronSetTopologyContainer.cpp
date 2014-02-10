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

#include <sofa/component/topology/BezierTetrahedronSetTopologyContainer.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>


namespace sofa
{

namespace component
{

namespace topology
{

using namespace std;
using namespace sofa::defaulttype;

SOFA_DECL_CLASS(BezierTetrahedronSetTopologyContainer)
int BezierTetrahedronSetTopologyContainerClass = core::RegisterObject("Bezier Tetrahedron set topology container")
        .add< BezierTetrahedronSetTopologyContainer >()
        ;

const unsigned int edgesInTetrahedronArray[6][2] = {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};
///convention triangles in tetra (orientation interior)
const unsigned int trianglesInTetrahedronArray[4][3]= {{1,2,3}, {0,3,2}, {1,3,0},{0,2,1}};
typedef sofa::helper::fixed_array<int,4> ElementIndex;

std::map<TetrahedronBezierIndex,ElementIndex> elementMap;
typedef std::pair<TetrahedronBezierIndex,ElementIndex> ElementMapType;
typedef std::map<TetrahedronBezierIndex,ElementIndex>::iterator ElementMapIterator;

typedef std::pair<TetrahedronBezierIndex,size_t> OffsetMapType;
typedef std::map<TetrahedronBezierIndex,size_t>::iterator OffsetMapIterator;
std::map<TetrahedronBezierIndex,size_t> edgeOffsetMap;
std::map<TetrahedronBezierIndex,size_t> triangleOffsetMap;
std::map<TetrahedronBezierIndex,size_t> tetrahedronOffsetMap;

std::map<TetrahedronBezierIndex,size_t> localIndexMap;
sofa::helper::vector<TetrahedronBezierIndex> bezierIndexArray;

BezierTetrahedronSetTopologyContainer::BezierTetrahedronSetTopologyContainer()
    : TetrahedronSetTopologyContainer()
    , d_degree(initData(&d_degree, "degree", "Degree of Bezier Tetrahedra"))
{
    addAlias(&d_degree, "order");
}



void BezierTetrahedronSetTopologyContainer::init()
{
    d_degree.updateIfDirty(); // make sure m_tetrahedron is up to date
	TetrahedronSetTopologyContainer::init(); // initialize the tetrahedron array

	// fill the elementMap and the 3 offsetMap in order to get the global index of an element from its Bezier index 
	BezierDegreeType degree=d_degree.getValue();
	BezierDegreeType i,j,k,l;
	size_t localIndex=0;
	// vertex index
	for (i=0;i<4;++i) {
		TetrahedronBezierIndex bti(0,0,0,0);
		bti[i]=degree;
		elementMap.insert(ElementMapType(bti,ElementIndex(i,-1,-1,-1)));
		localIndexMap.insert(OffsetMapType(bti,localIndex));
		bezierIndexArray.push_back(bti);
		localIndex++;
	}
	// edge index
	if (degree>1) {
		for (i=0;i<6;++i) {

			for (j=1;j<degree;++j) {
				TetrahedronBezierIndex bti(0,0,0,0);
				bti[edgesInTetrahedronArray[i][0]]=degree-j;
				bti[edgesInTetrahedronArray[i][1]]=j;
				elementMap.insert(ElementMapType(bti,ElementIndex(-1,i,-1,-1)));
				edgeOffsetMap.insert(OffsetMapType(bti,j-1));
				localIndexMap.insert(OffsetMapType(bti,localIndex));
				bezierIndexArray.push_back(bti);
				localIndex++;
			}
		}
	}
	// triangle index
	if (degree>2) {
		size_t ind;
		for (i=0;i<4;++i) {
			for (ind=0,j=1;j<(degree-1);++j) {
				for (k=1;k<(degree-j);++k,++ind) {
					TetrahedronBezierIndex bti(0,0,0,0);
					bti[trianglesInTetrahedronArray[i][0]]=j;
					bti[trianglesInTetrahedronArray[i][1]]=k;
					bti[trianglesInTetrahedronArray[i][2]]=degree-j-k;
					elementMap.insert(ElementMapType(bti,ElementIndex(-1,-1,i,-1)));
					triangleOffsetMap.insert(OffsetMapType(bti,ind));
					localIndexMap.insert(OffsetMapType(bti,localIndex));
					bezierIndexArray.push_back(bti);
					localIndex++;
				}
			}
		}
	}
	// tetrahedron index
	if (degree>3) {
		size_t ind=0;
		for (i=1;i<(degree-2);++i) {
			for (j=1;j<(degree-i-1);++j) {
				for (k=1;k<(degree-j-i);++k,++ind) {
					TetrahedronBezierIndex bti(0,0,0,0);
					bti[0]=i;bti[1]=j;bti[2]=k;
					bti[3]=degree-i-j-k;
					elementMap.insert(ElementMapType(bti,ElementIndex(-1,-1,-1,0)));
					tetrahedronOffsetMap.insert(OffsetMapType(bti,ind));
					localIndexMap.insert(OffsetMapType(bti,localIndex));
					bezierIndexArray.push_back(bti);
					localIndex++;
				}
			}
		}
	}

	/*

	for (i=0;i<=degree;i++) {
		for (j=0;j<=(degree-i);j++) {
			for (k=0;k<=(degree-i-j);k++) {
				for (l=0;l<=(degree-i-j-k);l++) {
					if ((i==degree) || (j==degree) || (k==degree) || (l==degree)) {
						offsetType off(0,0,0,0);
						offsetMap.insert(OffsetMapType(TetrahedronBezierIndex(i,j,k,l),off));
					} else if (
				}
			}
		}
	}
	*/
}
BezierDegreeType BezierTetrahedronSetTopologyContainer::getDegree() const{
	return d_degree.getValue();
}
 size_t BezierTetrahedronSetTopologyContainer::getNumberOfTetrahedralPoints() const{
	 return numberOfTetrahedralPoints.getValue();
 }
size_t BezierTetrahedronSetTopologyContainer::getGlobalIndexOfBezierPoint(const size_t tetrahedronIndex,
	const TetrahedronBezierIndex id) {

		Tetrahedron tet=getTetrahedron(tetrahedronIndex);
		BezierDegreeType degree=d_degree.getValue();
		ElementMapIterator emi=elementMap.find(id);
		if (emi!=elementMap.end()) {
			ElementIndex ei=(*emi).second;
			if (ei[0]!= -1) {
				// point is a vertex of the tetrahedral mesh
				return (tet[ei[0]]);
			} else if (ei[1]!= -1) {
				// point is on an edge of the tetrahedral mesh
				// the points on edges are stored after the tetrahedron vertices
				// there are (degree-1) points store on each edge
				// eit[ei[1]] = id of edge where the point is located
				// ei[1] = the local index (<6) of the edge where the point is located 
				EdgesInTetrahedron eit=getEdgesInTetrahedron(tetrahedronIndex);
				Edge e=getEdge(eit[ei[1]]);
				// test if the edge is along the right direction
				OffsetMapIterator omi=edgeOffsetMap.find(id);
				if (e[0]==tet[edgesInTetrahedronArray[ei[1]][0]]) {
					return(getNumberOfTetrahedralPoints()+eit[ei[1]]*(degree-1)+(*omi).second);
				} else {
					// use the other direction
					return(getNumberOfTetrahedralPoints()+eit[ei[1]]*(degree-1)+degree-2-(*omi).second);
				}
			} else if (ei[2]!= -1) {
				// point is on an edge of the tetrahedral mesh
				TrianglesInTetrahedron tit=getTrianglesInTetrahedron(tetrahedronIndex);
				// the points on triangles are stored after the Bezier points on edges
				// there are (degree-1)*(degree-2)/2 points store on each triangle
				// eit[ei[2]] = id of triangle where the point is located
				// ei[2] = the local index (<4) of the triangle where the point is located 
				Triangle tr=getTriangle(tit[ei[2]]);
				Triangle indexTriangle;
				size_t k,j,i;
				for (k=0;(tr[0]!=tet[trianglesInTetrahedronArray[i][k]]);++k);
				indexTriangle[0]=k;
				if (tr[1]==tet[trianglesInTetrahedronArray[i][(k+1)%3]]) {
					indexTriangle[1]=(k+1)%3;
					indexTriangle[2]=(k+2)%3;
				} else {
					indexTriangle[2]=(k+1)%3;
					indexTriangle[1]=(k+2)%3;
				}
				i=ei[2];
				TetrahedronBezierIndex bti(0,0,0,0);
				bti[trianglesInTetrahedronArray[i][indexTriangle[0]]]=id[trianglesInTetrahedronArray[i][0]];
				bti[trianglesInTetrahedronArray[i][indexTriangle[1]]]=id[trianglesInTetrahedronArray[i][1]];
				bti[trianglesInTetrahedronArray[i][indexTriangle[2]]]=id[trianglesInTetrahedronArray[i][2]];
				OffsetMapIterator omi=localIndexMap.find(bti);
				return(getNumberOfTetrahedralPoints()+getNumberOfEdges()*(degree-1)+tit[i]*(degree-1)*(degree-2)/2+(*omi).second);
			} else if (ei[3]!= -1) {
				// the points on edges are stored after the tetrahedron vertices
				// there are (degree-1)*(degree-2)/2 points store on each edge
				// eit[ei[1]] = id of edge where the point is located
				// ei[1] = the local index (<6) of the edge where the point is located 
				OffsetMapIterator omi=tetrahedronOffsetMap.find(id);
				return(getNumberOfTetrahedralPoints()+getNumberOfEdges()*(degree-1)+getNumberOfTriangles()*(degree-1)*(degree-2)/2+tetrahedronIndex*(degree-1)*(degree-2)*(degree-3)/6+(*omi).second);
			}
		} else {
#ifndef NDEBUG
			sout << "Error. [BezierTetrahedronSetTopologyContainer::getIndexOfBezierPoint] Bezier Index "<< id <<" has not been recognized to be valid" << sendl;
#endif
		}
}

TetrahedronBezierIndex BezierTetrahedronSetTopologyContainer::getTetrahedronBezierIndex(const size_t localIndex) const
{
	
	if (localIndex<bezierIndexArray.size()) {
		return bezierIndexArray[localIndex];
	} else {
#ifndef NDEBUG
		sout << "Error. [BezierTetrahedronSetTopologyContainer::getBezierIndexInTetrahedron] Index "<< localIndex <<" is greater than the number "<< bezierIndexArray.size() <<" of control points." << sendl;
#endif
		TetrahedronBezierIndex id;
		return (id);
	}
}
sofa::helper::vector<TetrahedronBezierIndex> BezierTetrahedronSetTopologyContainer::getTetrahedronBezierIndexArray() const
{
	return (bezierIndexArray);
}
void BezierTetrahedronSetTopologyContainer::getGlobalIndexArrayOfBezierPointsInTetrahedron(const size_t tetrahedronIndex, VecPointID & indexArray) 
{
	Tetrahedron tet=getTetrahedron(tetrahedronIndex);
	indexArray.clear();
	// vertex index
	size_t i,j,k;
	for (i=0;i<4;++i)
		indexArray.push_back(tet[i]);

	size_t offset;
	// edge index
	BezierDegreeType degree=d_degree.getValue();
	if (degree>1) {
		EdgesInTetrahedron eit=getEdgesInTetrahedron(tetrahedronIndex);
		for (i=0;i<6;++i) {
			Edge e=getEdge(eit[i]);
			offset=getNumberOfTetrahedralPoints()+eit[i]*(degree-1);
			// check the order of the edge to be consistent with the tetrahedron
			if (e[0]==tet[edgesInTetrahedronArray[i][0]]) {
				for (j=0;j<(degree-1);++j) {
					indexArray.push_back(offset+j);
				}
			} else {
				for (j=degree-2;j>=0;--j) {
					indexArray.push_back(offset+j);
				}
			}
		}
	}
	// triangle index
	if (degree>2) {
		TrianglesInTetrahedron tit=getTrianglesInTetrahedron(tetrahedronIndex);
		for (i=0;i<4;++i) {
			offset=getNumberOfTetrahedralPoints()+getNumberOfEdges()*(degree-1);
			Triangle tr=getTriangle(tit[i]);
			Triangle indexTriangle;
			for (k=0;(tr[0]!=tet[trianglesInTetrahedronArray[i][k]]);++k);
			indexTriangle[0]=k;
			if (tr[1]==tet[trianglesInTetrahedronArray[i][(k+1)%3]]) {
				indexTriangle[1]=(k+1)%3;
				indexTriangle[2]=(k+2)%3;
			} else {
				indexTriangle[2]=(k+1)%3;
				indexTriangle[1]=(k+2)%3;
			}
			for (j=1;j<(degree-1);++j) {
				for (k=1;k<(degree-j);++k) {
					TetrahedronBezierIndex bti(0,0,0,0);
					bti[trianglesInTetrahedronArray[i][indexTriangle[0]]]=j;
					bti[trianglesInTetrahedronArray[i][indexTriangle[1]]]=k;
					bti[trianglesInTetrahedronArray[i][indexTriangle[2]]]=degree-j-k;
					OffsetMapIterator omi=localIndexMap.find(bti);
					indexArray.push_back(offset+(*omi).second);
				}
			}
		}
	}
	/*
			// check the origin of the triangle
			if (tr[0]==tet[trianglesInTetrahedronArray[i][0]]) {
				// check the second vertex of the triangle
				if (tr[1]==tet[trianglesInTetrahedronArray[i][1]]){
					rank=0;
					for (j=0;j<(degree-2);++j) {
						for (k=0;k<(degree-j-2);++k) {
							indexArray.push_back(offset+rank);
							rank++;
						}
					}
				} else {
					int rank= -1;
					for (j=0;j<(degree-2);++j) {
						rank+=degree-2-j;
						for (k=0;k<(degree-j-2);++k) {
							indexArray.push_back((PointID)(offset+j*(degree-2)+rank-k));
						}
					}
				}
			} else if (tr[1]==tet[trianglesInTetrahedronArray[i][0]]) {
				// check the second vertex of the triangle
				if (tr[3]==tet[trianglesInTetrahedronArray[i][1]]){
					for (j=0;j<(degree-2);++j) { 
						rank=degree-2;
						for (k=0;k<(degree-j-2);++k) {
							indexArray.push_back(offset+j+rank);
							rank+=rank-1;
						}
					}
				} else {
					for (j=0;j<(degree-2);++j) { 
						rank=degree-2;
						for (k=0;k<(degree-j-2);++k) {
							indexArray.push_back(offset+j+rank);
							rank+=rank-1;
						}
					}
				}
			} else if (tr[2]==tet[trianglesInTetrahedronArray[i][0]]) {

			}
		}
	} */

	// tetrahedron index
	if (degree>3) {
		offset=getNumberOfTetrahedralPoints()+getNumberOfEdges()*(degree-1)+getNumberOfTriangles()*(degree-1)*(degree-2)/2;
		size_t rank=0;
		for (i=0;i<(degree-3);++i) {
			for (j=0;j<(degree-i-3);++j) {
				for (k=0;k<(degree-j-i-3);++k) {
					indexArray.push_back(offset+rank);
					rank++;
				}
			}
		}
	}

}
void BezierTetrahedronSetTopologyContainer::getLocationFromGlobalIndex(const size_t globalIndex, BezierTetrahedronPointLocation &location, 
	size_t &elementIndex, size_t &elementOffset)
{
	size_t gi=globalIndex;
	if (gi<getNumberOfTetrahedralPoints()) {
		location=POINT;
		elementIndex=gi;
		elementOffset=0;
	} else {
		gi-=getNumberOfTetrahedralPoints();
		BezierDegreeType degree=d_degree.getValue();
		if (gi<(getNumberOfEdges()*(degree-1))) {
			location=EDGE;
			elementIndex=gi/(degree-1);
			elementOffset=gi%(degree-1);
		} else {
			gi-=getNumberOfEdges()*(degree-1);
			size_t pointsPerTriangle=(degree-1)*(degree-2)/2;
			if (gi<(getNumberOfTriangles()*pointsPerTriangle)) {
				location=TRIANGLE;
				elementIndex=gi/pointsPerTriangle;
				elementOffset=gi%pointsPerTriangle;
			} else {
				gi-=getNumberOfTriangles()*pointsPerTriangle;
				size_t pointsPerTetrahedron=(degree-1)*(degree-2)*(degree-3)/6;
				if (gi<(getNumberOfTetrahedra()*pointsPerTetrahedron)) {
					location=TETRAHEDRON;
					elementIndex=gi/pointsPerTetrahedron;
					elementOffset=gi%pointsPerTetrahedron;
				}  else {
#ifndef NDEBUG
					sout << "Error. [BezierTetrahedronSetTopologyContainer::getLocationFromGlobalIndex] Global Index "<< globalIndex <<" exceed the number of Bezier Points" << sendl;
#endif
				}
			}
		}
	}
}

bool BezierTetrahedronSetTopologyContainer::checkBezierPointTopology()
{
	return( true);
}

} // namespace topology

} // namespace component

} // namespace sofa
