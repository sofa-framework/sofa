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

#include <SofaBaseTopology/BezierTriangleSetTopologyContainer.h>
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

SOFA_DECL_CLASS(BezierTriangleSetTopologyContainer)
int BezierTriangleSetTopologyContainerClass = core::RegisterObject("Bezier Triangle set topology container")
        .add< BezierTriangleSetTopologyContainer >()
        ;


typedef std::pair<TriangleBezierIndex,BezierTriangleSetTopologyContainer::ElementTriangleIndex> ElementMapType;
typedef std::map<TriangleBezierIndex,BezierTriangleSetTopologyContainer::ElementTriangleIndex>::iterator ElementMapIterator;

typedef std::pair<TriangleBezierIndex,size_t> OffsetMapType;
typedef std::map<TriangleBezierIndex,size_t>::iterator OffsetMapIterator;
typedef std::map<TriangleBezierIndex,size_t>::const_iterator OffsetMapConstIterator;

BezierTriangleSetTopologyContainer::BezierTriangleSetTopologyContainer()
    : TriangleSetTopologyContainer()
    , d_degree(initData(&d_degree, (size_t)0,"degree", "Degree of Bezier Tetrahedra"))
    , d_numberOfTriangularPoints(initData(&d_numberOfTriangularPoints, (size_t) 0,"NbTriangularVertices", "Number of Triangular Vertices"))
   , d_isRationalSpline(initData(&d_isRationalSpline, SeqBools(),"isRational", "If a bezier triangle  is rational or integral"))
	 , d_weightArray(initData(&d_weightArray, SeqWeights(),"weights", "Array of weights for rational bezier triangles"))
{
    addAlias(&d_degree, "order");
}



void BezierTriangleSetTopologyContainer::init()
{
     d_degree.updateIfDirty(); // make sure m_Triangle is up to date
	 d_numberOfTriangularPoints.updateIfDirty();
	TriangleSetTopologyContainer::init(); // initialize the Triangle array
	reinit();
}
void BezierTriangleSetTopologyContainer::reinit()
{
 	if (d_degree.getValue()>0) {
		// clear previous entries if it exists
		elementMap.clear();
		localIndexMap.clear();
		bezierIndexArray.clear();
		edgeOffsetMap.clear();
		triangleOffsetMap.clear();

		// fill the elementMap and the 3 offsetMap in order to get the global index of an element from its Bezier index 
		BezierDegreeType degree=d_degree.getValue();
		BezierDegreeType i,j;
		size_t localIndex=0;
		// vertex index
		for (i=0;i<3;++i) {
			TriangleBezierIndex bti(0,0,0);
			bti[i]=degree;
			elementMap.insert(ElementMapType(bti,ElementTriangleIndex(i,-1,-1)));
			localIndexMap.insert(OffsetMapType(bti,localIndex));
			bezierIndexArray.push_back(bti);
			localIndex++;
		}
		// edge index
		if (degree>1) {
			for (i=0;i<3;++i) {

				for (j=1;j<degree;++j) {
					TriangleBezierIndex bti(0,0,0);
					bti[(i+1)%3]=degree-j;
					bti[(i+2)%3]=j;
					elementMap.insert(ElementMapType(bti,ElementTriangleIndex(-1,i,-1)));
					edgeOffsetMap.insert(OffsetMapType(bti,j-1));
					localIndexMap.insert(OffsetMapType(bti,localIndex));
					bezierIndexArray.push_back(bti);
					localIndex++;
				}
			}
		}
		// triangle index
		if (degree>2) {
			offsetToTriangleBezierIndexArray.clear();
			size_t ind=0;
			for (i=1;i<(degree-1);++i) {
				for (j=1;j<(degree-i);++j,++ind) {
					TriangleBezierIndex bti(0,0,0);
					bti[0]=i;bti[1]=j;
					bti[2]=degree-i-j;
					offsetToTriangleBezierIndexArray.push_back(bti);
					elementMap.insert(ElementMapType(bti,ElementTriangleIndex(-1,-1,0)));
					triangleOffsetMap.insert(OffsetMapType(bti,ind));
					//						std::cerr << "offsetMap["<<(size_t)bti[0]<<' '<<(size_t)bti[1]<<' '<<(size_t)bti[2]<<' '<<(size_t)bti[3]<<" ]= "<<ind<<std::endl;
					localIndexMap.insert(OffsetMapType(bti,localIndex));
					bezierIndexArray.push_back(bti);
					localIndex++;
				}

			}

		}
		// initialize the array of weights if necessary
		if (d_weightArray.getValue().empty()){
				helper::WriteOnlyAccessor<Data <SeqWeights> >  wa=d_weightArray;
				wa.resize(this->getNbPoints());
				std::fill(wa.begin(),wa.end(),(SReal)1);
			}
		// initialize the array of boolean indicating if the nature of the Bezier triangle if necessary
		if ((d_isRationalSpline.getValue().empty()) && (getNumberOfTriangles()>0)){
			helper::WriteOnlyAccessor<Data <SeqBools> >  isRationalSpline=d_isRationalSpline;
			isRationalSpline.resize(this->getNumberOfTriangles());
			std::fill(isRationalSpline.begin(),isRationalSpline.end(),false);
		}
		// manually creates the edge and triangle structures.
		createEdgeSetArray();
		createEdgesInTriangleArray();
	}
	if ((d_numberOfTriangularPoints.getValue()==0) && (getNumberOfTriangles()>0)){
		// compute the number of triangular point if it is not provided
		std::set<size_t> vertexSet;
		size_t i;
		// count the number of vertices involved in the list of triangles
		const sofa::helper::vector<Triangle> &tra=getTriangleArray();
		for (i=0;i<tra.size();++i) {
			vertexSet.insert(tra[i][0]);
			vertexSet.insert(tra[i][1]);
			vertexSet.insert(tra[i][2]);
		}
		d_numberOfTriangularPoints.setValue(vertexSet.size());

	}
}
SReal BezierTriangleSetTopologyContainer::getWeight(int i) const {
	return(d_weightArray.getValue()[i]);
}
bool BezierTriangleSetTopologyContainer::isRationalSpline(int i) const {
	 return(d_isRationalSpline.getValue()[i]);
}
const BezierTriangleSetTopologyContainer::SeqWeights & BezierTriangleSetTopologyContainer::getWeightArray() const {
	return(d_weightArray.getValue());
}

BezierDegreeType BezierTriangleSetTopologyContainer::getDegree() const{
	return d_degree.getValue();
}

size_t BezierTriangleSetTopologyContainer::getNumberOfTriangularPoints() const{
     return d_numberOfTriangularPoints.getValue();
}

size_t BezierTriangleSetTopologyContainer::getGlobalIndexOfBezierPoint(const TetraID triangleIndex,
     const TriangleBezierIndex id) {

         if (locationToGlobalIndexMap.empty()) {
             Triangle tr=getTriangle(triangleIndex);
             BezierDegreeType degree=d_degree.getValue();
             ElementMapIterator emi=elementMap.find(id);
             if (emi!=elementMap.end()) {
                 ElementTriangleIndex ei=(*emi).second;
                 if (ei[0]!= -1) {
                     // point is a vertex of the triangular mesh
                     return (tr[ei[0]]);
                 } else if (ei[1]!= -1) {
                     // point is on an edge of the triangular mesh
                     // the points on edges are stored after the Triangle vertices
                     // there are (degree-1) points store on each edge
                     // eit[ei[1]] = id of edge where the point is located
                     // ei[1] = the local index (<3) of the edge where the point is located
                     EdgesInTriangle eit=getEdgesInTriangle(triangleIndex);
                     Edge e=getEdge(eit[ei[1]]);
                     // test if the edge is along the right direction
                     OffsetMapIterator omi=edgeOffsetMap.find(id);
                     if (e[0]==tr[(ei[1]+1)%3]) {
                         return(getNumberOfTriangularPoints()+eit[ei[1]]*(degree-1)+(*omi).second);
                     } else {
                         // use the other direction
                         return(getNumberOfTriangularPoints()+eit[ei[1]]*(degree-1)+degree-2-(*omi).second);
                     }

                 } else if (ei[2]!= -1) {
                     // the points on edges are stored after the tetrahedron vertices
                     // there are (degree-1)*(degree-2)/2 points store on each edge
                     // eit[ei[1]] = id of edge where the point is located
                     // ei[1] = the local index (<6) of the edge where the point is located
                     OffsetMapIterator omi=triangleOffsetMap.find(id);
                     return(getNumberOfTriangularPoints()+getNumberOfEdges()*(degree-1)+triangleIndex*(degree-1)*(degree-2)/2+(*omi).second);
                 }
                 else
                 {
#ifndef NDEBUG
                    sout << "Unexpected error in [BezierTriangleSetTopologyContainer::getGlobalIndexOfBezierPoint]" << sendl;
#endif
                    return 0; //Warning fix but maybe the author of this code would want to print a more meaningful error message for this "ei[0] = ei[1] = ei[2] = -1" case ?
                 }
             } else {
#ifndef NDEBUG
                 sout << "Error. [BezierTriangleSetTopologyContainer::getGlobalIndexOfBezierPoint] Bezier Index "<< (sofa::defaulttype::Vec<3,int> )(id) <<" has not been recognized to be valid" << sendl;
#endif
                 return 0;
             }
         } else {
             std::map<ControlPointLocation,size_t>::const_iterator itgi;

             itgi=locationToGlobalIndexMap.find(ControlPointLocation(triangleIndex,id));
             if (itgi!=locationToGlobalIndexMap.end()) {
                 return(itgi->second);
             } else {
#ifndef NDEBUG
                 sout << "Error. [BezierTriangleSetTopologyContainer::getGlobalIndexOfBezierPoint] Cannot find global index of control point with TRBI  "<< (sofa::defaulttype::Vec<3,int> )(id) <<" and triangle index " << triangleIndex <<sendl;
#endif
                 return 0;
             }

         }
 }


TriangleBezierIndex BezierTriangleSetTopologyContainer::getTriangleBezierIndex(const size_t localIndex) const
{
	
	if (localIndex<bezierIndexArray.size()) {
		return bezierIndexArray[localIndex];
	} else {
#ifndef NDEBUG
		sout << "Error. [BezierTriangleSetTopologyContainer::getTriangleBezierIndex] Index "<< localIndex <<" is greater than the number "<< bezierIndexArray.size() <<" of control points." << sendl;
#endif
		TriangleBezierIndex id;
		return (id);
	}
}
size_t BezierTriangleSetTopologyContainer::getLocalIndexFromTriangleBezierIndex(const TriangleBezierIndex id) const {
	OffsetMapConstIterator omi=localIndexMap.find(id);
	if (omi==localIndexMap.end())
	{
#ifndef NDEBUG
		sout << "Error. [BezierTriangleSetTopologyContainer::getLocalIndexFromTriangleBezierIndex] Triangle Bezier Index "<< id  <<" is out of range." << sendl;
#endif
		return(0);
	} else {
		return ((*omi).second);
	}
}
sofa::helper::vector<TriangleBezierIndex> BezierTriangleSetTopologyContainer::getTriangleBezierIndexArray() const
{
	return (bezierIndexArray);
}
sofa::helper::vector<TriangleBezierIndex> BezierTriangleSetTopologyContainer::getTriangleBezierIndexArrayOfGivenDegree(const BezierDegreeType deg) const
{
	// vertex index
	size_t i,j;
	sofa::helper::vector<TriangleBezierIndex> tbiArray;
	for (i=0;i<3;++i) {
		TriangleBezierIndex bti(0,0,0);
		bti[i]=deg;
		tbiArray.push_back(bti);
	}
	// edge index
	if (deg>1) {
		for (i=0;i<3;++i) {
			for (j=1;j<deg;++j) {
				TriangleBezierIndex bti(0,0,0);
				bti[(i+1)%3]=(size_t)(deg-j);
				bti[(i+2)%3]=j;
				tbiArray.push_back(bti);
			}
		}
	}

	// Triangle index
	if (deg>2) {
		size_t ind=0;
        for (i=1;i<(BezierDegreeType)(deg-1);++i) {
			for (j=1;j<(deg-i);++j,++ind) {
				TriangleBezierIndex bti(0,0,0);
				bti[0]=i;bti[1]=j;
				bti[2]=deg-i-j;
				tbiArray.push_back(bti);
			}
		}
	}

	return(tbiArray);
}
sofa::helper::vector<BezierTriangleSetTopologyContainer::LocalTriangleIndex> BezierTriangleSetTopologyContainer::getMapOfTriangleBezierIndexArrayFromInferiorDegree() const
{
	BezierDegreeType degree=d_degree.getValue();
	sofa::helper::vector<TriangleBezierIndex> tbiDerivArray=getTriangleBezierIndexArrayOfGivenDegree(degree-1);
	sofa::helper::vector<TriangleBezierIndex> tbiLinearArray=getTriangleBezierIndexArrayOfGivenDegree(1);
	TriangleBezierIndex tbi;
	sofa::helper::vector<LocalTriangleIndex> correspondanceArray;
//	correspondanceArray.resize(tbiDerivArray.size());
	size_t i,j;
	for (i=0;i<tbiDerivArray.size();++i) {
		LocalTriangleIndex correspondance;
		for (j=0;j<3;++j) {
			tbi=tbiDerivArray[i]+tbiLinearArray[j];
			correspondance[j]=getLocalIndexFromTriangleBezierIndex(tbi);
		}
		correspondanceArray.push_back(correspondance);
	}
	return(correspondanceArray);
}
void BezierTriangleSetTopologyContainer::getGlobalIndexArrayOfBezierPointsInTriangle(const TetraID triangleIndex, VecPointID & indexArray)
{
	
	indexArray.clear();
	if (locationToGlobalIndexMap.empty()) {
		Triangle tr=getTriangle(triangleIndex);
		// vertex index
		size_t i,j;
		for (i=0;i<3;++i)
			indexArray.push_back(tr[i]);

		size_t offset;
		// edge index
		BezierDegreeType degree=d_degree.getValue();
		if (degree>1) {
			EdgesInTriangle eit=getEdgesInTriangle(triangleIndex);
			for (i=0;i<3;++i) {
				Edge e=getEdge(eit[i]);
				offset=getNumberOfTriangularPoints()+eit[i]*(degree-1);
				// check the order of the edge to be consistent with the Triangle
				if (e[0]==tr[(i+1)%3] ) {
					for (j=0;j<(size_t)(degree-1);++j) {
						indexArray.push_back(offset+j);
					}
				} else {
					int jj;
					for (jj=degree-2;jj>=0;--jj) {
						indexArray.push_back(offset+jj);
					}
				}
			}
		}



		// Triangle index
		if (degree>2) {
			size_t pointsPerTriangle=(degree-1)*(degree-2)/2;
			offset=getNumberOfTriangularPoints()+getNumberOfEdges()*(degree-1)+triangleIndex*pointsPerTriangle;
			size_t rank=0;
			for (i=0;i<(size_t)(degree-2);++i) {
				for (j=0;j<(degree-i-2);++j) {
					indexArray.push_back(offset+rank);
					rank++;
				}
			}
		}
	} else {
		size_t i;
		std::map<ControlPointLocation,size_t>::const_iterator itgi;
		for (i=0;i<bezierIndexArray.size();++i) {
			itgi=locationToGlobalIndexMap.find(ControlPointLocation(triangleIndex,bezierIndexArray[i]));
			if (itgi!=locationToGlobalIndexMap.end()) {
				indexArray.push_back(itgi->second);
			} else {
#ifndef NDEBUG
				sout << "Error. [BezierTriangleSetTopologyContainer::getGlobalIndexArrayOfBezierPointsInTriangle] Cannot find global index of control point with TRBI  "<< (sofa::defaulttype::Vec<3,int> )(bezierIndexArray[i]) <<" and triangle index " << triangleIndex <<sendl;
#endif
			}

		}

	}
	}
void BezierTriangleSetTopologyContainer::getLocationFromGlobalIndex(const size_t globalIndex, BezierTrianglePointLocation &location, 
	size_t &elementIndex, size_t &elementOffset)
{
	size_t gi=globalIndex;
	if (globalIndexToLocationMap.empty()) {
		if (gi<getNumberOfTriangularPoints()) {
			location=POINT;
			elementIndex=gi;
			elementOffset=0;
		} else {
			gi-=getNumberOfTriangularPoints();
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
#ifndef NDEBUG
					sout << "Error. [BezierTriangleSetTopologyContainer::getLocationFromGlobalIndex] Global Index "<< globalIndex <<" exceeds the number of Bezier Points" << sendl;
#endif
				}
			}

		}
	} else {
		std::multimap<size_t,ControlPointLocation>::const_iterator itcpl;
		itcpl=globalIndexToLocationMap.find(gi); 
		if (itcpl!=globalIndexToLocationMap.end()) {
			// get the local index and triangle index of that control point
			size_t offset=getLocalIndexFromTriangleBezierIndex(itcpl->second.second);
			// if its local index is less than 3 then it is a triangle vertex
			if (offset<3) {
				location=POINT;
				elementIndex=getTriangle(itcpl->second.first)[offset];
				elementOffset=0;
			} else {
				offset -= 3;
				BezierDegreeType degree=d_degree.getValue();
                if ((BezierDegreeType)offset<3*(degree-1)){
					location=EDGE;
					// get the id of the edge on which it lies
					elementIndex=getEdgesInTriangle(itcpl->second.first)[offset/(degree-1)];
					elementOffset=offset%(degree-1);
				} else {
					offset -= 3*(degree-1);
					location=TRIANGLE;
					elementIndex=itcpl->second.first;
					elementOffset=offset;
				}
			}
		} else {
			location=NONE;
			elementIndex=0;
			elementOffset=0;
		}
	}
}
sofa::helper::vector<BezierTriangleSetTopologyContainer::LocalTriangleIndex> BezierTriangleSetTopologyContainer::getLocalIndexSubtriangleArray() const {
	sofa::helper::vector<LocalTriangleIndex> subtriangleArray;
	BezierDegreeType degree=d_degree.getValue();
	TriangleBezierIndex tbi1,tbi2,tbi3;
	LocalTriangleIndex lti;
	for (size_t i=1;i<=degree;++i) {
		for (size_t j=0;j<(degree-i+1);++j) {
			tbi1=TriangleBezierIndex(i,j,degree-i-j);
			tbi2=TriangleBezierIndex(i-1,j+1,degree-i-j);
			tbi3=TriangleBezierIndex(i-1,j,degree-i-j+1);
			
			lti[0]=getLocalIndexFromTriangleBezierIndex(tbi1);
			lti[1]=getLocalIndexFromTriangleBezierIndex(tbi2);
			lti[2]=getLocalIndexFromTriangleBezierIndex(tbi3);
			subtriangleArray.push_back(lti);
			if ((i+j)<degree) {
				tbi3=TriangleBezierIndex(i,j+1,degree-i-j-1);
				lti[2]=lti[1];	
				lti[1]=getLocalIndexFromTriangleBezierIndex(tbi3);
				subtriangleArray.push_back(lti);
			}
		}
	}
    assert(subtriangleArray.size()==(size_t)((degree+1)*(degree+1)));
	return(subtriangleArray);

}
sofa::helper::vector<BezierTriangleSetTopologyContainer::LocalTriangleIndex> BezierTriangleSetTopologyContainer::getLocalIndexSubtriangleArrayOfGivenDegree(const BezierDegreeType deg)  const {

	sofa::helper::vector<TriangleBezierIndex> tbia=getTriangleBezierIndexArrayOfGivenDegree(deg);
	// create a local map for indexing
	std::map<TriangleBezierIndex,size_t> tmpLocalIndexMap;
	size_t i;
	for (i=0;i<tbia.size();++i)
		tmpLocalIndexMap.insert(OffsetMapType(tbia[i],i));
	// now create the array of subtriangles
	sofa::helper::vector<LocalTriangleIndex> subtriangleArray;
	
	TriangleBezierIndex tbi[3];
	size_t k;
	LocalTriangleIndex lti;
	std::map<TriangleBezierIndex,size_t>::iterator omi;
	for ( i=1;i<=deg;++i) {
		for (size_t j=0;j<(deg-i+1);++j) {
			tbi[0]=TriangleBezierIndex(i,j,deg-i-j);
			tbi[1]=TriangleBezierIndex(i-1,j+1,deg-i-j);
			tbi[2]=TriangleBezierIndex(i-1,j,deg-i-j+1);
			for (k=0;k<3;++k) {
				omi=tmpLocalIndexMap.find(tbi[k]);
				if (omi==tmpLocalIndexMap.end())
				{
#ifndef NDEBUG
					sout << "Error. [BezierTriangleSetTopologyContainer::getLocalIndexSubtriangleArrayOfGivenDegree(const BezierDegreeType deg) ] Triangle Bezier Index "<< tbi[k]  <<" is out of range." << sendl;
#endif
				} else {
					lti[k]= (*omi).second;
				}
			}

			subtriangleArray.push_back(lti);
			if ((i+j)<deg) {
				tbi[2]=TriangleBezierIndex(i,j+1,deg-i-j-1);
				lti[2]=lti[1];
				omi=tmpLocalIndexMap.find(tbi[2]);
				if (omi==tmpLocalIndexMap.end())
				{
#ifndef NDEBUG
					sout << "Error. [BezierTriangleSetTopologyContainer::getLocalIndexSubtriangleArrayOfGivenDegree(const BezierDegreeType deg) ] Triangle Bezier Index "<< tbi[2]  <<" is out of range." << sendl;
#endif
				} else {
					lti[1]= (*omi).second;
				}

				subtriangleArray.push_back(lti);
			}
		}
	}
	assert(subtriangleArray.size()==((deg)*(deg)));
	return(subtriangleArray);
}
void BezierTriangleSetTopologyContainer::getEdgeBezierIndexFromEdgeOffset(size_t offset, EdgeBezierIndex &ebi){
	assert(offset<d_degree.getValue());
	ebi[0]=offset+1;
	ebi[1]=d_degree.getValue()-offset-1;
}
void BezierTriangleSetTopologyContainer::getTriangleBezierIndexFromTriangleOffset(size_t offset, TriangleBezierIndex &tbi){
	assert(offset<(d_degree.getValue()-1)*(d_degree.getValue()-2)/2);
	tbi=offsetToTriangleBezierIndexArray[offset];
}
bool BezierTriangleSetTopologyContainer::checkBezierPointTopology()
{
	#ifndef NDEBUG
	size_t nTrians,elem;
	BezierDegreeType degree=d_degree.getValue();
	// check the total number of vertices.
    assert(getNbPoints()==(int)(getNumberOfTriangularPoints()+getNumberOfEdges()*(degree-1)+getNumberOfTriangles()*(degree-1)*(degree-2)/2));
	sofa::helper::vector<TriangleBezierIndex> tbiArray=getTriangleBezierIndexArray();
	VecPointID indexArray;
	BezierTrianglePointLocation location; 
    size_t elementIndex, elementOffset/*,localIndex*/;
	for (nTrians=0;nTrians<getNumberOfTriangles();++nTrians) {
		indexArray.clear();
		getGlobalIndexArrayOfBezierPointsInTriangle(nTrians,indexArray);
		// check the number of control points per Triangle is correct
        assert(indexArray.size()==(size_t)(3+3*(degree-1)+(degree-1)*(degree-2)/2));
		for(elem=0;elem<indexArray.size();++elem) {
			size_t globalIndex=getGlobalIndexOfBezierPoint(nTrians,tbiArray[elem]);
			// check that getGlobalIndexOfBezierPoint and getGlobalIndexArrayOfBezierPointsInTriangle give the same answer
			assert(globalIndex==indexArray[elem]);

            TriangleBezierIndex tbi=getTriangleBezierIndex(elem);

			assert(elem==getLocalIndexFromTriangleBezierIndex(tbi));
			// check that getTriangleBezierIndex is consistent with getTriangleBezierIndexArray
			assert(tbiArray[elem][0]==tbi[0]);
			assert(tbiArray[elem][1]==tbi[1]);
			assert(tbiArray[elem][2]==tbi[2]);
			// check that getLocationFromGlobalIndex is consistent with 
			getLocationFromGlobalIndex(globalIndex,location,elementIndex,elementOffset);
			if (elem<3) {
				assert(location==POINT);
				assert(elementIndex==getTriangle(nTrians)[elem]);
				assert(elementOffset==0);
			}
			else if (elem<(size_t)(3+3*(degree-1))){
				assert(location==EDGE);
				assert(elementIndex==getEdgesInTriangle(nTrians)[(elem-3)/(degree-1)]);
			}
		}

	}
	if (locationToGlobalIndexMap.size()>0) {
		// check consistency between both maps
		assert(locationToGlobalIndexMap.size()==globalIndexToLocationMap.size());
		std::map<ControlPointLocation,size_t>::iterator itcpl;
		std::map<size_t,ControlPointLocation>::iterator itgi;
		std::pair<std::map<size_t,ControlPointLocation>::iterator,std::map<size_t,ControlPointLocation>::iterator> itgir;
		for (itcpl=locationToGlobalIndexMap.begin();itcpl!=locationToGlobalIndexMap.end();++itcpl) {
			itgir=globalIndexToLocationMap.equal_range(itcpl->second);
			assert(itgir.first!=itgir.second);
			for (itgi=itgir.first;itgi->second!=itcpl->first && itgi!=itgir.second;++itgi);
			assert(itgi->second==itcpl->first);
		}
	}
	#endif
	return( true);
}

} // namespace topology

} // namespace component

} // namespace sofa
