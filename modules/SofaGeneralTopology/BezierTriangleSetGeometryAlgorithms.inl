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
#ifndef SOFA_COMPONENT_TOPOLOGY_BEZIERTRIANGLESETGEOMETRYALGORITHMS_INL
#define SOFA_COMPONENT_TOPOLOGY_BEZIERTRIANGLESETGEOMETRYALGORITHMS_INL

#include <SofaGeneralTopology/BezierTriangleSetGeometryAlgorithms.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseTopology/CommonAlgorithms.h>
#include <sofa/helper/rmath.h>
namespace sofa
{

namespace component
{

namespace topology
{

double multinomial(const size_t n,const TriangleBezierIndex tbiIn)
 {
	size_t i,ival;
	TriangleBezierIndex tbi=tbiIn;
	// divide n! with the largest of the multinomial coefficient
	std::sort(tbi.begin(),tbi.end());
	ival=1;
	for (i=n;i>tbi[2];--i){
		ival*=i;
	}
    return(((double)ival)/(sofa::helper::factorial(tbi[0])*sofa::helper::factorial(tbi[1])));
 }
template< class DataTypes>
 BezierTriangleSetGeometryAlgorithms< DataTypes >::BezierTriangleSetGeometryAlgorithms() : 
TriangleSetGeometryAlgorithms<DataTypes>()
        ,drawControlPointsEdges (core::objectmodel::Base::initData(&drawControlPointsEdges, (bool) false, "drawControlPointsEdges", "Debug : draw Control point edges "))
        ,drawSmoothEdges (core::objectmodel::Base::initData(&drawSmoothEdges, (bool) false, "drawSmoothEdges", "Debug : draw Bezier curves as edges of the  Bezier triangle"))
        ,drawControlPoints (core::objectmodel::Base::initData(&drawControlPoints, (bool) false, "drawControlPoints", "Debug : draw Control points with a color depending on its status "))
        ,degree(0)

    {
    }
template< class DataTypes>
 void BezierTriangleSetGeometryAlgorithms< DataTypes >::init()
{
	TriangleSetGeometryAlgorithms<DataTypes>::init();
	// recovers the pointer to a BezierTriangleSetTopologyContainer
	BezierTriangleSetTopologyContainer *btstc = NULL;
	this->getContext()->get(btstc, sofa::core::objectmodel::BaseContext::Local);
	if (!btstc) {
		serr << " Could not find a BezierTriangleSetTopologyContainer object"<< sendl;
	} else {
		container=btstc;
		/// get the degree of the Bezier Triangle
		degree=container->getDegree();
		/// store the Triangle bezier index for each Triangle
		tbiArray=container->getTriangleBezierIndexArray();
		/// compute the Bernstein coefficient for each control point in a Triangle
		bernsteinCoefficientArray.clear();
		bernsteinCoeffMap.clear();
		bernsteinCoefficientArray.resize(tbiArray.size());

		TriangleBezierIndex tbi;
		/// precompute the factorial of the degree.
		for (size_t i=0;i<tbiArray.size();++i) {
			tbi=tbiArray[i];
			bernsteinCoefficientArray[i]=multinomial(degree,tbi); 
            bernsteinCoeffMap.insert(std::pair<TriangleBezierIndex,Real>(tbi,(Real) bernsteinCoefficientArray[i]));
		}
		/// insert coefficient for the inferior degree
        BezierDegreeType i,j,k,/*l,*/m,n,index1,index2;
		for (i=0;i<=(degree-1);++i) {
			for (j=0;j<=(degree-i-1);++j) {
				k=degree-1-i-j;
				tbi=TriangleBezierIndex(i,j,k);
                bernsteinCoeffMap.insert(std::pair<TriangleBezierIndex,Real>(tbi,(Real) multinomial(degree-1,tbi)));
			}
		}
	
		/// fills the array of edges
		bezierTriangleEdgeSet.clear();
		TriangleBezierIndex tbiNext;

		for (i=0;i<=degree;++i) {
			for (j=0;j<=(degree-i);++j) {
				k=degree-i-j;
				tbi=TriangleBezierIndex(i,j,k);
				index1=container->getLocalIndexFromTriangleBezierIndex(tbi);
				for(m=0;m<3;++m) {
					if (tbi[m]<degree) {
						for (n=1;n<3;++n) {
							if (tbi[(m+n)%3]!=0) {
								tbiNext=tbi;
								tbiNext[m]=tbi[m]+1;
								tbiNext[(m+n)%3]=tbi[(m+n)%3]-1;
								index2=container->getLocalIndexFromTriangleBezierIndex(tbiNext);
								Edge e((PointID)std::min(index1,index2),(PointID)std::max(index1,index2));
								// test if both control points are on an edge
								if (tbi[(m+3-n)%3]==0)
									bezierTriangleEdgeSet.insert(std::pair<Edge,bool>(e,true));
								else 
									bezierTriangleEdgeSet.insert(std::pair<Edge,bool>(e,false));
							}
						}
					}
				}
			}
		}
	}



 }

template< class DataTypes>
 void BezierTriangleSetGeometryAlgorithms< DataTypes >::reinit()
{
}
template< class DataTypes>
typename DataTypes::Coord BezierTriangleSetGeometryAlgorithms< DataTypes >::computeNodalValue(const size_t triangleIndex,const Vec3 barycentricCoordinate, const typename DataTypes::VecCoord& p)
{
	Coord nodalValue;
	nodalValue.clear();
	VecPointID indexArray;
	TriangleBezierIndex tbi;
	bool isRational=container->isRationalSpline(triangleIndex);

	container->getGlobalIndexArrayOfBezierPointsInTriangle(triangleIndex, indexArray);
	if (isRational) {
		const BezierTriangleSetTopologyContainer::SeqWeights &wa=container->getWeightArray();
		Real weight=(Real)0.0f;
		Real bernsteinPolynonial;
		for(size_t i=0; i<tbiArray.size(); ++i)
		{
			tbi=tbiArray[i];
			bernsteinPolynonial=bernsteinCoefficientArray[i]*pow(barycentricCoordinate[0],tbi[0])*pow(barycentricCoordinate[1],tbi[1])*pow(barycentricCoordinate[2],tbi[2]);
			nodalValue+=wa[indexArray[i]]*bernsteinPolynonial*p[indexArray[i]];
			weight+=wa[indexArray[i]]*bernsteinPolynonial;
		}
		nodalValue/=weight;
	} else {
		for(size_t i=0; i<tbiArray.size(); ++i)
		{
			tbi=tbiArray[i];
			nodalValue+=p[indexArray[i]]*bernsteinCoefficientArray[i]*pow(barycentricCoordinate[0],tbi[0])*pow(barycentricCoordinate[1],tbi[1])*pow(barycentricCoordinate[2],tbi[2]);
		}
	}

	return(nodalValue);
}
template< class DataTypes>
typename DataTypes::Coord BezierTriangleSetGeometryAlgorithms< DataTypes >::computeNodalValue(const size_t triangleIndex,const Vec3 barycentricCoordinate)
{
	const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());
	return(computeNodalValue(triangleIndex,barycentricCoordinate,p));
}
template<class DataTypes>
typename DataTypes::Real BezierTriangleSetGeometryAlgorithms<DataTypes>::computeBernsteinPolynomial(const TriangleBezierIndex tbi, const Vec3 barycentricCoordinate)
{
	Real  val=pow(barycentricCoordinate[0],tbi[0])*pow(barycentricCoordinate[1],tbi[1])*pow(barycentricCoordinate[2],tbi[2]);
    typename std::map<TriangleBezierIndex,Real>::iterator it=bernsteinCoeffMap.find(tbi);
	if (it!=bernsteinCoeffMap.end()) {
		val*=(*it).second;
		return(val);
	} else {
		val*=multinomial(tbi[0]+tbi[1]+tbi[2],tbi);
		return(val);
	}
}
 template<class DataTypes>
 typename BezierTriangleSetGeometryAlgorithms<DataTypes>::Deriv 
	 BezierTriangleSetGeometryAlgorithms<DataTypes>::computeJacobian(const size_t triangleIndex, const Vec3 barycentricCoordinate, const typename DataTypes::VecCoord& p)
 {
	/// the 2 derivatives
	Deriv dpos[2];
	VecPointID indexArray;
	bool isRational=container->isRationalSpline(triangleIndex);
	TriangleBezierIndex tbi;
	size_t j;
	Real val;
	container->getGlobalIndexArrayOfBezierPointsInTriangle(triangleIndex, indexArray);
	if (isRational) {
		const BezierTriangleSetTopologyContainer::SeqWeights &wa=container->getWeightArray();
		Real weight=(Real)0.0f;
        Real dweight[2]={0,0};
		Coord pos;
		for(size_t i=0; i<tbiArray.size(); ++i)
		{
			tbi=tbiArray[i];
			val=bernsteinCoefficientArray[i]*pow(barycentricCoordinate[0],tbi[0])*pow(barycentricCoordinate[1],tbi[1])*pow(barycentricCoordinate[2],tbi[2]);
			// compute the numerator and denominator
			pos+=wa[indexArray[i]]*val*p[indexArray[i]];
			weight+=wa[indexArray[i]]*val;

			Vec3 dval(0,0,0);
			for (j=0;j<3;++j) {
				if(tbi[j] && barycentricCoordinate[j]){
					dval[j]=(Real)tbi[j]*val/barycentricCoordinate[j];
				} else if ((barycentricCoordinate[j]==0.0f)&&(tbi[j]==1)) {
					dval[j]=bernsteinCoefficientArray[i];
					for (size_t k=1;k<=2;++k)
						dval[j]*=(Real)(pow( barycentricCoordinate[(j+k)%3],tbi[(j+k)%3]));
				}
			}
			for (j=0;j<2;++j) {
				dpos[j]+=(dval[j]-dval[2])*wa[indexArray[i]]*p[indexArray[i]];
				dweight[j]+=(dval[j]-dval[2])*wa[indexArray[i]];
			}
		}
		// computes the derivatives of the ratio of the 2 polynomial terms
		for (j=0;j<2;++j) {
			dpos[j]=dpos[j]/weight-(dweight[j]/(weight*weight))*pos;
		}
	} else {
		for(size_t i=0; i<tbiArray.size(); ++i)
		{
			tbi=tbiArray[i];
			val=bernsteinCoefficientArray[i]*pow(barycentricCoordinate[0],tbi[0])*pow(barycentricCoordinate[1],tbi[1])*pow(barycentricCoordinate[2],tbi[2]);
			Vec3 dval(0,0,0);
			for (j=0;j<3;++j) {
				if(tbi[j] && barycentricCoordinate[j]){
					dval[j]=(Real)tbi[j]*val/barycentricCoordinate[j];
				} else if ((barycentricCoordinate[j]==0.0f)&&(tbi[j]==1)) {
					dval[j]=bernsteinCoefficientArray[i];
					for (size_t k=1;k<=2;++k)
						dval[j]*=(Real)(pow( barycentricCoordinate[(j+k)%3],tbi[(j+k)%3]));
				}
			}
			for (j=0;j<2;++j) {
				dpos[j]+=(dval[j]-dval[2])*p[indexArray[i]];
			}
		}
	}
	return(cross(dpos[0],dpos[1]));
 }
 template<class DataTypes>
 typename BezierTriangleSetGeometryAlgorithms<DataTypes>::Deriv 
	 BezierTriangleSetGeometryAlgorithms<DataTypes>::computeJacobian(const size_t triangleIndex, const Vec3 barycentricCoordinate)
 {
	 const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());
	 return(computeJacobian(triangleIndex,barycentricCoordinate,p));

 }
template<class DataTypes>
void BezierTriangleSetGeometryAlgorithms<DataTypes>::computeDeCasteljeauPoints(const size_t triangleIndex, const Vec3 barycentricCoordinate,  const VecCoord& p, Coord dpos[3])
{
	/// the 4 derivatives
	VecPointID indexArray;
	TriangleBezierIndex tbi;
	bool isRational=container->isRationalSpline(triangleIndex);
	size_t j;
	Real val;
	container->getGlobalIndexArrayOfBezierPointsInTriangle(triangleIndex, indexArray);
	// initialize dpos
	for (j=0;j<3;++j) 
		dpos[j]=Coord();
	if (isRational) {
		Real weight=(Real)0.0f;
		Real dweight[3];
		const BezierTriangleSetTopologyContainer::SeqWeights &wa=container->getWeightArray();
		Coord pos;
		dweight[0]=dweight[1]=0.0;
		for(size_t i=0; i<tbiArray.size(); ++i)
		{
			tbi=tbiArray[i];
			val=bernsteinCoefficientArray[i]*pow(barycentricCoordinate[0],tbi[0])*pow(barycentricCoordinate[1],tbi[1])*pow(barycentricCoordinate[2],tbi[2]);
			pos+=val*p[indexArray[i]];
			weight+=val*wa[indexArray[i]];
			Vec3 dval(0,0,0);
			for (j=0;j<3;++j) {
				if(tbi[j] && barycentricCoordinate[j]){
					dval[j]=(Real)tbi[j]*val/barycentricCoordinate[j];
				}else if ((barycentricCoordinate[j]==0.0f)&&(tbi[j]==1)) {
					dval[j]=bernsteinCoefficientArray[i];
					for (size_t k=1;k<=2;++k)
						dval[j]*=(Real)(pow( barycentricCoordinate[(j+k)%3],tbi[(j+k)%3]));
				}
				dpos[j]+=dval[j]*p[indexArray[i]];
				dweight[j]+=dval[j]*wa[indexArray[i]];
			}
		}
		for (j=0;j<3;++j) {
			dpos[j]=dpos[j]/weight-(dweight[j]/(weight*weight))*pos;
		}
	} else{
		for(size_t i=0; i<tbiArray.size(); ++i)
		{
			tbi=tbiArray[i];
			val=bernsteinCoefficientArray[i]*pow(barycentricCoordinate[0],tbi[0])*pow(barycentricCoordinate[1],tbi[1])*pow(barycentricCoordinate[2],tbi[2]);
			Vec3 dval(0,0,0);
			for (j=0;j<3;++j) {
				if(tbi[j] && barycentricCoordinate[j]){
					dval[j]=(Real)tbi[j]*val/barycentricCoordinate[j];
					
				} else if ((barycentricCoordinate[j]==0.0f)&&(tbi[j]==1)) {
					dval[j]=bernsteinCoefficientArray[i];
					for (size_t k=1;k<=2;++k)
						dval[j]*=(Real)(pow( barycentricCoordinate[(j+k)%3],tbi[(j+k)%3]));
				}
				dpos[j]+=dval[j]*p[indexArray[i]];
			}
		}
	}

	
}

template<class DataTypes>
bool BezierTriangleSetGeometryAlgorithms<DataTypes>::isBezierTriangleAffine(const size_t triangleIndex,const VecCoord& p, Real tolerance) const{
	// get the global indices of all points
	VecPointID indexArray;
	container->getGlobalIndexArrayOfBezierPointsInTriangle(triangleIndex, indexArray);
	bool affine=true;

	/// skip the first 4 control points corresponding to the 4 corners
	size_t index=0;
	Coord corner[3],pos,actualPos;
	// store the position of the 4 corners
	for (index=0;index<3;++index)
		corner[index]=p[indexArray[index]];
	do {
		// compute the position of the control point as if the Triangle was affine
		pos=corner[0]*tbiArray[index][0]+corner[1]*tbiArray[index][1]+corner[2]*tbiArray[index][2];
			
		pos/=degree;
		// measure the distance between the real position and the affine position
		actualPos=p[indexArray[index]];
		if ((actualPos-pos).norm2()>tolerance) {
			affine=false;
		}
		index++;
	} while ((affine) && (index<indexArray.size()));
	return (affine);
}

 template<class DataTypes>
 typename BezierTriangleSetGeometryAlgorithms<DataTypes>::Vec3 BezierTriangleSetGeometryAlgorithms<DataTypes>::computeBernsteinPolynomialGradient(const TriangleBezierIndex tbi, const Vec3 barycentricCoordinate)
 {
     Real  val=computeBernsteinPolynomial(tbi,barycentricCoordinate);
     Vec3 dval(0,0,0);
     for(unsigned i=0;i<3;++i)
         if(tbi[i] && barycentricCoordinate[i])
             dval[i]=(Real)tbi[i]*val/barycentricCoordinate[i];
     return dval;
 }

 template<class DataTypes>
 typename BezierTriangleSetGeometryAlgorithms<DataTypes>::Mat33 BezierTriangleSetGeometryAlgorithms<DataTypes>::computeBernsteinPolynomialHessian(const TriangleBezierIndex tbi, const Vec3 barycentricCoordinate)
 {
     Vec3 dval = computeBernsteinPolynomialGradient(tbi,barycentricCoordinate);
     Mat33 ddval;
     for(unsigned i=0;i<3;++i)
         if(barycentricCoordinate[i])
             for(unsigned j=0;j<3;++j)
             {
                 if(i==j) { if(tbi[i]>1) ddval[j][i]=((Real)tbi[i]-1.)*dval[j]/barycentricCoordinate[i]; }
                 else { if(tbi[i]) ddval[j][i]=(Real)tbi[i]*dval[j]/barycentricCoordinate[i]; }
             }
     return ddval;
 }

template<class DataTypes>
void BezierTriangleSetGeometryAlgorithms<DataTypes>::draw(const core::visual::VisualParams* vparams)
{

	
	if ((degree>0) && (container) ){
		TriangleSetGeometryAlgorithms<DataTypes>::draw(vparams);	
		if (drawControlPoints.getValue())
		{
			size_t nbPoints=container->getNbPoints();
			size_t i,elementIndex,elementOffset;
			const typename DataTypes::VecCoord& pos =(this->object->read(core::ConstVecCoordId::position())->getValue());
			BezierTriangleSetTopologyContainer::BezierTrianglePointLocation location;

			if (container->getNbTriangles()>0) {
				// estimate the  mean radius of the spheres from the first Bezier triangle 
				VecPointID indexArray;
				container->getGlobalIndexArrayOfBezierPointsInTriangle(0, indexArray);
				std::vector<Real> edgeLengthArray;
				// compute median of the edge distance between control points	
                std::set<std::pair<Edge,bool> >::iterator ite=bezierTriangleEdgeSet.begin();
//				Real val=0;
				Coord pp;
				for (; ite!=bezierTriangleEdgeSet.end(); ite++)
				{
						pp = pos[indexArray[(*ite).first[0]]] -pos[indexArray[(*ite).first[1]]] ;
						edgeLengthArray.push_back(pp.norm());
				}
				std::nth_element(edgeLengthArray.begin(), edgeLengthArray.begin() + edgeLengthArray.size()/2, edgeLengthArray.end());
				Real radius=edgeLengthArray[edgeLengthArray.size()/2]/5;
				std::vector<sofa::defaulttype::Vector3> pointsVertices,pointsEdges,pointsTriangles;
				std::vector<float> radiusVertices,radiusEdges,radiusTriangles;
				sofa::defaulttype::Vector3 p;


				for (i=0;i<nbPoints;++i) {
					container->getLocationFromGlobalIndex(i,location,elementIndex,elementOffset);
					if (location==BezierTriangleSetTopologyContainer::NONE) {
					} else if (location==BezierTriangleSetTopologyContainer::POINT) {
						p=pos[i];
						pointsVertices.push_back(p);

						radiusVertices.push_back(radius*container->getWeight(i));

					} else if (location==BezierTriangleSetTopologyContainer::EDGE) {
						p=pos[i];
						pointsEdges.push_back(p);

						radiusEdges.push_back(radius*container->getWeight(i));

					} else {
						p=pos[i];
						pointsTriangles.push_back(p);

						radiusTriangles.push_back(radius*container->getWeight(i));

					}
				}
				vparams->drawTool()->setLightingEnabled(true); //Enable lightning
				vparams->drawTool()->drawSpheres(pointsVertices, radiusVertices,  defaulttype::Vec<4,float>(1.0f,0,0,1.0f));
				vparams->drawTool()->drawSpheres(pointsEdges, radiusEdges,  defaulttype::Vec<4,float>(0,1.0f,0,1.0f));
				vparams->drawTool()->drawSpheres(pointsTriangles, radiusTriangles,  defaulttype::Vec<4,float>(0,0,1.0f,1.0f));
				vparams->drawTool()->setLightingEnabled(false); //Disable lightning
			}
		}
		// Draw edges linking Bezier Triangle control points with a color code
		if (drawSmoothEdges.getValue())
		{
			
			const sofa::helper::vector<Triangle> &trianArray = this->m_topology->getTriangles();

			if (!trianArray.empty())
			{
				// estimate the  mean radius of the spheres from the first Bezier triangle 
				VecPointID indexArray;
				size_t nbPoints=container->getNbPoints();
				size_t i,elementIndex,elementOffset;
				const typename DataTypes::VecCoord& coords =(this->object->read(core::ConstVecCoordId::position())->getValue());
				BezierTriangleSetTopologyContainer::BezierTrianglePointLocation location;
				container->getGlobalIndexArrayOfBezierPointsInTriangle(0, indexArray);
				std::vector<Real> edgeLengthArray;
				// compute median of the edge distance between control points	
                std::set<std::pair<Edge,bool> >::iterator ite=bezierTriangleEdgeSet.begin();
//				Real val=0;
				Coord pp;
				for (; ite!=bezierTriangleEdgeSet.end(); ite++)
				{
					pp = coords[indexArray[(*ite).first[0]]] -coords[indexArray[(*ite).first[1]]] ;
					edgeLengthArray.push_back(pp.norm());
				}
				std::nth_element(edgeLengthArray.begin(), edgeLengthArray.begin() + edgeLengthArray.size()/2, edgeLengthArray.end());
				Real radius=edgeLengthArray[edgeLengthArray.size()/2]/5;
				std::vector<sofa::defaulttype::Vector3> pointsVertices;
				std::vector<float> radiusVertices;
				sofa::defaulttype::Vector3 p1;


				for (i=0;i<nbPoints;++i) {
					container->getLocationFromGlobalIndex(i,location,elementIndex,elementOffset);
					if (location==BezierTriangleSetTopologyContainer::POINT) {
						p1=coords[i];
						pointsVertices.push_back(p1);

						radiusVertices.push_back(radius*container->getWeight(i));

					} 
				}
				vparams->drawTool()->setLightingEnabled(true); //Enable lightning
				vparams->drawTool()->drawSpheres(pointsVertices, radiusVertices,  defaulttype::Vec<4,float>(1.0f,0,0,1.0f));
				vparams->drawTool()->setLightingEnabled(false); //Disable lightning

#ifndef SOFA_NO_OPENGL
				glDisable(GL_LIGHTING);
				
				glColor3f(0.0f, 1.0f, 0.0f);
				glLineWidth(3.0);
				 glEnable(GL_DEPTH_TEST);
				glEnable(GL_POLYGON_OFFSET_LINE);
				glPolygonOffset(-1.0,100.0);
				
				
				// how many points is used to discretize the edge
				const size_t edgeTesselation=9;
				sofa::defaulttype::Vec3f p; //,p2;
				for ( i = 0; i<trianArray.size(); i++)
				{
					indexArray.clear();
					container->getGlobalIndexArrayOfBezierPointsInTriangle(i, indexArray);
					sofa::helper::vector <sofa::defaulttype::Vec3f> trianCoord;
					// process each edge of the triangle
					for (size_t j = 0; j<3; j++) {
						Vec3 baryCoord;
						baryCoord[j]=0;
						glBegin(GL_LINE_STRIP);
						for (size_t k=0;k<=edgeTesselation;++k) {
							baryCoord[(j+1)%3]=(Real)k/(Real)edgeTesselation;
							baryCoord[(j+2)%3]=(Real)(edgeTesselation-k)/(Real)edgeTesselation;
							p=DataTypes::getCPos(computeNodalValue(i,baryCoord));
							glVertex3f(p[0],p[1],p[2]);
						}
						glEnd();
					}
				}
				glDisable(GL_POLYGON_OFFSET_LINE);
				
			}
#endif // SOFA_NO_OPENGL
		}
		
		// Draw edges linking Bezier Triangle control points with a color code
		if (drawControlPointsEdges.getValue())
		{
#ifndef SOFA_NO_OPENGL
			const sofa::helper::vector<Triangle> &trianArray = this->m_topology->getTriangles();

			if (!trianArray.empty())
			{
				glDisable(GL_LIGHTING);
				const sofa::defaulttype::Vec4f& color =  this->_drawColor.getValue();
				glColor3f(color[0], color[1], color[2]);
				glBegin(GL_LINES);
				const VecCoord& coords =(this->object->read(core::ConstVecCoordId::position())->getValue());
				VecPointID indexArray;
				Vec3 baryCoord;
				sofa::defaulttype::Vec3f p; //,p2;
				for (unsigned int i = 0; i<trianArray.size(); i++)
				{
					indexArray.clear();
					container->getGlobalIndexArrayOfBezierPointsInTriangle(i, indexArray);
					sofa::helper::vector <sofa::defaulttype::Vec3f> trianCoord;

					for (unsigned int j = 0; j<indexArray.size(); j++)
					{
						p = DataTypes::getCPos(coords[indexArray[j]]);
						trianCoord.push_back(p);
					}
					
                    std::set<std::pair<Edge,bool> >::iterator ite=bezierTriangleEdgeSet.begin();
					for (; ite!=bezierTriangleEdgeSet.end(); ite++)
					{
						if ((*ite).second) {
							glColor3f(0.0f, 1.0f, 0.0f);
						} else {
							glColor3f(0.0f, 0.0f, 1.0f );
						}
						glVertex3f(trianCoord[(*ite).first[0]][0], trianCoord[(*ite).first[0]][1], trianCoord[(*ite).first[0]][2]);
						glVertex3f(trianCoord[(*ite).first[1]][0], trianCoord[(*ite).first[1]][1], trianCoord[(*ite).first[1]][2]);
					}
				}
				glEnd();
#endif // SOFA_NO_OPENGL
			}

		}
	}
	

}



} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_TRIANGLESETGEOMETRYALGORITHMS_INL
