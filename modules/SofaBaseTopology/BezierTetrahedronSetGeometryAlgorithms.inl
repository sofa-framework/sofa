/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_BEZIERTETRAHEDRONSETGEOMETRYALGORITHMS_INL
#define SOFA_COMPONENT_TOPOLOGY_BEZIERTETRAHEDRONSETGEOMETRYALGORITHMS_INL

#include <SofaBaseTopology/BezierTetrahedronSetGeometryAlgorithms.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseTopology/CommonAlgorithms.h>
#include <sofa/helper/rmath.h>
namespace sofa
{

namespace component
{

namespace topology
{

double multinomial(const size_t n,const TetrahedronBezierIndex tbiIn)
 {
	size_t i,ival;
	TetrahedronBezierIndex tbi=tbiIn;
	// divide n! with the largest of the multinomial coefficient
	std::sort(tbi.begin(),tbi.end());
	ival=1;
	for (i=n;i>tbi[3];--i){
		ival*=i;
	}
    return(((double)ival)/(sofa::helper::factorial(tbi[0])*sofa::helper::factorial(tbi[1])*sofa::helper::factorial(tbi[2])));
 }
template< class DataTypes>
 BezierTetrahedronSetGeometryAlgorithms< DataTypes >::BezierTetrahedronSetGeometryAlgorithms() : 
TetrahedronSetGeometryAlgorithms<DataTypes>()
		,degree(0)
        ,drawControlPointsEdges (core::objectmodel::Base::initData(&drawControlPointsEdges, (bool) false, "drawControlPointsEdges", "Debug : draw Control point edges "))
        ,drawSmoothEdges (core::objectmodel::Base::initData(&drawSmoothEdges, (bool) false, "drawSmoothEdges", "Debug : draw Bezier curves as edges of the  Bezier triangle"))
 	   ,drawControlPoints (core::objectmodel::Base::initData(&drawControlPoints, (bool) false, "drawControlPoints", "Debug : draw Control points with a color depending on its status "))
	   ,d_referenceRadius (core::objectmodel::Base::initData(&d_referenceRadius, (Real) -1.0f, "referenceRadius", "Debug : radius of control points when drawing "))
    {
    }
template< class DataTypes>
 void BezierTetrahedronSetGeometryAlgorithms< DataTypes >::init()
{
	TetrahedronSetGeometryAlgorithms<DataTypes>::init();
	// recovers the pointer to a BezierTetrahedronSetTopologyContainer
	BezierTetrahedronSetTopologyContainer *btstc = NULL;
	this->getContext()->get(btstc, sofa::core::objectmodel::BaseContext::Local);
	if (!btstc) {
		serr << " Could not find a BezierTetrahedronSetTopologyContainer object"<< sendl;
	} else {
		container=btstc;
		/// get the degree of the Bezier tetrahedron
		degree=container->getDegree();
		/// store the tetrahedron bezier index for each tetrahedron
		tbiArray=container->getTetrahedronBezierIndexArray();
		/// compute the Bernstein coefficient for each control point in a tetrahedron
		bernsteinCoefficientArray.clear();
		bernsteinCoeffMap.clear();
		bernsteinCoefficientArray.resize(tbiArray.size());

		TetrahedronBezierIndex tbi;
		/// precompute the factorial of the degree.
		for (size_t i=0;i<tbiArray.size();++i) {
			tbi=tbiArray[i];
			bernsteinCoefficientArray[i]=multinomial(degree,tbi); 
			bernsteinCoeffMap.insert(std::pair<TetrahedronBezierIndex,Real>(tbi,(Real)bernsteinCoefficientArray[i]));
		}
		/// insert coefficient for the inferior degree
		BezierDegreeType i,j,k,l,m,n,index1,index2;
		for (i=0;i<=(degree-1);++i) {
			for (j=0;j<=(degree-i-1);++j) {
				for (k=0;k<=(degree-j-i-1);++k) {
					l=degree-1-i-j-k;
					tbi=TetrahedronBezierIndex(i,j,k,l);
					bernsteinCoeffMap.insert(std::pair<TetrahedronBezierIndex,Real>(tbi,(Real)multinomial(degree-1,tbi)));
				}
			}
		}
		/// fills the array of edges
		bezierTetrahedronEdgeSet.clear();
		TetrahedronBezierIndex tbiNext;
	
		for (i=0;i<=degree;++i) {
			for (j=0;j<=(degree-i);++j) {
				for (k=0;k<=(degree-j-i);++k) {
					l=degree-i-j-k;
					tbi=TetrahedronBezierIndex(i,j,k,l);
					index1=container->getLocalIndexFromTetrahedronBezierIndex(tbi);
					for(m=0;m<4;++m) {
						if (tbi[m]<degree) {
							for (n=1;n<4;++n) {
								if (tbi[(m+n)%4]!=0) {
									tbiNext=tbi;
									tbiNext[m]=tbi[m]+1;
									tbiNext[(m+n)%4]=tbi[(m+n)%4]-1;
									index2=container->getLocalIndexFromTetrahedronBezierIndex(tbiNext);
									Edge e((PointID)std::min(index1,index2),(PointID)std::max(index1,index2));
									// test if both control points are on an edge or an
									if (tbi[(m+1+(n%3))%4]==0) {
										if (tbi[(m+1+((n+1)%3))%4]==0) {
											// edge connects points along an edge
											bezierTetrahedronEdgeSet.insert(std::pair<Edge,size_t>(e,(size_t)2));
										} else 
											// edge connects points along a triangle
											bezierTetrahedronEdgeSet.insert(std::pair<Edge,size_t>(e,(size_t)1));
									} else if  (tbi[(m+1+((n+1)%3))%4]==0) {
										bezierTetrahedronEdgeSet.insert(std::pair<Edge,size_t>(e,(size_t)1));
									} else
										bezierTetrahedronEdgeSet.insert(std::pair<Edge,size_t>(e,(size_t)0));
								}
							}
						}
					}
				}
			}
		}
	}


}

template< class DataTypes>
 void BezierTetrahedronSetGeometryAlgorithms< DataTypes >::reinit()
{
}
template< class DataTypes>
typename DataTypes::Coord BezierTetrahedronSetGeometryAlgorithms< DataTypes >::computeNodalValue(const size_t tetrahedronIndex,const Vec4 barycentricCoordinate, const typename DataTypes::VecCoord& p)
{
	Coord nodalValue;
	nodalValue.clear();

	TetrahedronBezierIndex tbi;
	const VecPointID &indexArray=container->getGlobalIndexArrayOfBezierPoints(tetrahedronIndex);
	bool isRational=container->isRationalSpline(tetrahedronIndex);
	if (isRational) {
		const BezierTetrahedronSetTopologyContainer::SeqWeights &wa=container->getWeightArray();
		Real weight=(Real)0.0f;
		Real bernsteinPolynonial;
		for(size_t i=0; i<tbiArray.size(); ++i)
		{
			tbi=tbiArray[i];
			bernsteinPolynonial=bernsteinCoefficientArray[i]*pow(barycentricCoordinate[0],tbi[0])*pow(barycentricCoordinate[1],tbi[1])*pow(barycentricCoordinate[2],tbi[2])*pow(barycentricCoordinate[3],tbi[3]);
			nodalValue+=wa[indexArray[i]]*p[indexArray[i]]*bernsteinPolynonial;
			weight+=wa[indexArray[i]]*bernsteinPolynonial;
		}
		nodalValue/=weight;
	} else {
		for(size_t i=0; i<tbiArray.size(); ++i)
		{
			tbi=tbiArray[i];
			nodalValue+=p[indexArray[i]]*bernsteinCoefficientArray[i]*pow(barycentricCoordinate[0],tbi[0])*pow(barycentricCoordinate[1],tbi[1])*pow(barycentricCoordinate[2],tbi[2])*pow(barycentricCoordinate[3],tbi[3]);
		}
	}

	return(nodalValue);
}
template< class DataTypes>
typename DataTypes::Coord BezierTetrahedronSetGeometryAlgorithms< DataTypes >::computeNodalValue(const size_t tetrahedronIndex,const Vec4 barycentricCoordinate)
{
	const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());
	return(computeNodalValue(tetrahedronIndex,barycentricCoordinate,p));
}
template<class DataTypes>
typename DataTypes::Real BezierTetrahedronSetGeometryAlgorithms<DataTypes>::computeBernsteinPolynomial(const TetrahedronBezierIndex tbi, const Vec4 barycentricCoordinate)
{
	Real  val=pow(barycentricCoordinate[0],tbi[0])*pow(barycentricCoordinate[1],tbi[1])*pow(barycentricCoordinate[2],tbi[2])*pow(barycentricCoordinate[3],tbi[3]);
    typename std::map<TetrahedronBezierIndex,Real>::iterator it=bernsteinCoeffMap.find(tbi);
	if (it!=bernsteinCoeffMap.end()) {
		val*=(*it).second;
		return(val);
	} else {
		val*=multinomial(tbi[0]+tbi[1]+tbi[2]+tbi[3],tbi);
		return(val);
	}
}
 template<class DataTypes>
 typename BezierTetrahedronSetGeometryAlgorithms<DataTypes>::Real 
	 BezierTetrahedronSetGeometryAlgorithms<DataTypes>::computeJacobian(const size_t tetrahedronIndex, const Vec4 barycentricCoordinate, const typename DataTypes::VecCoord& p)
 {
	/// the 3 derivatives
	Coord dpos[3];
	
	TetrahedronBezierIndex tbi;
	size_t j;
	Real val;
	const VecPointID &indexArray=container->getGlobalIndexArrayOfBezierPoints(tetrahedronIndex);
	bool isRational=container->isRationalSpline(tetrahedronIndex);
	if (isRational) {
		const BezierTetrahedronSetTopologyContainer::SeqWeights &wa=container->getWeightArray();
		Real weight=(Real)0.0f;
        Real dweight[3]= {0,0,0};
		dweight[0]=dweight[1]=dweight[2]=(Real)0.0f;
		Coord pos;
		for(size_t i=0; i<tbiArray.size(); ++i)
		{
			tbi=tbiArray[i];
			val=bernsteinCoefficientArray[i]*pow(barycentricCoordinate[0],tbi[0])*pow(barycentricCoordinate[1],tbi[1])*pow(barycentricCoordinate[2],tbi[2])*pow(barycentricCoordinate[3],tbi[3]);
			Vec4 dval(0,0,0,0);
			pos+=wa[indexArray[i]]*val*p[indexArray[i]];
			weight+=wa[indexArray[i]]*val;
			for (j=0;j<4;++j) {
				if(tbi[j] && barycentricCoordinate[j]){
					dval[j]=(Real)tbi[j]*val/barycentricCoordinate[j];
				} else if ((barycentricCoordinate[j]==0.0f)&&(tbi[j]==1)) {
					dval[j]=bernsteinCoefficientArray[i];
					for (size_t k=1;k<=3;++k)
						dval[j]*=(Real)(pow( barycentricCoordinate[(j+k)%4],tbi[(j+k)%4]));
				}
			}
			for (j=0;j<3;++j) {
				dpos[j]+=(dval[j]-dval[3])*wa[indexArray[i]]*p[indexArray[i]];
				dweight[j]+=(dval[j]-dval[3])*wa[indexArray[i]];
			}
		}
		// computes the derivatives of the ratio of the 2 polynomial terms
		for (j=0;j<3;++j) {
			dpos[j]=dpos[j]/weight-(dweight[j]/(weight*weight))*pos;
		}
	}else {
		for(size_t i=0; i<tbiArray.size(); ++i)
		{
			tbi=tbiArray[i];
			val=bernsteinCoefficientArray[i]*pow(barycentricCoordinate[0],tbi[0])*pow(barycentricCoordinate[1],tbi[1])*pow(barycentricCoordinate[2],tbi[2])*pow(barycentricCoordinate[3],tbi[3]);
			Vec4 dval(0,0,0,0);
			for (j=0;j<4;++j) {
				if(tbi[j] && barycentricCoordinate[j]){
					dval[j]=(Real)tbi[j]*val/barycentricCoordinate[j];
				} else if ((barycentricCoordinate[j]==0.0f)&&(tbi[j]==1)) {
					dval[j]=bernsteinCoefficientArray[i];
					for (size_t k=1;k<=3;++k)
						dval[j]*=(Real)(pow( barycentricCoordinate[(j+k)%4],tbi[(j+k)%4]));
				}
			}
			for (j=0;j<3;++j) {
				dpos[j]+=(dval[j]-dval[3])*p[indexArray[i]];
			}
		}
	}

	
	return(tripleProduct(dpos[0],dpos[1],dpos[2]));
 }
 template<class DataTypes>
 typename BezierTetrahedronSetGeometryAlgorithms<DataTypes>::Real 
	 BezierTetrahedronSetGeometryAlgorithms<DataTypes>::computeJacobian(const size_t tetrahedronIndex, const Vec4 barycentricCoordinate)
 {
	 const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());
	 return(computeJacobian(tetrahedronIndex,barycentricCoordinate,p));

 }
template<class DataTypes>
void BezierTetrahedronSetGeometryAlgorithms<DataTypes>::computeDeCasteljeauPoints(const size_t tetrahedronIndex, const Vec4 barycentricCoordinate,  const VecCoord& p, Coord dpos[4])
{
	/// the 4 derivatives
	
	TetrahedronBezierIndex tbi;
	size_t j;
	Real val;
	const VecPointID &indexArray=container->getGlobalIndexArrayOfBezierPoints(tetrahedronIndex);
	// initialize dpos
	for (j=0;j<4;++j) 
		dpos[j]=Coord();
	bool isRational=container->isRationalSpline(tetrahedronIndex);
	if (isRational) {
		const BezierTetrahedronSetTopologyContainer::SeqWeights &wa=container->getWeightArray();
		Real weight=(Real)0.0f;
		Real dweight[4];
		dweight[0]=0.0f;dweight[1]=0.0f;dweight[2]=0.0f;dweight[3]=0.0f;

		Coord pos;
		for(size_t i=0; i<tbiArray.size(); ++i)
		{
			tbi=tbiArray[i];
			val=bernsteinCoefficientArray[i]*pow(barycentricCoordinate[0],tbi[0])*pow(barycentricCoordinate[1],tbi[1])*pow(barycentricCoordinate[2],tbi[2])*pow(barycentricCoordinate[3],tbi[3]);
			pos+=val*wa[indexArray[i]]*p[indexArray[i]];
			weight+=val*wa[indexArray[i]];
			Vec4 dval(0,0,0,0);
			for (j=0;j<4;++j) {
				if(tbi[j] && barycentricCoordinate[j]){
					dval[j]=(Real)tbi[j]*val/barycentricCoordinate[j];

				} else if ((barycentricCoordinate[j]==0.0f)&&(tbi[j]==1)) {
					dval[j]=bernsteinCoefficientArray[i];
					for (size_t k=1;k<=3;++k)
						dval[j]*=(Real)(pow( barycentricCoordinate[(j+k)%4],tbi[(j+k)%4]));
				}
				dpos[j]+=dval[j]*wa[indexArray[i]]*p[indexArray[i]];
				dweight[j]+=dval[j]*wa[indexArray[i]];
			}
		}
		for (j=0;j<4;++j) {
			dpos[j]=dpos[j]/weight-(dweight[j]/(weight*weight))*pos;
		}
		
		
	} else {
		for(size_t i=0; i<tbiArray.size(); ++i)
		{
			tbi=tbiArray[i];
			val=bernsteinCoefficientArray[i]*pow(barycentricCoordinate[0],tbi[0])*pow(barycentricCoordinate[1],tbi[1])*pow(barycentricCoordinate[2],tbi[2])*pow(barycentricCoordinate[3],tbi[3]);
			Vec4 dval(0,0,0,0);
			for (j=0;j<4;++j) {
				if(tbi[j] && barycentricCoordinate[j]){
					dval[j]=(Real)tbi[j]*val/barycentricCoordinate[j];

				} else if ((barycentricCoordinate[j]==0.0f)&&(tbi[j]==1)) {
					dval[j]=bernsteinCoefficientArray[i];
					for (size_t k=1;k<=3;++k)
						dval[j]*=(Real)(pow( barycentricCoordinate[(j+k)%4],tbi[(j+k)%4]));
				}
				dpos[j]+=dval[j]*p[indexArray[i]];
			}
		}

	}


}

template<class DataTypes>
bool BezierTetrahedronSetGeometryAlgorithms<DataTypes>::isBezierTetrahedronAffine(const size_t tetrahedronIndex,const VecCoord& p, Real tolerance) const{
	// get the global indices of all points
	
	const VecPointID &indexArray=container->getGlobalIndexArrayOfBezierPoints(tetrahedronIndex);
	bool affine=true;

	/// skip the first 4 control points corresponding to the 4 corners
	size_t index=0;
	Coord corner[4],pos,actualPos;
	// store the position of the 4 corners
	for (index=0;index<4;++index)
		corner[index]=p[indexArray[index]];
	do {
		// compute the position of the control point as if the tetrahedron was affine
		pos=corner[0]*tbiArray[index][0]+corner[1]*tbiArray[index][1]+corner[2]*tbiArray[index][2]+
			corner[3]*tbiArray[index][3];
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
 typename BezierTetrahedronSetGeometryAlgorithms<DataTypes>::Vec4 BezierTetrahedronSetGeometryAlgorithms<DataTypes>::computeBernsteinPolynomialGradient(const TetrahedronBezierIndex tbi, const Vec4 barycentricCoordinate)
 {
     Real  val=computeBernsteinPolynomial(tbi,barycentricCoordinate);
     Vec4 dval(0,0,0,0);
     for(unsigned i=0;i<4;++i)
         if(tbi[i] && barycentricCoordinate[i])
             dval[i]=(Real)tbi[i]*val/barycentricCoordinate[i];
     return dval;
 }

 template<class DataTypes>
 typename BezierTetrahedronSetGeometryAlgorithms<DataTypes>::Mat44 BezierTetrahedronSetGeometryAlgorithms<DataTypes>::computeBernsteinPolynomialHessian(const TetrahedronBezierIndex tbi, const Vec4 barycentricCoordinate)
 {
     Vec4 dval = computeBernsteinPolynomialGradient(tbi,barycentricCoordinate);
     Mat44 ddval;
     for(unsigned i=0;i<4;++i)
         if(barycentricCoordinate[i])
             for(unsigned j=0;j<4;++j)
             {
                 if(i==j) { if(tbi[i]>1) ddval[j][i]=((Real)tbi[i]-1.)*dval[j]/barycentricCoordinate[i]; }
                 else { if(tbi[i]) ddval[j][i]=(Real)tbi[i]*dval[j]/barycentricCoordinate[i]; }
             }
     return ddval;
 }

template<class DataTypes>
void BezierTetrahedronSetGeometryAlgorithms<DataTypes>::draw(const core::visual::VisualParams* vparams)
{

    if ((degree>0) && (container) )
    {
        TetrahedronSetGeometryAlgorithms<DataTypes>::draw(vparams);
        // Draw Tetra
        // reference radius
        if (d_referenceRadius.getValue()<0.0) {
            // estimate the  mean radius of the spheres from the first Bezier triangle

//			size_t nbPoints=container->getNbPoints();
//			size_t i,elementIndex,elementOffset;
            const typename DataTypes::VecCoord& coords =(this->object->read(core::ConstVecCoordId::position())->getValue());
//			BezierTetrahedronSetTopologyContainer::BezierTetrahedronPointLocation location;
            const VecPointID &indexArray=container->getGlobalIndexArrayOfBezierPoints(0);
            std::vector<Real> edgeLengthArray;
            // compute median of the edge distance between control points
            sofa::helper::set<std::pair<Edge,size_t> >::iterator ite=bezierTetrahedronEdgeSet.begin();
            //Real val=0;
            Coord pp;
            for (; ite!=bezierTetrahedronEdgeSet.end(); ite++)
            {
                pp = coords[indexArray[(*ite).first[0]]] -coords[indexArray[(*ite).first[1]]] ;
                edgeLengthArray.push_back(pp.norm());
            }
            std::nth_element(edgeLengthArray.begin(), edgeLengthArray.begin() + edgeLengthArray.size()/2, edgeLengthArray.end());
            Real radius=edgeLengthArray[edgeLengthArray.size()/2]/5;
            d_referenceRadius.setValue(radius);
        }

        if (drawControlPoints.getValue())
        {
            size_t nbPoints=container->getNbPoints();
            size_t i,elementIndex,elementOffset;
            const typename DataTypes::VecCoord& pos =(this->object->read(core::ConstVecCoordId::position())->getValue());
            BezierTetrahedronSetTopologyContainer::BezierTetrahedronPointLocation location;

            if (container->getNbTriangles()>0) {
                // estimate the  mean radius of the spheres from the first Bezier triangle
                VecPointID indexArray;
                float radius=	d_referenceRadius.getValue();
                std::vector<sofa::defaulttype::Vector3> pointsVertices,pointsEdges,pointsTriangles,pointsTetrahedra;
                std::vector<float> radiusVertices,radiusEdges,radiusTriangles,radiusTetrahedra;
                sofa::defaulttype::Vector3 p;


                for (i=0;i<nbPoints;++i) {
                    container->getLocationFromGlobalIndex(i,location,elementIndex,elementOffset);
                    if (location==BezierTetrahedronSetTopologyContainer::POINT) {
                        p=pos[i];
                        pointsVertices.push_back(p);

                        radiusVertices.push_back(radius*container->getWeight(i));

                    } else if (location==BezierTetrahedronSetTopologyContainer::EDGE) {
                        p=pos[i];
                        pointsEdges.push_back(p);

                        radiusEdges.push_back(radius*container->getWeight(i));

                    } else if (location==BezierTetrahedronSetTopologyContainer::TRIANGLE) {
                        p=pos[i];
                        pointsTriangles.push_back(p);

                        radiusTriangles.push_back(radius*container->getWeight(i));

                    } else {
                        p=pos[i];
                        pointsTetrahedra.push_back(p);

                        radiusTetrahedra.push_back(radius*container->getWeight(i));
                    }
                }
                vparams->drawTool()->setLightingEnabled(true); //Enable lightning
                vparams->drawTool()->drawSpheres(pointsVertices, radiusVertices,  defaulttype::Vec<4,float>(1.0f,0,0,1.0f));
                vparams->drawTool()->drawSpheres(pointsEdges, radiusEdges,  defaulttype::Vec<4,float>(0,1.0f,0,1.0f));
                vparams->drawTool()->drawSpheres(pointsTriangles, radiusTriangles,  defaulttype::Vec<4,float>(0,0,1.0f,1.0f));
                vparams->drawTool()->drawSpheres(pointsTetrahedra, radiusTetrahedra,  defaulttype::Vec<4,float>(1,0,1.0f,1.0f));
                vparams->drawTool()->setLightingEnabled(false); //Disable lightning
            }
        }
        // Draw edges linking Bezier tetrahedra control points with a color code
        if (drawSmoothEdges.getValue())
        {

            const sofa::helper::vector<Tetrahedron> &tetraArray = this->m_topology->getTetrahedra();

            if (!tetraArray.empty())
            {
                float radius=	d_referenceRadius.getValue();
                const typename DataTypes::VecCoord& coords =(this->object->read(core::ConstVecCoordId::position())->getValue());
                std::vector<sofa::defaulttype::Vector3> pointsVertices;
                std::vector<float> radiusVertices;
                sofa::defaulttype::Vector3 p1;
                size_t elementIndex,i,elementOffset;
                BezierTetrahedronSetTopologyContainer::BezierTetrahedronPointLocation location;
                size_t nbPoints=container->getNbPoints();

                for (i=0;i<nbPoints;++i) {
                    container->getLocationFromGlobalIndex(i,location,elementIndex,elementOffset);
                    if (location==BezierTetrahedronSetTopologyContainer::POINT) {
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

                const unsigned int edgesInTetrahedronArray[6][2] = {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};
                const unsigned int oppositeEdgesInTetrahedronArray[6][2] = {{2,3}, {1,3}, {1,2}, {0,3}, {0,2}, {0,1}};
                // how many points is used to discretize the edge
                const size_t edgeTesselation=9;
                sofa::defaulttype::Vec3f p; //,p2;
                for ( i = 0; i<tetraArray.size(); i++)
                {

//					const VecPointID &indexArray=container->getGlobalIndexArrayOfBezierPoints(i);
//					sofa::helper::vector <sofa::defaulttype::Vec3f> trianCoord;
                    // process each edge of the tetrahedron
                    for (size_t j = 0; j<6; j++) {
                        Vec4 baryCoord;
                        baryCoord[oppositeEdgesInTetrahedronArray[j][0]]=0;
                        baryCoord[oppositeEdgesInTetrahedronArray[j][1]]=0;
                        glBegin(GL_LINE_STRIP);
                        for (size_t k=0;k<=edgeTesselation;++k) {
                            baryCoord[edgesInTetrahedronArray[j][0]]=(Real)k/(Real)edgeTesselation;
                            baryCoord[edgesInTetrahedronArray[j][1]]=(Real)(edgeTesselation-k)/(Real)edgeTesselation;
                            p=DataTypes::getCPos(computeNodalValue(i,baryCoord));
                            glVertex3f(p[0],p[1],p[2]);
                        }
                        glEnd();
                    }
                }
                glDisable(GL_POLYGON_OFFSET_LINE);
#endif // SOFA_NO_OPENGL
            }

        }

        if (drawControlPointsEdges.getValue())
        {
#ifndef SOFA_NO_OPENGL
            const sofa::helper::vector<Tetrahedron> &tetraArray = this->m_topology->getTetrahedra();
            if (!tetraArray.empty())
            {
                glDisable(GL_LIGHTING);
                const sofa::defaulttype::Vec3f& color =  this->_drawColor.getValue();
                glColor3f(color[0], color[1], color[2]);
                glBegin(GL_LINES);
                const VecCoord& coords =(this->object->read(core::ConstVecCoordId::position())->getValue());

                Vec4 baryCoord;
                sofa::defaulttype::Vec3f p; //,p2;
                for (unsigned int i = 0; i<tetraArray.size(); i++)
                {

                    const VecPointID &indexArray=container->getGlobalIndexArrayOfBezierPoints(i);
                    sofa::helper::vector <sofa::defaulttype::Vec3f> tetraCoord;

                    for (unsigned int j = 0; j<indexArray.size(); j++)
                    {
                        p = DataTypes::getCPos(coords[indexArray[j]]);
                        tetraCoord.push_back(p);
                    }


                    sofa::helper::set<std::pair<Edge,size_t> >::iterator ite=bezierTetrahedronEdgeSet.begin();
                    for (; ite!=bezierTetrahedronEdgeSet.end(); ite++)
                    {
                        if ((*ite).second==2) {
                            glColor3f(0.0f, 1.0f, 0.0f);
                        } else 	if ((*ite).second==1)  {
                            glColor3f(0.0f, 0.0f, 1.0f );
                        } else {
                            glColor3f(1.0f, 0.0f, 1.0f );
                        }
                        glVertex3f(tetraCoord[(*ite).first[0]][0], tetraCoord[(*ite).first[0]][1], tetraCoord[(*ite).first[0]][2]);
                        glVertex3f(tetraCoord[(*ite).first[1]][0], tetraCoord[(*ite).first[1]][1], tetraCoord[(*ite).first[1]][2]);


                    }
                }
                glEnd();
            }
#endif // SOFA_NO_OPENGL
        }
    }

}



} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_TETEAHEDRONSETGEOMETRYALGORITHMS_INL
