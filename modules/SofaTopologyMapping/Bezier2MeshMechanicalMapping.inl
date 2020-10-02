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
#ifndef SOFA_COMPONENT_MAPPING_BEZIER2MESHMECHANICALMAPPING_INL
#define SOFA_COMPONENT_MAPPING_BEZIER2MESHMECHANICALMAPPING_INL

#include "Bezier2MeshMechanicalMapping.h"

#include <SofaTopologyMapping/Bezier2MeshTopologicalMapping.h>
#include <SofaGeneralTopology/BezierTriangleSetTopologyContainer.h>
#include <SofaGeneralTopology/BezierTriangleSetGeometryAlgorithms.h>
#include <SofaBaseTopology/CommonAlgorithms.h>

namespace sofa
{

namespace component
{

namespace mapping
{


template <class TIn, class TOut>
Bezier2MeshMechanicalMapping<TIn, TOut>::Bezier2MeshMechanicalMapping(core::State<In>* from, core::State<Out>* to)
    : Inherit(from, to)
    , topoMap(NULL)
    , bezierDegree(0)
    , tesselationDegree(0)
{
}

template <class TIn, class TOut>
Bezier2MeshMechanicalMapping<TIn, TOut>::~Bezier2MeshMechanicalMapping()
{
}

double multinomial(const size_t n,const sofa::component::topology::TriangleBezierIndex tbiIn)
{
	size_t i,ival;
	sofa::component::topology::TriangleBezierIndex tbi=tbiIn;
	// divide n! with the largest of the multinomial coefficient
	std::sort(tbi.begin(),tbi.end());
	ival=1;
	for (i=n;i>tbi[2];--i){
		ival*=i;
	}
	return(((double)ival)/(sofa::helper::factorial(tbi[0])*sofa::helper::factorial(tbi[1])));
}
template <class TIn, class TOut>
void Bezier2MeshMechanicalMapping<TIn, TOut>::init()
{

	this->getContext()->get(topoMap);
	if (!topoMap) {
		serr << "Could not find any Bezier2MeshTopologicalMapping object"<<sendl;
		return;
	}
	this->fromModel->getContext()->get(btstc);
	if (!btstc){
		topoMap->fromModel->getContext()->get(btstc);
		if (!btstc){
			serr << "Could not find any BezierTriangleSetTopologyContainer object"<<sendl;
			return;
		}
	}
	this->fromModel->getContext()->get(btsga);
	if (!btsga){
		topoMap->fromModel->getContext()->get(btsga);
		if (!btsga){
			serr << "Could not find any BezierTriangleSetGeometryAlgorithms object"<<sendl;
			return;
		}
	}
	bezierDegree=btstc->getDegree();
	tesselationDegree=topoMap->d_tesselationTriangleDegree.getValue();
	size_t bezierTesselation=tesselationDegree;
	size_t i;
	// resize bezierTesselationWeightArray
	bezierTesselationWeightArray.resize(topoMap->nbPoints);
	this->toModel->resize(topoMap->nbPoints);

	if (precomputedLinearBernsteinCoefficientArray.size()!=(bezierTesselation-1)) {
		// (re)compute the interpolation factor associated with a given tesselation level and bezier order
		precomputedLinearBernsteinCoefficientArray.resize(bezierTesselation-1);
		Real u,v;
		size_t j;
		for (i=1;i<bezierTesselation;++i) {
			sofa::helper::vector<Real> weightArray(bezierDegree+1);
			// linear barycentric coordinates of the point
			u=(Real)i/(Real)bezierTesselation;
			v=(Real)(1.0f-u);
			for(j=0;j<=bezierDegree;++j){
				weightArray[j]=(Real)(pow(u,(int)j)*pow(v,(int)(bezierDegree-j))*sofa::component::topology::binomial<Real>(bezierDegree-j,j));
			}
			precomputedLinearBernsteinCoefficientArray[i-1]=weightArray;
		}

	}

	if (precomputedTriangularBernsteinCoefficientArray.size()!= ((bezierTesselation-1)*(bezierTesselation-2)/2)) {
		// for each point inside the tesselated triangle, store the weights of each control points (bivariate Bernstein polynomials)
		// there are (bezierTesselation-1)*(bezierTesselation-2)/2 points inside each triangle and (bezierDegree+1)*(bezierDegree+2)/2 control points 
		// (re)compute the interpolation factor associated with a given tesselation level and bezier order
		precomputedTriangularBernsteinCoefficientArray.resize( ((bezierTesselation-1)*(bezierTesselation-2)/2));
		Real u,v,w;
		size_t j,k,ind;
		sofa::helper::vector<sofa::component::topology::TriangleBezierIndex> tbia=btstc->getTriangleBezierIndexArray();
		for (ind=0,i=1;i<(bezierTesselation-1);++i) {
			for (j=1;j<(bezierTesselation-i);++j,++ind) {

				sofa::helper::vector<Real> weightArray(tbia.size());
				// linear barycentric coordinates of the point
				u=(Real)i/(Real)bezierTesselation;
				v=(Real)j/(Real)bezierTesselation;
				w=(Real)(1.0f-u-v);
				for(k=0;k<tbia.size();++k){
					weightArray[k]=(Real)(pow(u,tbia[k][0])*pow(v,tbia[k][1])*pow(w,tbia[k][2])*multinomial(bezierDegree,tbia[k]));
				}
				precomputedTriangularBernsteinCoefficientArray[ind]=weightArray;
			}
		}

	}
	// get the local indices of the points in the macro triangle
	if (tesselatedTriangleIndices.size()!=(bezierTesselation+1)*(bezierTesselation+2)/2 ) {
		tesselatedTriangleIndices=btstc->getTriangleBezierIndexArrayOfGivenDegree(bezierTesselation);
	}
	if (precomputedDerivUTriangularBernsteinCoefficientArray.size()!=((bezierTesselation+1)*(bezierTesselation+2)/2)) {
		precomputedDerivUTriangularBernsteinCoefficientArray.resize( ((bezierTesselation+1)*(bezierTesselation+2)/2));
		precomputedDerivVTriangularBernsteinCoefficientArray.resize( ((bezierTesselation+1)*(bezierTesselation+2)/2));

		size_t j,k,l;
		Real val;
		InCoord barycentricCoordinate;
		sofa::helper::vector<sofa::component::topology::TriangleBezierIndex> tbia=btstc->getTriangleBezierIndexArray();


		for (j=0;j<tesselatedTriangleIndices.size();++j) {
			for (k=0;k<3;++k) 
				barycentricCoordinate[k]=(Real)tesselatedTriangleIndices[j][k]/(Real)bezierTesselation;


			sofa::helper::vector<Real> weightArrayDU(tbia.size());
			sofa::helper::vector<Real> weightArrayDV(tbia.size());
			for(k=0;k<tbia.size();++k){
				val=(Real)(pow( barycentricCoordinate[0],tbia[k][0])*pow( barycentricCoordinate[1],tbia[k][1])*pow( barycentricCoordinate[2],tbia[k][2])*multinomial(bezierDegree,tbia[k]));
				InCoord dval;
				for (l=0;l<3;++l) {
					if(tbia[k][l] && barycentricCoordinate[l]){
						dval[l]=(Real)tbia[k][l]*val/barycentricCoordinate[l];
					} else if ((barycentricCoordinate[l]==0.0f)&&(tbia[k][l]==1)) {
						dval[l]=multinomial(bezierDegree,tbia[k]);
						for (i=1;i<=2;++i)
							dval[l]*=(Real)(pow( barycentricCoordinate[(l+i)%3],tbia[k][(l+i)%3]));
					}
				}
				weightArrayDU[k]=dval[0]-dval[2];
				weightArrayDV[k]=dval[1]-dval[2];
			}
			precomputedDerivUTriangularBernsteinCoefficientArray[j]=weightArrayDU;
			precomputedDerivVTriangularBernsteinCoefficientArray[j]=weightArrayDV;
		}

	}
	this->Inherit::init();
}


template <class TIn, class TOut>
void Bezier2MeshMechanicalMapping<TIn, TOut>::apply(const core::MechanicalParams * /*mparams*/, Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
{
	helper::WriteAccessor< Data<OutVecCoord> > out = dOut;
	helper::ReadAccessor< Data<InVecCoord> > in = dIn;


	size_t i;
	// copy the points of underlying triangulation
	for (i=0;i<btstc->getNumberOfTriangularPoints();++i) {
		out[i]=in[topoMap->local2GlobalBezierVertexArray[i]];
		bezierTesselationWeightArray[i]=btstc->getWeight(topoMap->local2GlobalBezierVertexArray[i]);
	}
	const size_t bezierTesselation=tesselationDegree;

	// copy the points on  the edges pf the Bezier patches
	if (bezierTesselation>1) {
		size_t l,k;
		size_t edgeRank=0;
		sofa::helper::vector<InCoord> edgeControlPoints;
		sofa::helper::vector<Real> edgeControlPointWeights;
		InCoord p;
		Real weight;
		sofa::helper::vector<sofa::core::topology::Topology::Edge >::iterator ite=topoMap->edgeTriangleArray.begin();
		for (;ite!=topoMap->edgeTriangleArray.end();++ite,++edgeRank) {
			// get control points along edges and their weights
			edgeControlPoints.clear();
			edgeControlPointWeights.clear();
			// must decide if the edge is a rational Bezier or integral Bezier curve
			const core::topology::BaseMeshTopology::TrianglesAroundEdge &tae=btstc->getTrianglesAroundEdge(edgeRank);
			 bool isRational=false;
			for (l=0;l<tae.size();++l){
				if (btstc->isRationalSpline(tae[l]))
					isRational=true;
			}
				
			edgeControlPoints.push_back(in[(*ite)[0]]);
			if (isRational)
				edgeControlPointWeights.push_back(btstc->getWeight((*ite)[0]));
			sofa::helper::vector<size_t> bezierEdge=topoMap->bezierEdgeArray[edgeRank];
			for(l=1;l<=(bezierDegree-1);++l) {

				edgeControlPoints.push_back(in[bezierEdge[l-1]]);
				if (isRational)
					edgeControlPointWeights.push_back(btstc->getWeight(bezierEdge[l-1]));
			}
			edgeControlPoints.push_back(in[(*ite)[1]]);
			if (isRational)
				edgeControlPointWeights.push_back(btstc->getWeight((*ite)[1]));

			// then interpolate position based on the 2 previous arrays
			size_t  tesselatedEdgeRank=btstc->getNumberOfTriangularPoints()+edgeRank*(bezierTesselation-1);
			for (l=1;l<bezierTesselation;++l) {

				p=InCoord();
				sofa::helper::vector<Real> weightArray=precomputedLinearBernsteinCoefficientArray[l-1];
				if (isRational) {
					weight=0;
					for (k=0;k<=bezierDegree;++k) {
						/// univariate Bernstein polynomial
						p+=edgeControlPoints[k]*weightArray[k]*edgeControlPointWeights[k];
						weight+=edgeControlPointWeights[k]*weightArray[k];
					}
				} else{
					for (k=0;k<=bezierDegree;++k) {
						/// univariate Bernstein polynomial
						p+=edgeControlPoints[k]*weightArray[k];
					}
				}
				if (isRational) {
					p/=weight;
					bezierTesselationWeightArray[tesselatedEdgeRank]=weight;
				}

				out[tesselatedEdgeRank++]=p;
			}
		}
		// copy the points on  the ed the Bezier patches
		if (bezierTesselation>2) {
			 sofa::component::topology::BezierTriangleSetTopologyContainer::VecPointID indexArray;
			 size_t j,ind;
			size_t tesselatedTriangleRank=btstc->getNumberOfTriangularPoints()+btstc->getNbEdges()*(bezierTesselation-1);
			for (i=0;i<btstc->getNumberOfTriangles();++i) {

				 bool isRational=btstc->isRationalSpline(i);
				// first get  the Bezier control points in the triangle 
				btstc->getGlobalIndexArrayOfBezierPointsInTriangle(i, indexArray);
				for (ind=0,j=1;j<(bezierTesselation-1);++j) {
					for (k=1;k<(bezierTesselation-j);++k,++ind) {
						p=InCoord();
						sofa::helper::vector<Real> &weigthArray=precomputedTriangularBernsteinCoefficientArray[ind];
						if (isRational) {
							weight=0;
							for (l=0;l<indexArray.size();++l) {
								p+=in[indexArray[l]]*btstc->getWeight( indexArray[l])*weigthArray[l];
								weight+=btstc->getWeight( indexArray[l])*weigthArray[l];
							}
						} else {
							for (l=0;l<indexArray.size();++l) {
								/// univariate Bernstein polynomial
								p+=in[indexArray[l]]*weigthArray[l];
							}
						}
						if (isRational) {
							p/=weight;
							bezierTesselationWeightArray[tesselatedTriangleRank]=weight;
						}
						// store the triangle point on the surface
						out[tesselatedTriangleRank++]=p;
					}
				}

			}
		}
	}
	/* 

	  // then compute normals if necessary
		  if (m_useBezierNormals.getValue()) {
			  m_updateNormals.setValue(false);
	
			  sofa::component::topology::BezierTriangleSetTopologyContainer::VecPointID indexArray;

			  for (i=0;i<btstc->getNumberOfTriangles();++i) {
				  // first get  the Bezier control points in the triangle 
				  btstc->getGlobalIndexArrayOfBezierPointsInTriangle(i, indexArray);

				  for (j=0;j<tesselatedTriangleIndices.size();++j) {
					  Deriv du,dv,normal,pos;
					  Real dweightu,dweightv,weight;
					  dweightu=dweightv=0.0f;
					  // for each point in the tesselated triangle compute dpos/du and dpos/dv as a weighted combination of the control points
					  sofa::helper::vector<Real> &weigthArrayDU=precomputedDerivUTriangularBernsteinCoefficientArray[j];
					  sofa::helper::vector<Real> &weigthArrayDV=precomputedDerivVTriangularBernsteinCoefficientArray[j];
					  if (isRational){
						  for (k=0;k<indexArray.size();++k) {
							  du+=bezierControlPointsArray[indexArray[k]]*bezierWeightArray[indexArray[k]]*weigthArrayDU[k];
							  dv+=bezierControlPointsArray[indexArray[k]]*bezierWeightArray[indexArray[k]]*weigthArrayDV[k];
							  dweightu+=bezierWeightArray[indexArray[k]]*weigthArrayDU[k];
							  dweightv+=bezierWeightArray[indexArray[k]]*weigthArrayDV[k];
						  }
					  } else {
						  for (k=0;k<indexArray.size();++k) {
							  du+=bezierControlPointsArray[indexArray[k]]*weigthArrayDU[k];
							  dv+=bezierControlPointsArray[indexArray[k]]*weigthArrayDV[k];
						  }
					  }
					  if (isRational){
						  du=(du-dweightu*vertices[globalIndexTesselatedBezierTriangleArray[i][j]])/bezierTesselationWeightArray[globalIndexTesselatedBezierTriangleArray[i][j]];
						  dv=(dv-dweightv*vertices[globalIndexTesselatedBezierTriangleArray[i][j]])/bezierTesselationWeightArray[globalIndexTesselatedBezierTriangleArray[i][j]];
					  }
					  normal=cross(du,dv);

					  normals[globalIndexTesselatedBezierTriangleArray[i][j]]+=normal;
				  }
			  }
			  // renormalize all normals. Normals at vertex or along edges are averaged
			  for (i = 0; i < normals.size(); i++)
				  normals[i].normalize();
				  }
				  */

	

}

template <class TIn, class TOut>
void Bezier2MeshMechanicalMapping<TIn, TOut>::applyJ(const core::MechanicalParams * /*mparams*/, Data<OutVecDeriv>& dOut, const Data<InVecDeriv>& dIn)
{
    if (!topoMap) return;
	
    helper::WriteAccessor< Data<OutVecDeriv> > out = dOut;
    helper::ReadAccessor< Data<InVecDeriv> > in = dIn;


	size_t i;
	// copy the points of underlying triangulation
	for (i=0;i<btstc->getNumberOfTriangularPoints();++i) {
		out[i]=in[topoMap->local2GlobalBezierVertexArray[i]];
		
	}
	const size_t bezierTesselation=tesselationDegree;
	
	// copy the points on  the edges pf the Bezier patches
	if (bezierTesselation>1) {
		size_t l,k;
		size_t edgeRank=0;
		sofa::helper::vector<InCoord> edgeControlPoints;
		sofa::helper::vector<Real> edgeControlPointWeights;
		InCoord p;
		Real weight;
		sofa::helper::vector<sofa::core::topology::Topology::Edge >::iterator ite=topoMap->edgeTriangleArray.begin();
		for (;ite!=topoMap->edgeTriangleArray.end();++ite,++edgeRank) {
			// get control points along edges and their weights
			edgeControlPoints.clear();
			edgeControlPointWeights.clear();
			// must decide if the edge is a rational Bezier or integral Bezier curve
			const core::topology::BaseMeshTopology::TrianglesAroundEdge &tae=btstc->getTrianglesAroundEdge(edgeRank);
			bool isRational=false;
			for (l=0;l<tae.size();++l){
				if (btstc->isRationalSpline(tae[l]))
					isRational=true;
			}
			edgeControlPoints.push_back(in[(*ite)[0]]);
			if (isRational)
				edgeControlPointWeights.push_back(btstc->getWeight((*ite)[0]));
			sofa::helper::vector<size_t> bezierEdge=topoMap->bezierEdgeArray[edgeRank];
			for(l=1;l<=(bezierDegree-1);++l) {

				edgeControlPoints.push_back(in[bezierEdge[l-1]]);
				if (isRational)
					edgeControlPointWeights.push_back(btstc->getWeight(bezierEdge[l-1]));
			}
			edgeControlPoints.push_back(in[(*ite)[1]]);
			if (isRational)
				edgeControlPointWeights.push_back(btstc->getWeight((*ite)[1]));

			// then interpolate position based on the 2 previous arrays
			size_t  tesselatedEdgeRank=btstc->getNumberOfTriangularPoints()+edgeRank*(bezierTesselation-1);
			for (l=1;l<bezierTesselation;++l) {

				p=InCoord();
				sofa::helper::vector<Real> weightArray=precomputedLinearBernsteinCoefficientArray[l-1];
				if (isRational) {
					weight=0;
					for (k=0;k<=bezierDegree;++k) {
						/// univariate Bernstein polynomial
						p+=edgeControlPoints[k]*weightArray[k]*edgeControlPointWeights[k];
						weight+=edgeControlPointWeights[k]*weightArray[k];
					}
				} else{
					for (k=0;k<=bezierDegree;++k) {
						/// univariate Bernstein polynomial
						p+=edgeControlPoints[k]*weightArray[k];
					}
				}
				if (isRational) {
					p/=weight;
					bezierTesselationWeightArray[tesselatedEdgeRank]=weight;
				}

				out[tesselatedEdgeRank++]=p;
			}
		}
		// copy the points on  the ed the Bezier patches
		if (bezierTesselation>2) {
			 sofa::component::topology::BezierTriangleSetTopologyContainer::VecPointID indexArray;
			 size_t j,ind;
			size_t tesselatedTriangleRank=btstc->getNumberOfTriangularPoints()+btstc->getNbEdges()*(bezierTesselation-1);
			for (i=0;i<btstc->getNumberOfTriangles();++i) {
				 bool isRational=btstc->isRationalSpline(i);
				// first get  the Bezier control points in the triangle 
				btstc->getGlobalIndexArrayOfBezierPointsInTriangle(i, indexArray);
				for (ind=0,j=1;j<(bezierTesselation-1);++j) {
					for (k=1;k<(bezierTesselation-j);++k,++ind) {
						p=InCoord();
						sofa::helper::vector<Real> &weigthArray=precomputedTriangularBernsteinCoefficientArray[ind];
						if (isRational) {
							weight=0;
							for (l=0;l<indexArray.size();++l) {
								p+=in[indexArray[l]]*btstc->getWeight( indexArray[l])*weigthArray[l];
								weight+=btstc->getWeight( indexArray[l])*weigthArray[l];
							}
						} else {
							for (l=0;l<indexArray.size();++l) {
								/// univariate Bernstein polynomial
								p+=in[indexArray[l]]*weigthArray[l];
							}
						}
						if (isRational) {
							p/=weight;
							bezierTesselationWeightArray[tesselatedTriangleRank]=weight;
						}
						// store the triangle point on the surface
						out[tesselatedTriangleRank++]=p;
					}
				}

			}
		}
	}



}

template <class TIn, class TOut>
void Bezier2MeshMechanicalMapping<TIn, TOut>::applyJT(const core::MechanicalParams * /*mparams*/, Data<InVecDeriv>& dOut, const Data<OutVecDeriv>& dIn)
{
    if (!topoMap) return;

    helper::WriteAccessor< Data<InVecDeriv> > out = dOut;
    helper::ReadAccessor< Data<OutVecDeriv> > in = dIn;

	
	size_t i;
	// copy the points of underlying triangulation
	for (i=0;i<btstc->getNumberOfTriangularPoints();++i) {
		out[i]=in[topoMap->local2GlobalBezierVertexArray[i]];
		
	}
		const size_t bezierTesselation=tesselationDegree;
	
	// copy the points on  the edges pf the Bezier patches
	if (bezierTesselation>1) {
		size_t l,k;
		size_t edgeRank=0;

		Real weight;
		sofa::helper::vector<sofa::core::topology::Topology::Edge >::iterator ite=topoMap->edgeTriangleArray.begin();
		for (;ite!=topoMap->edgeTriangleArray.end();++ite,++edgeRank) {
			// must decide if the edge is a rational Bezier or integral Bezier curve
			const core::topology::BaseMeshTopology::TrianglesAroundEdge &tae=btstc->getTrianglesAroundEdge(edgeRank);
			bool isRational=false;
			for (l=0;l<tae.size();++l){
				if (btstc->isRationalSpline(tae[l]))
					isRational=true;
			}
			// get control points along edges and their weights
			sofa::helper::vector<size_t> bezierEdge=topoMap->bezierEdgeArray[edgeRank];
			size_t  tesselatedEdgeRank=btstc->getNumberOfTriangularPoints()+edgeRank*(bezierTesselation-1);
			if (isRational) {
				for (l=1;l<bezierTesselation;++l) {
					sofa::helper::vector<Real> weightArray=precomputedLinearBernsteinCoefficientArray[l-1];
					// first compute the weight at the denominator
					weight=btstc->getWeight((*ite)[0])*weightArray[0];
					for(k=1;k<=(bezierDegree-1);++k) {
						weight+=weightArray[k]*btstc->getWeight(bezierEdge[k-1]);
					}
					weight+=btstc->getWeight((*ite)[1])*weightArray[bezierDegree];
					// now compute the normalized weight
					out[(*ite)[0]]+=in[tesselatedEdgeRank]*weightArray[0]*btstc->getWeight((*ite)[0])/weight;
					for(k=1;k<=(bezierDegree-1);++k) {
						out[bezierEdge[k-1]]+=in[tesselatedEdgeRank]*weightArray[k]*btstc->getWeight(bezierEdge[k-1])/weight;
					}
					out[(*ite)[1]]+=in[tesselatedEdgeRank]*weightArray[bezierDegree]*btstc->getWeight((*ite)[1])/weight;
					tesselatedEdgeRank++;
				}
			} else {

				for (l=1;l<bezierTesselation;++l) {
					sofa::helper::vector<Real> weightArray=precomputedLinearBernsteinCoefficientArray[l-1];
					out[(*ite)[0]]+=in[tesselatedEdgeRank]*weightArray[0];
					for(k=1;k<=(bezierDegree-1);++k) {
						out[bezierEdge[k-1]]+=in[tesselatedEdgeRank]*weightArray[k];
					}
					out[(*ite)[1]]+=in[tesselatedEdgeRank]*weightArray[bezierDegree];
					tesselatedEdgeRank++;
				}

			}
		}
		// update according to the control points located inside triangles
		if (bezierTesselation>2) {
			 sofa::component::topology::BezierTriangleSetTopologyContainer::VecPointID indexArray;
			 size_t j,ind;
			size_t tesselatedTriangleRank=btstc->getNumberOfTriangularPoints()+btstc->getNbEdges()*(bezierTesselation-1);
			for (i=0;i<btstc->getNumberOfTriangles();++i) {
					 bool isRational=btstc->isRationalSpline(i);
				// first get  the Bezier control points in the triangle 
				btstc->getGlobalIndexArrayOfBezierPointsInTriangle(i, indexArray);
				for (ind=0,j=1;j<(bezierTesselation-1);++j) {
					for (k=1;k<(bezierTesselation-j);++k,++ind) {

						sofa::helper::vector<Real> &weigthArray=precomputedTriangularBernsteinCoefficientArray[ind];
						if (isRational) {
							// first compute the whole weight at the denominator
							weight=0;
							for (l=0;l<indexArray.size();++l) {
								weight+=btstc->getWeight( indexArray[l])*weigthArray[l];
							}
							// then compute the output
							for (l=0;l<indexArray.size();++l) {
								out[indexArray[l]]+=in[tesselatedTriangleRank]*weigthArray[l]*btstc->getWeight( indexArray[l])/weight;
							}
						} else  {

							for (l=0;l<indexArray.size();++l) {
								out[indexArray[l]]+=in[tesselatedTriangleRank]*weigthArray[l];
							}
						} 
						tesselatedTriangleRank++;
		
					}
				}

			}
		}
	}


}


template <class TIn, class TOut>
void Bezier2MeshMechanicalMapping<TIn, TOut>::applyJT(const core::ConstraintParams * /*cparams*/, Data<InMatrixDeriv>& /*dOut*/, const Data<OutMatrixDeriv>& /*dIn*/)
{

    if (!topoMap)
        return;

//    const sofa::helper::vector< std::pair< Mesh2PointTopologicalMapping::Element, int> >& pointSource = topoMap->getPointSource();

  //  if (pointSource.empty())
   //     return;
	/*
    InMatrixDeriv& out = *dOut.beginEdit();
    const OutMatrixDeriv& in = dIn.getValue();

    const core::topology::BaseMeshTopology::SeqEdges& edges = inputTopo->getEdges();
    const core::topology::BaseMeshTopology::SeqTriangles& triangles = inputTopo->getTriangles();
    const core::topology::BaseMeshTopology::SeqQuads& quads = inputTopo->getQuads();
    const core::topology::BaseMeshTopology::SeqTetrahedra& tetrahedra = inputTopo->getTetrahedra();
    const core::topology::BaseMeshTopology::SeqHexahedra& hexahedra = inputTopo->getHexahedra();

    typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

    for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();
        typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();

        // Creates a constraints if the input constraint is not empty.
        if (colIt != colItEnd)
        {
            typename In::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());

            while (colIt != colItEnd)
            {
                const unsigned int indexIn = colIt.index();
                const OutDeriv data = colIt.val();
                std::pair< Mesh2PointTopologicalMapping::Element, int> source = pointSource[indexIn];

                switch (source.first)
                {
                case topology::Mesh2PointTopologicalMapping::POINT:
                {
                    o.addCol(source.second, data);

                    break;
                }
                case topology::Mesh2PointTopologicalMapping::EDGE:
                {
                    core::topology::BaseMeshTopology::Edge e = edges[source.second];
                    typename In::Deriv f = data;
                    double fx = topoMap->getEdgeBaryCoords()[indexIn][0];

                    o.addCol(e[0], f * (1 - fx));
                    o.addCol(e[1], f * fx);

                    break;
                }
                case topology::Mesh2PointTopologicalMapping::TRIANGLE:
                {
                    core::topology::BaseMeshTopology::Triangle t = triangles[source.second];
                    typename In::Deriv f = data;
                    double fx = topoMap->getTriangleBaryCoords()[indexIn][0];
                    double fy = topoMap->getTriangleBaryCoords()[indexIn][1];

                    o.addCol(t[0], f * (1 - fx - fy));
                    o.addCol(t[1], f * fx);
                    o.addCol(t[2], f * fy);

                    break;
                }
                case topology::Mesh2PointTopologicalMapping::QUAD:
                {
                    core::topology::BaseMeshTopology::Quad q = quads[source.second];
                    typename In::Deriv f = data;
                    double fx = topoMap->getQuadBaryCoords()[indexIn][0];
                    double fy = topoMap->getQuadBaryCoords()[indexIn][1];

                    o.addCol(q[0], f * ((1-fx) * (1-fy)));
                    o.addCol(q[1], f * (fx * (1-fy)));
					o.addCol(q[2], f * ((1-fx) * fy));
					o.addCol(q[3], f * (fx * fy));

                    break;
                }
                case topology::Mesh2PointTopologicalMapping::TETRA:
                {
                    core::topology::BaseMeshTopology::Tetra t = tetrahedra[source.second];
                    typename In::Deriv f = data;
                    double fx = topoMap->getTetraBaryCoords()[indexIn][0];
                    double fy = topoMap->getTetraBaryCoords()[indexIn][1];
                    double fz = topoMap->getTetraBaryCoords()[indexIn][2];

                    o.addCol(t[0], f * (1-fx-fy-fz));
                    o.addCol(t[1], f * fx);
					o.addCol(t[2], f * fy);
					o.addCol(t[3], f * fz);

                    break;
                }
                case topology::Mesh2PointTopologicalMapping::HEXA:
                {
                    core::topology::BaseMeshTopology::Hexa h = hexahedra[source.second];
                    typename In::Deriv f = data;
                    const double fx = topoMap->getHexaBaryCoords()[indexIn][0];
                    const double fy = topoMap->getHexaBaryCoords()[indexIn][1];
                    const double fz = topoMap->getHexaBaryCoords()[indexIn][2];
                    const double oneMinFx = 1 - fx;
                    const double oneMinFy = 1 - fy;
                    const double oneMinFz = 1 - fz;

                    o.addCol(h[0] , f * oneMinFx * oneMinFy * oneMinFz);
                    o.addCol(h[1] , f * fx * oneMinFy * oneMinFz);
					o.addCol(h[3] , f * oneMinFx * fy * oneMinFz);
					o.addCol(h[2] , f * fx * fy * oneMinFz);
					o.addCol(h[4] , f * oneMinFx * oneMinFy * fz);
					o.addCol(h[5] , f * fx * oneMinFy * fz);
					o.addCol(h[6] , f * fx * fy * fz);
					o.addCol(h[7] , f * oneMinFx * fy * fz);

                    break;
                }
                default:

                    break;
                }

                ++colIt;
            }
        }
    }
	
    dOut.endEdit(); */
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
