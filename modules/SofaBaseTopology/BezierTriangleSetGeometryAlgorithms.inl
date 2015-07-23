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
#ifndef SOFA_COMPONENT_TOPOLOGY_BEZIERTRIANGLESETGEOMETRYALGORITHMS_INL
#define SOFA_COMPONENT_TOPOLOGY_BEZIERTRIANGLESETGEOMETRYALGORITHMS_INL

#include <SofaBaseTopology/BezierTriangleSetGeometryAlgorithms.h>
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
		,degree(0)
        ,drawControlPointsEdges (core::objectmodel::Base::initData(&drawControlPointsEdges, (bool) false, "drawControlPointsEdges", "Debug : draw Control point edges "))
        ,drawVolumeEdges (core::objectmodel::Base::initData(&drawVolumeEdges, (bool) false, "drawVolumeEdges", "Debug : draw edges on Bezier volume"))
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
			bernsteinCoeffMap.insert(std::pair<TriangleBezierIndex,Real>(tbi,(double) bernsteinCoefficientArray[i]));
		}
		/// insert coefficient for the inferior degree
		BezierDegreeType i,j,k,l,m,n,index1,index2;
		for (i=0;i<=(degree-1);++i) {
			for (j=0;j<=(degree-i-1);++j) {
				k=degree-1-i-j;
				tbi=TriangleBezierIndex(i,j,k);
				bernsteinCoeffMap.insert(std::pair<TriangleBezierIndex,Real>(tbi,(double) multinomial(degree-1,tbi)));
			}
		}
	
		/// fills the array of edges
		bezierTriangleEdgeSet.clear();
		TriangleBezierIndex tbiNext;

		for (i=0;i<=degree;++i) {
			for (j=0;j<=(degree-i-1);++j) {
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
								bezierTriangleEdgeSet.insert(e);
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
	container->getGlobalIndexArrayOfBezierPointsInTriangle(triangleIndex, indexArray);
	for(size_t i=0; i<tbiArray.size(); ++i)
	{
		tbi=tbiArray[i];
		nodalValue+=p[indexArray[i]]*bernsteinCoefficientArray[i]*pow(barycentricCoordinate[0],tbi[0])*pow(barycentricCoordinate[1],tbi[1])*pow(barycentricCoordinate[2],tbi[2]);
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
	TriangleBezierIndex tbi;
	size_t j;
	Real val;
	container->getGlobalIndexArrayOfBezierPointsInTriangle(triangleIndex, indexArray);
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
	size_t j;
	Real val;
	container->getGlobalIndexArrayOfBezierPointsInTriangle(triangleIndex, indexArray);
	// initialize dpos
	for (j=0;j<3;++j) 
		dpos[j]=Coord();
	for(size_t i=0; i<tbiArray.size(); ++i)
	{
		tbi=tbiArray[i];
		val=bernsteinCoefficientArray[i]*pow(barycentricCoordinate[0],tbi[0])*pow(barycentricCoordinate[1],tbi[1])*pow(barycentricCoordinate[2],tbi[2]);
		Vec3 dval(0,0,0);
		for (j=0;j<3;++j) {
			if(tbi[j] && barycentricCoordinate[j]){
				 dval[j]=(Real)tbi[j]*val/barycentricCoordinate[j];
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
#ifndef SOFA_NO_OPENGL
	
	if ((degree>0) && (container) ){
		TriangleSetGeometryAlgorithms<DataTypes>::draw(vparams);	
		// Draw Tetra
		if ((drawControlPointsEdges.getValue())||(drawVolumeEdges.getValue()))
		{
			const sofa::helper::vector<Triangle> &trianArray = this->m_topology->getTriangles();

			if (!trianArray.empty())
			{
				glDisable(GL_LIGHTING);
				const sofa::defaulttype::Vec3f& color =  this->_drawColor.getValue();
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
					if (drawControlPointsEdges.getValue()) {
						for (unsigned int j = 0; j<indexArray.size(); j++)
						{
							p = DataTypes::getCPos(coords[indexArray[j]]);
							trianCoord.push_back(p);
						}
					} else {
						for (unsigned int j = 0; j<indexArray.size(); j++)
						{
							baryCoord=Vec3((double)tbiArray[j][0]/degree,(double)tbiArray[j][1]/degree,(double)tbiArray[j][2]/degree);
							p=DataTypes::getCPos(computeNodalValue(i,baryCoord));

						//	if ((p-p2).norm2()>1e-3) std::cerr<< "error in tetra"<< i << std::endl;
						//	p = DataTypes::getCPos(coords[indexArray[j]]);
							trianCoord.push_back(p);
						}
					}
					sofa::helper::set<Edge>::iterator ite=bezierTriangleEdgeSet.begin();
					for (; ite!=bezierTriangleEdgeSet.end(); ite++)
					{
						glVertex3f(trianCoord[(*ite)[0]][0], trianCoord[(*ite)[0]][1], trianCoord[(*ite)[0]][2]);
						glVertex3f(trianCoord[(*ite)[1]][0], trianCoord[(*ite)[1]][1], trianCoord[(*ite)[1]][2]);
					}
				}
				glEnd();
			}
		}
	}
	
#endif
}



} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_TRIANGLESETGEOMETRYALGORITHMS_INL
