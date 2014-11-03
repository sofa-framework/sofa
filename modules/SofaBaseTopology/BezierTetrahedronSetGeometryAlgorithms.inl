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
        ,drawVolumeEdges (core::objectmodel::Base::initData(&drawVolumeEdges, (bool) false, "drawVolumeEdges", "Debug : draw edges on Bezier volume"))
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
		size_t i;
		TetrahedronBezierIndex tbi;
		/// precompute the factorial of the degree.
		for (i=0;i<tbiArray.size();++i) {
			tbi=tbiArray[i];
			bernsteinCoefficientArray[i]=multinomial(degree,tbi); 
			bernsteinCoeffMap.insert(std::pair<TetrahedronBezierIndex,Real>(tbi,(double) bernsteinCoefficientArray[i]));
		}
		/// insert coefficient for the inferior degree
		size_t j,k,l,m,n,index1,index2;
		for (i=0;i<=(size_t)(degree-1);++i) {
			for (j=0;j<=(degree-i-1);++j) {
				for (k=0;k<=(degree-j-i-1);++k) {
					l=degree-1-i-j-k;
					tbi=TetrahedronBezierIndex(i,j,k,l);
					bernsteinCoeffMap.insert(std::pair<TetrahedronBezierIndex,Real>(tbi,(double) multinomial(degree-1,tbi)));
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
									Edge e(std::min(index1,index2),std::max(index1,index2));
									bezierTetrahedronEdgeSet.insert(e);
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
	VecPointID indexArray;
	TetrahedronBezierIndex tbi;
	container->getGlobalIndexArrayOfBezierPointsInTetrahedron(tetrahedronIndex, indexArray);
	for(size_t i=0; i<tbiArray.size(); ++i)
	{
		tbi=tbiArray[i];
		nodalValue+=p[indexArray[i]]*bernsteinCoefficientArray[i]*pow(barycentricCoordinate[0],tbi[0])*pow(barycentricCoordinate[1],tbi[1])*pow(barycentricCoordinate[2],tbi[2])*pow(barycentricCoordinate[3],tbi[3]);
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
	VecPointID indexArray;
	TetrahedronBezierIndex tbi;
	size_t j;
	Real val;
	container->getGlobalIndexArrayOfBezierPointsInTetrahedron(tetrahedronIndex, indexArray);
	for(size_t i=0; i<tbiArray.size(); ++i)
	{
		tbi=tbiArray[i];
		val=bernsteinCoefficientArray[i]*pow(barycentricCoordinate[0],tbi[0])*pow(barycentricCoordinate[1],tbi[1])*pow(barycentricCoordinate[2],tbi[2])*pow(barycentricCoordinate[3],tbi[3]);
		Vec4 dval(0,0,0,0);
		for (j=0;j<4;++j) {
			if(tbi[j] && barycentricCoordinate[j]){
				 dval[j]=(Real)tbi[j]*val/barycentricCoordinate[j];
			}
		}
		for (j=0;j<3;++j) {
			dpos[j]+=(dval[j]-dval[3])*p[indexArray[i]];
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
	VecPointID indexArray;
	TetrahedronBezierIndex tbi;
	size_t j;
	Real val;
	container->getGlobalIndexArrayOfBezierPointsInTetrahedron(tetrahedronIndex, indexArray);
	// initialize dpos
	for (j=0;j<4;++j) 
		dpos[j]=Coord();
	for(size_t i=0; i<tbiArray.size(); ++i)
	{
		tbi=tbiArray[i];
		val=bernsteinCoefficientArray[i]*pow(barycentricCoordinate[0],tbi[0])*pow(barycentricCoordinate[1],tbi[1])*pow(barycentricCoordinate[2],tbi[2])*pow(barycentricCoordinate[3],tbi[3]);
		Vec4 dval(0,0,0,0);
		for (j=0;j<4;++j) {
			if(tbi[j] && barycentricCoordinate[j]){
				 dval[j]=(Real)tbi[j]*val/barycentricCoordinate[j];
				 dpos[j]+=dval[j]*p[indexArray[i]];
			}
		}
	}
	
}

template<class DataTypes>
bool BezierTetrahedronSetGeometryAlgorithms<DataTypes>::isBezierTetrahedronAffine(const size_t tetrahedronIndex,const VecCoord& p, Real tolerance) const{
	// get the global indices of all points
	VecPointID indexArray;
	container->getGlobalIndexArrayOfBezierPointsInTetrahedron(tetrahedronIndex, indexArray);
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
#ifndef SOFA_NO_OPENGL
	if ((degree>0) && (container) ){
		TetrahedronSetGeometryAlgorithms<DataTypes>::draw(vparams);	
		// Draw Tetra
		if ((drawControlPointsEdges.getValue())||(drawVolumeEdges.getValue()))
		{
			const sofa::helper::vector<Tetrahedron> &tetraArray = this->m_topology->getTetrahedra();

			if (!tetraArray.empty())
			{
				glDisable(GL_LIGHTING);
				const sofa::defaulttype::Vec3f& color =  this->_drawColor.getValue();
				glColor3f(color[0], color[1], color[2]);
				glBegin(GL_LINES);
				const VecCoord& coords =(this->object->read(core::ConstVecCoordId::position())->getValue());
				VecPointID indexArray;
				Vec4 baryCoord;
				sofa::defaulttype::Vec3f p; //,p2;
				for (unsigned int i = 0; i<tetraArray.size(); i++)
				{
					indexArray.clear();
					container->getGlobalIndexArrayOfBezierPointsInTetrahedron(i, indexArray);
					sofa::helper::vector <sofa::defaulttype::Vec3f> tetraCoord;
					if (drawControlPointsEdges.getValue()) {
						for (unsigned int j = 0; j<indexArray.size(); j++)
						{
							p = DataTypes::getCPos(coords[indexArray[j]]);
							tetraCoord.push_back(p);
						}
					} else {
						for (unsigned int j = 0; j<indexArray.size(); j++)
						{
							baryCoord=Vec4((double)tbiArray[j][0]/degree,(double)tbiArray[j][1]/degree,(double)tbiArray[j][2]/degree,(double)tbiArray[j][3]/degree);
							p=DataTypes::getCPos(computeNodalValue(i,baryCoord));

						//	if ((p-p2).norm2()>1e-3) std::cerr<< "error in tetra"<< i << std::endl;*/
						//	p = DataTypes::getCPos(coords[indexArray[j]]);
							tetraCoord.push_back(p);
						}
					}
					sofa::helper::set<Edge>::iterator ite=bezierTetrahedronEdgeSet.begin();
					for (; ite!=bezierTetrahedronEdgeSet.end(); ite++)
					{
						glVertex3f(tetraCoord[(*ite)[0]][0], tetraCoord[(*ite)[0]][1], tetraCoord[(*ite)[0]][2]);
						glVertex3f(tetraCoord[(*ite)[1]][0], tetraCoord[(*ite)[1]][1], tetraCoord[(*ite)[1]][2]);
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

#endif // SOFA_COMPONENTS_TETEAHEDRONSETGEOMETRYALGORITHMS_INL
