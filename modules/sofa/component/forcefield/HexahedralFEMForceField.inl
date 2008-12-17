/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_HEXAHEDRALFEMFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_HEXAHEDRALFEMFORCEFIELD_INL

#include <sofa/core/componentmodel/behavior/ForceField.inl>
#include <sofa/component/forcefield/HexahedralFEMForceField.h>
#include <sofa/helper/PolarDecompose.h>
#include <sofa/helper/gl/template.h>
#include <assert.h>
#include <iostream>
#include <set>

#include <sofa/component/topology/HexahedronData.inl>



using std::set;



// indices ordering  (same as in HexahedronSetTopology):
//
// 	   Y  7---------6
//     ^ /	       /|
//     |/	 Z    / |
//     3----^----2  |
//     |   /	 |  |
//     |  4------|--5
//     | / 	     | /
//     |/	     |/
//     0---------1-->X



namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;
using namespace	sofa::component::topology;
using namespace core::componentmodel::topology;

template< class DataTypes>
void HexahedralFEMForceField<DataTypes>::FHexahedronCreationFunction (int hexahedronIndex, void* param,
        HexahedronInformation &,
        const Hexahedron& ,
        const helper::vector< unsigned int > &,
        const helper::vector< double >&)
{
    HexahedralFEMForceField<DataTypes> *ff= (HexahedralFEMForceField<DataTypes> *)param;
    if (ff)
    {

        switch(ff->method)
        {
        case LARGE :
            ff->computeMaterialStiffness(hexahedronIndex);
            ff->initLarge(hexahedronIndex);

            break;
        case POLAR :
            ff->computeMaterialStiffness(hexahedronIndex);
            ff->initPolar(hexahedronIndex);
            break;
        }
    }
}

template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::parse(core::objectmodel::BaseObjectDescription* arg)
{
    this->core::componentmodel::behavior::ForceField<DataTypes>::parse(arg);
    //this->setComputeGlobalMatrix(std::string(arg->getAttribute("computeGlobalMatrix","false"))=="true"); // necessary ?

}

template <class DataTypes> void HexahedralFEMForceField<DataTypes>::handleTopologyChange()
{
    std::list<const TopologyChange *>::const_iterator itBegin=_topology->firstChange();
    std::list<const TopologyChange *>::const_iterator itEnd=_topology->lastChange();


    hexahedronInfo.handleTopologyEvents(itBegin,itEnd);

}

template <class DataTypes>
void HexahedralFEMForceField<DataTypes>::init()
{
    if(_alreadyInit)return;
    else _alreadyInit=true;

    this->core::componentmodel::behavior::ForceField<DataTypes>::init();

    _topology = getContext()->getMeshTopology();

    if (_topology->getNbHexas()==0)
    {
        serr << "ERROR(HexahedralFEMForceField): object must have a Hexahedral Set Topology."<<sendl;
        return;
    }


    /*
    if( this->getContext()->getMeshTopology()==NULL )
    {
    	serr << "ERROR(HexahedralFEMForceField): object must have a Topology."<<sendl;
    	return;
    }

    if ( _topology==NULL)
    {
    	serr << "ERROR(HexahedralFEMForceField): object must have a MeshTopology."<<sendl;
    	return;
    }
    else if( _topology->getNbCubes()<=0 )
    {
    	serr << "ERROR(HexahedralFEMForceField): object must have a hexahedric MeshTopology."<<sendl;
    	serr << _topology->getName()<<sendl;
    	serr << _topology->getTypeName()<<sendl;
    	serr<<_topology->getNbPoints()<<sendl;
    	return;
    }

    */

// 	if (!_topology->getCubes().empty())
// 	else
// 	{
    //_indexedElements = & (_topology->getCubes());
// 	}
    //_sparseGrid = dynamic_cast<topology::SparseGridTopology*>(_topology);


    /*
    if (_initialPoints.getValue().size() == 0)
    {
      VecCoord& p = *this->mstate->getX();
      _initialPoints.setValue(p);
    }
    */

    //hexahedronInfo[i].materialMatrix.resize(_indexedElements->size() );
    //_rotations.resize( _indexedElements->size() );
    //_rotatedInitialElements.resize(_topology->getNbHexas()); //_indexedElements->size()
    // _elementStiffnesses.resize(_topology->getNbHexas()); //_indexedElements->size()
    // 	_stiffnesses.resize( _initialPoints.getValue().size()*3 ); // assembly ?

    reinit(); // compute per-element stiffness matrices and other precomputed values


// 	unsigned int i=0;
// 	typename VecElement::const_iterator it;
// 	for(it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i)
// 	{
// 		Element c = *it;
// 		for(int w=0;w<8;++w)
// 		{
// 			serr<<"sparse w : "<<c[w]<<"    "<<_initialPoints.getValue()[c[w]]<<sendl;
// 		}
// 		serr<<"------"<<sendl;
// 	}






}


template <class DataTypes>
void HexahedralFEMForceField<DataTypes>::reinit()
{
    if (f_method.getValue() == "large")
        this->setMethod(LARGE);
    else if (f_method.getValue()  == "polar")
        this->setMethod(POLAR);

    hexahedronInfo.resize(_topology->getNbHexas());

    for (int i=0; i<_topology->getNbHexas(); ++i)
    {
        FHexahedronCreationFunction(i, (void*) this, hexahedronInfo[i],
                _topology->getHexa(i),  (const std::vector< unsigned int > )0,
                (const std::vector< double >)0);
    }

    hexahedronInfo.setCreateFunction(FHexahedronCreationFunction);
    hexahedronInfo.setCreateParameter( (void *) this );
    hexahedronInfo.setDestroyParameter( (void *) this );

}



/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////




template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::addForce (VecDeriv& f, const VecCoord& p, const VecDeriv& /*v*/)
{
    f.resize(p.size());

    switch(method)
    {
    case LARGE :
    {
        for(int i = 0 ; i<_topology->getNbHexas(); ++i)
        {
            accumulateForceLarge( f, p, i);
        }
        break;
    }
    case POLAR :
    {
        for(int i = 0 ; i<_topology->getNbHexas(); ++i)
        {
            accumulateForcePolar( f, p, i);
        }
        break;
    }
    }

}

template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::addDForce (VecDeriv& v, const VecDeriv& x)
{
    if( v.size()!=x.size() ) v.resize(x.size());

    //typename VecElement::const_iterator it;

    //for(it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i)
    for(int i = 0 ; i<_topology->getNbHexas(); ++i)
    {

        Transformation R_0_2;
        R_0_2.transpose(hexahedronInfo[i].rotation);

        Displacement X;

        for(int w=0; w<8; ++w)
        {
            Coord x_2;
            x_2 = R_0_2 * x[_topology->getHexa(i)[w]];
            X[w*3] = x_2[0];
            X[w*3+1] = x_2[1];
            X[w*3+2] = x_2[2];
        }

        Displacement F;
        computeForce( F, X, hexahedronInfo[i].stiffness );//computeForce( F, X, hexahedronInfo[i].stiffness );


        for(int w=0; w<8; ++w)
            v[_topology->getHexa(i)[w]] -= hexahedronInfo[i].rotation * Deriv( F[w*3],  F[w*3+1],  F[w*3+2]  );
    }

}

template <class DataTypes>
double HexahedralFEMForceField<DataTypes>::getPotentialEnergy(const VecCoord&)
{
    serr<<"HexahedralFEMForceField::getPotentialEnergy-not-implemented !!!"<<sendl;
    return 0;
}





/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////






template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::computeElementStiffness( ElementStiffness &K, const MaterialStiffness &M, const Vec<8,Coord> &nodes, const int /* elementIndice */)
{
    Mat33 J_1; // only accurate for orthogonal regular hexa
    J_1.fill( 0.0 );
    Coord l = nodes[6] - nodes[0];
    J_1[0][0]=2.0f / l[0];
    J_1[1][1]=2.0f / l[1];
    J_1[2][2]=2.0f / l[2];


    Real vol = ((nodes[1]-nodes[0]).norm()*(nodes[3]-nodes[0]).norm()*(nodes[4]-nodes[0]).norm());
    vol /= 8.0; // ???

    K.clear();


    for(int i=0; i<8; ++i)
    {
        Mat33 k = vol*integrateStiffness(  _coef[i][0], _coef[i][1],_coef[i][2],  _coef[i][0], _coef[i][1],_coef[i][2], M[0][0], M[0][1],M[2][2], J_1  );


        for(int m=0; m<3; ++m)
            for(int l=0; l<3; ++l)
            {
                K[i*3+m][i*3+l] += k[m][l];
            }



        for(int j=i+1; j<8; ++j)
        {
            Mat33 k = vol*integrateStiffness(  _coef[i][0], _coef[i][1],_coef[i][2],  _coef[j][0], _coef[j][1],_coef[j][2], M[0][0], M[0][1],M[2][2], J_1  );


            for(int m=0; m<3; ++m)
                for(int l=0; l<3; ++l)
                {
                    K[i*3+m][j*3+l] += k[m][l];
// 					K[j*3+l][i*3+m] += k[m][l];
                }
        }
    }

    for(int i=0; i<24; ++i)
        for(int j=i+1; j<24; ++j)
        {
            K[j][i] = K[i][j];
        }


}




template<class DataTypes>
typename HexahedralFEMForceField<DataTypes>::Mat33 HexahedralFEMForceField<DataTypes>::integrateStiffness( int signx0, int signy0, int signz0, int signx1, int signy1, int signz1, const Real u, const Real v, const Real w, const Mat33& J_1  )
{
// 	Real xmax=1,ymax=1,zmax=1,xmin=-1,ymin=-1,zmin=-1;
// 	Real t1 = (Real)(signx0*signy0);
// 	Real t2 = signz0*w;
// 	Real t3 = t2*signx1;
// 	Real t4 = t1*t3;
// 	Real t5 = xmax-xmin;
// 	Real t6 = t5*signy1/2.0f;
// 	Real t7 = ymax-ymin;
// 	Real t8 = t6*t7/2.0f;
// 	Real t9 = zmax-zmin;
// 	Real t10 = signz1*t9/2.0f;
// 	Real t12 = t8*t10*J_1[0][0];
// 	Real t15 = signz0*u;
// 	Real t17 = t1*t15*signx1;
// 	Real t20 = (Real)(signy0*signz0);
// 	Real t23 = 1.0f+signx1*(xmax+xmin)/2.0f;
// 	Real t24 = w*t23;
// 	Real t26 = signy1*t7/2.0f;
// 	Real t27 = t26*t10;
// 	Real t28 = t20*t24*t27;
// 	Real t29 = (Real)(signx0*signz0);
// 	Real t30 = u*signx1;
// 	Real t34 = 1.0f+signy1*(ymax+ymin)/2.0f;
// 	Real t35 = t5*t34/2.0f;
// 	Real t36 = t35*t10;
// 	Real t37 = t29*t30*t36;
// 	Real t44 = 1.0f+signz1*(zmax+zmin)/2.0f;
// 	Real t46 = t6*t7*t44/2.0f;
// 	Real t47 = t1*t30*t46;
// 	Real t53 = t35*t44;
// 	Real t55 = t2*t23;
// 	Real t57 = t34*signz1*t9/2.0f;
// 	Real t58 = t55*t57;
// 	Real t59 = signy0*w;
// 	Real t60 = t59*t23;
// 	Real t61 = t26*t44;
// 	Real t62 = t60*t61;
// 	Real t66 = w*signx1;
// 	Real t67 = t1*t66;
// 	Real t68 = t67*t46;
// 	Real t69 = t29*t66;
// 	Real t70 = t69*t36;
// 	Real t75 = v*t23;
// 	Real t78 = t20*t66;
// 	Real t84 = signx0*v*t23;
// 	Real t104 = v*signx1;
// 	Real t105 = t20*t104;
// 	Real t112 = signy0*v;
// 	Real t115 = signx0*w;
// 	Real t116 = t115*t23;
// 	Real t123 = t8*t10*J_1[1][1];
// 	Real t130 = t20*u*t23*t27;
// 	Real t141 = t115*signx1*t53;
// 	Real t168 = signz0*v;
// 	Real t190 = t8*t10*J_1[2][2];
// 	Mat33 K;
// 	K[0][0] = t4*t12/36.0f+t17*t12/72.0f+(t28+t37)*J_1[0][0]/24.0f+(t47+t28)*J_1[0][0]/
// 			24.0f+(signx0*u*signx1*t53+t58+t62)*J_1[0][0]/8.0f+(t68+t70)*J_1[0][0]/24.0f;
// 	K[0][1] = (t29*t75*t27+t78*t36)*J_1[1][1]/24.0f+(t84*t61+t59*signx1*t53)*J_1[1][1]
// /8.0f;
// 	K[0][2] = (t1*t75*t27+t78*t46)*J_1[2][2]/24.0f+(t84*t57+t3*t53)*J_1[2][2]/8.0f;
// 	K[1][0] = (t105*t36+t29*t24*t27)*J_1[0][0]/24.0f+(t112*signx1*t53+t116*t61)
// 			*J_1[0][0]/8.0f;
// 	K[1][1] = t17*t123/72.0f+t4*t123/36.0f+(t70+t130)*J_1[1][1]/24.0f+(t68+t28)*
// 			J_1[1][1]/24.0f+(signy0*u*t23*t61+t58+t141)*J_1[1][1]/8.0f+(t47+t70)*J_1[1][1]/24.0f;
// 	K[1][2] = (t1*t104*t36+t69*t46)*J_1[2][2]/24.0f+(t112*t23*t57+t55*t61)*J_1[2][2]/
// 			8.0f;
// 	K[2][0] = (t105*t46+t1*t24*t27)*J_1[0][0]/24.0f+(t168*signx1*t53+t116*t57)*
// 			J_1[0][0]/8.0f;
// 	K[2][1] = (t29*t104*t46+t67*t36)*J_1[1][1]/24.0f+(t168*t23*t61+t60*t57)*J_1[1][1]/
// 			8.0f;
// 	K[2][2] = t4*t190/36.0f+(t28+t70)*J_1[2][2]/24.0f+t17*t190/72.0f+(t68+t130)*
// 			J_1[2][2]/24.0f+(t15*t23*t57+t62+t141)*J_1[2][2]/8.0f+(t68+t37)*J_1[2][2]/24.0f;
//
// 	return J_1 * K;

    Mat33 K;

    Real t1 = J_1[0][0]*J_1[0][0];
    Real t2 = t1*signx0;
    Real t3 = (Real)(signy0*signz0);
    Real t4 = t2*t3;
    Real t5 = w*signx1;
    Real t6 = (Real)(signy1*signz1);
    Real t7 = t5*t6;
    Real t10 = t1*signy0;
    Real t12 = w*signy1;
    Real t13 = t12*signz1;
    Real t16 = t2*signz0;
    Real t17 = u*signx1;
    Real t18 = t17*signz1;
    Real t21 = t17*t6;
    Real t24 = t2*signy0;
    Real t25 = t17*signy1;
    Real t28 = t5*signy1;
    Real t32 = w*signz1;
    Real t37 = t5*signz1;
    Real t43 = J_1[0][0]*signx0;
    Real t45 = v*J_1[1][1];
    Real t49 = J_1[0][0]*signy0;
    Real t50 = t49*signz0;
    Real t51 = w*J_1[1][1];
    Real t52 = (Real)(signx1*signz1);
    Real t53 = t51*t52;
    Real t56 = t45*signy1;
    Real t64 = v*J_1[2][2];
    Real t68 = w*J_1[2][2];
    Real t69 = (Real)(signx1*signy1);
    Real t70 = t68*t69;
    Real t73 = t64*signz1;
    Real t81 = J_1[1][1]*signy0;
    Real t83 = v*J_1[0][0];
    Real t87 = J_1[1][1]*signx0;
    Real t88 = t87*signz0;
    Real t89 = w*J_1[0][0];
    Real t90 = t89*t6;
    Real t93 = t83*signx1;
    Real t100 = J_1[1][1]*J_1[1][1];
    Real t101 = t100*signx0;
    Real t102 = t101*t3;
    Real t110 = t100*signy0;
    Real t111 = t110*signz0;
    Real t112 = u*signy1;
    Real t113 = t112*signz1;
    Real t116 = t101*signy0;
    Real t144 = J_1[2][2]*signy0;
    Real t149 = J_1[2][2]*signx0;
    Real t150 = t149*signy0;
    Real t153 = J_1[2][2]*signz0;
    Real t172 = J_1[2][2]*J_1[2][2];
    Real t173 = t172*signx0;
    Real t174 = t173*t3;
    Real t177 = t173*signz0;
    Real t180 = t172*signy0;
    Real t181 = t180*signz0;
    K[0][0] = (float)(t4*t7/36.0+t10*signz0*t13/12.0+t16*t18/24.0+t4*t21/72.0+
            t24*t25/24.0+t24*t28/24.0+t1*signz0*t32/8.0+t10*t12/8.0+t16*t37/24.0+t2*t17/8.0);
    K[0][1] = (float)(t43*signz0*t45*t6/24.0+t50*t53/24.0+t43*t56/8.0+t49*t51*
            signx1/8.0);
    K[0][2] = (float)(t43*signy0*t64*t6/24.0+t50*t70/24.0+t43*t73/8.0+J_1[0][0]*signz0
            *t68*signx1/8.0);
    K[1][0] = (float)(t81*signz0*t83*t52/24.0+t88*t90/24.0+t81*t93/8.0+t87*t89*
            signy1/8.0);
    K[1][1] = (float)(t102*t7/36.0+t102*t21/72.0+t101*signz0*t37/12.0+t111*t113
            /24.0+t116*t28/24.0+t100*signz0*t32/8.0+t111*t13/24.0+t116*t25/24.0+t110*t112/
            8.0+t101*t5/8.0);
    K[1][2] = (float)(t87*signy0*t64*t52/24.0+t88*t70/24.0+t81*t73/8.0+J_1[1][1]*
            signz0*t68*signy1/8.0);
    K[2][0] = (float)(t144*signz0*t83*t69/24.0+t150*t90/24.0+t153*t93/8.0+t149*
            t89*signz1/8.0);
    K[2][1] = (float)(t149*signz0*t45*t69/24.0+t150*t53/24.0+t153*t56/8.0+t144*
            t51*signz1/8.0);
    K[2][2] = (float)(t174*t7/36.0+t177*t37/24.0+t181*t13/24.0+t174*t21/72.0+
            t173*signy0*t28/12.0+t180*t12/8.0+t181*t113/24.0+t177*t18/24.0+t172*signz0*u*
            signz1/8.0+t173*t5/8.0);

    return K /*/(J_1[0][0]*J_1[1][1]*J_1[2][2])*/;

}



template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::computeMaterialStiffness(int i)
{
    //const Real youngModulus = (localStiffnessFactor.empty() ? 1.0f : localStiffnessFactor[i*localStiffnessFactor.size()/_topology->getNbHexas()])*_youngModulus;

    hexahedronInfo[i].materialMatrix[0][0] = hexahedronInfo[i].materialMatrix[1][1] = hexahedronInfo[i].materialMatrix[2][2] = 1;
    hexahedronInfo[i].materialMatrix[0][1] = hexahedronInfo[i].materialMatrix[0][2] = hexahedronInfo[i].materialMatrix[1][0]
            = hexahedronInfo[i].materialMatrix[1][2] = hexahedronInfo[i].materialMatrix[2][0] =
                    hexahedronInfo[i].materialMatrix[2][1] = f_poissonRatio.getValue()/(1-f_poissonRatio.getValue());
    hexahedronInfo[i].materialMatrix[0][3] = hexahedronInfo[i].materialMatrix[0][4] =	hexahedronInfo[i].materialMatrix[0][5] = 0;
    hexahedronInfo[i].materialMatrix[1][3] = hexahedronInfo[i].materialMatrix[1][4] =	hexahedronInfo[i].materialMatrix[1][5] = 0;
    hexahedronInfo[i].materialMatrix[2][3] = hexahedronInfo[i].materialMatrix[2][4] =	hexahedronInfo[i].materialMatrix[2][5] = 0;
    hexahedronInfo[i].materialMatrix[3][0] = hexahedronInfo[i].materialMatrix[3][1] = hexahedronInfo[i].materialMatrix[3][2] = hexahedronInfo[i].materialMatrix[3][4] =	hexahedronInfo[i].materialMatrix[3][5] = 0;
    hexahedronInfo[i].materialMatrix[4][0] = hexahedronInfo[i].materialMatrix[4][1] = hexahedronInfo[i].materialMatrix[4][2] = hexahedronInfo[i].materialMatrix[4][3] =	hexahedronInfo[i].materialMatrix[4][5] = 0;
    hexahedronInfo[i].materialMatrix[5][0] = hexahedronInfo[i].materialMatrix[5][1] = hexahedronInfo[i].materialMatrix[5][2] = hexahedronInfo[i].materialMatrix[5][3] =	hexahedronInfo[i].materialMatrix[5][4] = 0;
    hexahedronInfo[i].materialMatrix[3][3] = hexahedronInfo[i].materialMatrix[4][4] = hexahedronInfo[i].materialMatrix[5][5] = (1-2*f_poissonRatio.getValue())/(2*(1-f_poissonRatio.getValue()));
    hexahedronInfo[i].materialMatrix *= (f_youngModulus.getValue()*(1-f_poissonRatio.getValue()))/((1+f_poissonRatio.getValue())*(1-2*f_poissonRatio.getValue()));
}

template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::computeForce( Displacement &F, const Displacement &Depl, const ElementStiffness &K )
{
    F = K*Depl;
}









/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
////////////// large displacements method


template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::initLarge(int i)
{
    // Rotation matrix (initial Hexahedre/world)
    // first vector on first edge
    // second vector in the plane of the two first edges
    // third vector orthogonal to first and second

    const VecCoord *X0=this->mstate->getX0();

    Vec<8,Coord> nodes;
    for(int w=0; w<8; ++w)
        nodes[w] = (*X0)[_topology->getHexa(i)[w]];


    Coord horizontal;
    horizontal = (nodes[1]-nodes[0] + nodes[2]-nodes[3] + nodes[5]-nodes[4] + nodes[6]-nodes[7])*.25;
    Coord vertical;
    vertical = (nodes[3]-nodes[0] + nodes[2]-nodes[1] + nodes[7]-nodes[4] + nodes[6]-nodes[5])*.25;
    Transformation R_0_1;
    computeRotationLarge( R_0_1, horizontal,vertical);

    for(int w=0; w<8; ++w)
        hexahedronInfo[i].rotatedInitialElements[w] = R_0_1*(*X0)[_topology->getHexa(i)[w]];


    computeElementStiffness( hexahedronInfo[i].stiffness, hexahedronInfo[i].materialMatrix, nodes, i );//computeElementStiffness( hexahedronInfo[i].stiffness, hexahedronInfo[i].materialMatrix, nodes, i );


// 		if(i==0) serr<<hexahedronInfo[i].stiffness<<sendl;
}

template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::computeRotationLarge( Transformation &r, Coord &edgex, Coord &edgey)
{
    // first vector on first edge
    // second vector in the plane of the two first edges
    // third vector orthogonal to first and second
    edgex.normalize();
    edgey.normalize();

    Coord edgez = cross( edgex, edgey );
    edgez.normalize();

    edgey = cross( edgez, edgex );
    edgey.normalize();

    r[0][0] = edgex[0];
    r[0][1] = edgex[1];
    r[0][2] = edgex[2];
    r[1][0] = edgey[0];
    r[1][1] = edgey[1];
    r[1][2] = edgey[2];
    r[2][0] = edgez[0];
    r[2][1] = edgez[1];
    r[2][2] = edgez[2];


// 	r[0][0] = 1;
// 	r[0][1] = 0;
// 	r[0][2] = 0;
// 	r[1][0] = 0;
// 	r[1][1] = 1;
// 	r[1][2] = 0;
// 	r[2][0] = 0;
// 	r[2][1] = 0;
// 	r[2][2] = 1;
}

template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::accumulateForceLarge( Vector& f, const Vector & p, int i)
{
    Vec<8,Coord> nodes;
    for(int w=0; w<8; ++w)
        nodes[w] = p[_topology->getHexa(i)[w]];

    Coord horizontal;
    horizontal = (nodes[1]-nodes[0] + nodes[2]-nodes[3] + nodes[5]-nodes[4] + nodes[6]-nodes[7])*.25;
    Coord vertical;
    vertical = (nodes[3]-nodes[0] + nodes[2]-nodes[1] + nodes[7]-nodes[4] + nodes[6]-nodes[5])*.25;

    Transformation R_0_2; // Rotation matrix (deformed and displaced Hexahedron/world)
    computeRotationLarge( R_0_2, horizontal,vertical);

    hexahedronInfo[i].rotation.transpose(R_0_2);

    // positions of the deformed and displaced Hexahedre in its frame
    Vec<8,Coord> deformed;
    for(int w=0; w<8; ++w)
        deformed[w] = R_0_2 * nodes[w];


    // displacement
    Displacement D;
    for(int k=0 ; k<8 ; ++k )
    {
        int indice = k*3;
        for(int j=0 ; j<3 ; ++j )
            D[indice+j] = hexahedronInfo[i].rotatedInitialElements[k][j] - deformed[k][j];
    }


// 	if(f_updateStiffnessMatrix.getValue())
// 		computeElementStiffness( hexahedronInfo[i].stiffness, hexahedronInfo[i].materialMatrix, deformed );


    Displacement F; //forces
    computeForce( F, D, hexahedronInfo[i].stiffness ); // computeForce( F, D, hexahedronInfo[i].stiffness ); // compute force on element

    for(int w=0; w<8; ++w)
        f[_topology->getHexa(i)[w]] += hexahedronInfo[i].rotation * Deriv( F[w*3],  F[w*3+1],   F[w*3+2]  );
}







/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
////////////// polar decomposition method



template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::initPolar(int i)
{
    const VecCoord *X0=this->mstate->getX0();

    Vec<8,Coord> nodes;
    for(int j=0; j<8; ++j)
        nodes[j] = (*X0)[_topology->getHexa(i)[j]];

    Transformation R_0_1; // Rotation matrix (deformed and displaced Hexahedron/world)
    computeRotationPolar( R_0_1, nodes );


    for(int j=0; j<8; ++j)
    {
        hexahedronInfo[i].rotatedInitialElements[j] = R_0_1 * nodes[j];
    }

    computeElementStiffness( hexahedronInfo[i].stiffness, hexahedronInfo[i].materialMatrix, nodes, i );

}



template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::computeRotationPolar( Transformation &r, Vec<8,Coord> &nodes)
{
    Transformation A;
    Coord Edge =(nodes[1]-nodes[0] + nodes[2]-nodes[3] + nodes[5]-nodes[4] + nodes[6]-nodes[7])*.25;
    A[0][0] = Edge[0];
    A[0][1] = Edge[1];
    A[0][2] = Edge[2];
    Edge = (nodes[3]-nodes[0] + nodes[2]-nodes[1] + nodes[7]-nodes[4] + nodes[6]-nodes[5])*.25;
    A[1][0] = Edge[0];
    A[1][1] = Edge[1];
    A[1][2] = Edge[2];
    Edge = (nodes[4]-nodes[0] + nodes[5]-nodes[1] + nodes[7]-nodes[3] + nodes[6]-nodes[2])*.25;
    A[2][0] = Edge[0];
    A[2][1] = Edge[1];
    A[2][2] = Edge[2];

    Mat33 HT;
    for(int k=0; k<3; ++k)
        for(int j=0; j<3; ++j)
            HT[k][j]=A[k][j];
    //HT[3][0] = HT[3][1] = HT[3][2] = HT[0][3] = HT[1][3] = HT[2][3] = 0;
    //HT[3][3] = 1;
    Mat33 S;

    polar_decomp(HT, r, S);
}


template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::accumulateForcePolar( Vector& f, const Vector & p, int i)
{
    Vec<8,Coord> nodes;
    for(int j=0; j<8; ++j)
        nodes[j] = p[_topology->getHexa(i)[j]];


    Transformation R_0_2; // Rotation matrix (deformed and displaced Hexahedron/world)
    computeRotationPolar( R_0_2, nodes );

    hexahedronInfo[i].rotation.transpose( R_0_2 );


    // positions of the deformed and displaced Hexahedre in its frame
    Vec<8,Coord> deformed;
    for(int j=0; j<8; ++j)
        deformed[j] = R_0_2 * nodes[j];



    // displacement
    Displacement D;
    for(int k=0 ; k<8 ; ++k )
    {
        int indice = k*3;
        for(int j=0 ; j<3 ; ++j )
            D[indice+j] = hexahedronInfo[i].rotatedInitialElements[k][j] - deformed[k][j];
    }

    //forces
    Displacement F;

// 	if(f_updateStiffnessMatrix.getValue())
// 		computeElementStiffness( hexahedronInfo[i].stiffness, hexahedronInfo[i].materialMatrix, deformed );

    // compute force on element
    computeForce( F, D, hexahedronInfo[i].stiffness );


    for(int j=0; j<8; ++j)
        f[_topology->getHexa(i)[j]] += hexahedronInfo[i].rotation * Deriv( F[j*3],  F[j*3+1],   F[j*3+2]  );
}






/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////





template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::draw()
{
// 	serr<<"HexahedralFEMForceField<DataTypes>::draw()"<<sendl;
    if (!getContext()->getShowForceFields()) return;
    if (!this->mstate) return;


    const VecCoord& x = *this->mstate->getX();

    if (getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glDisable(GL_LIGHTING);

    //typename VecElement::const_iterator it;
    //int i;
    //for(it = _indexedElements->begin(), i = 0 ; it != _indexedElements->end() ; ++it, ++i)

    for(int i = 0 ; i<_topology->getNbHexas(); ++i)
    {

        const Hexahedron &t=_topology->getHexa(i);

        Index a = t[0];
        Index b = t[1];
        Index d = t[2];
        Index c = t[3];
        Index e = t[4];
        Index f = t[5];
        Index h = t[6];
        Index g = t[7];

// 		Coord center = (x[a]+x[b]+x[c]+x[d]+x[e]+x[g]+x[f]+x[h])*0.0625;
// 		Real percentage = 0.666667;
// 		Coord pa = (x[a]+center)*percentage;
// 		Coord pb = (x[b]+center)*percentage;
// 		Coord pc = (x[c]+center)*percentage;
// 		Coord pd = (x[d]+center)*percentage;
// 		Coord pe = (x[e]+center)*percentage;
// 		Coord pf = (x[f]+center)*percentage;
// 		Coord pg = (x[g]+center)*percentage;
// 		Coord ph = (x[h]+center)*percentage;

        Coord center = (x[a]+x[b]+x[c]+x[d]+x[e]+x[g]+x[f]+x[h])*0.125;
        Real percentage = (Real) 0.15;
        Coord p0 = x[a]-(x[a]-center)*percentage;
        Coord p1 = x[b]-(x[b]-center)*percentage;
        Coord p2 = x[c]-(x[c]-center)*percentage;
        Coord p3 = x[d]-(x[d]-center)*percentage;
        Coord p4 = x[e]-(x[e]-center)*percentage;
        Coord p5 = x[f]-(x[f]-center)*percentage;
        Coord p6 = x[g]-(x[g]-center)*percentage;
        Coord p7 = x[h]-(x[h]-center)*percentage;


// 		if( _sparseGrid && _sparseGrid->getType(i)==topology::SparseGridTopology::BOUNDARY )
// 			continue;

        glColor4f(0.7f, 0.7f, 0.1f, (1.0f));
        glBegin(GL_QUADS);
        helper::gl::glVertexT(p5);
        helper::gl::glVertexT(p1);
        helper::gl::glVertexT(p3);
        helper::gl::glVertexT(p7);
        //glEnd();
        glColor4f(0.7f, 0, 0, (1.0f));
        //glBegin(GL_POLYGON);
        helper::gl::glVertexT(p1);
        helper::gl::glVertexT(p0);
        helper::gl::glVertexT(p2);
        helper::gl::glVertexT(p3);
        //glEnd();
        glColor4f(0, 0.7f, 0, (1.0f)); // ok
        //glBegin(GL_POLYGON);
        helper::gl::glVertexT(p0);
        helper::gl::glVertexT(p4);
        helper::gl::glVertexT(p6);
        helper::gl::glVertexT(p2);
        //glEnd();
        glColor4f(0, 0, 0.7f, (1.0f));
        //glBegin(GL_POLYGON);
        helper::gl::glVertexT(p4);
        helper::gl::glVertexT(p5);
        helper::gl::glVertexT(p7);
        helper::gl::glVertexT(p6);
        //glEnd();
        glColor4f(0.1f, 0.7f, 0.7f, (1.0f)); // ok
        //glBegin(GL_POLYGON);
        helper::gl::glVertexT(p7);
        helper::gl::glVertexT(p3);
        helper::gl::glVertexT(p2);
        helper::gl::glVertexT(p6);
        //glEnd();
        glColor4f(0.7f, 0.1f, 0.7f, (1.0f));
        //glBegin(GL_POLYGON);
        helper::gl::glVertexT(p1);
        helper::gl::glVertexT(p5);
        helper::gl::glVertexT(p4);
        helper::gl::glVertexT(p0);
        glEnd();
    }

    if (getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
