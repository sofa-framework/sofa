/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_HEXAHEDRALFEMFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_HEXAHEDRALFEMFORCEFIELD_INL

#include "HexahedralFEMForceField.h"
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/decompose.h>
#include <sofa/helper/gl/template.h>
#include <assert.h>
#include <iostream>
#include <set>

#include <SofaBaseTopology/TopologyData.inl>



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

template< class DataTypes>
void HexahedralFEMForceField<DataTypes>::HFFHexahedronHandler::applyCreateFunction(unsigned int hexahedronIndex,
        HexahedronInformation &,
        const core::topology::BaseMeshTopology::Hexahedron &,
        const sofa::helper::vector<unsigned int> &,
        const sofa::helper::vector<double> &)
{
    if (ff)
    {
        switch(ff->method)
        {
        case LARGE :
            ff->initLarge(hexahedronIndex);

            break;
        case POLAR :
            ff->initPolar(hexahedronIndex);
            break;
        }
    }
}

template <class DataTypes>
HexahedralFEMForceField<DataTypes>::HexahedralFEMForceField()
    : f_method(initData(&f_method,std::string("large"),"method","\"large\" or \"polar\" displacements"))
    , f_poissonRatio(initData(&f_poissonRatio,(Real)0.45f,"poissonRatio",""))
    , f_youngModulus(initData(&f_youngModulus,(Real)5000,"youngModulus",""))
    , f_drawing(initData(&f_drawing,true,"drawing"," draw the forcefield if true"))
    , hexahedronInfo(initData(&hexahedronInfo, "hexahedronInfo", "Internal hexahedron data"))
    , hexahedronHandler(NULL)
{

    _coef[0][0]= -1;		_coef[0][1]= -1;		_coef[0][2]= -1;
    _coef[1][0]=  1;		_coef[1][1]= -1;		_coef[1][2]= -1;
    _coef[2][0]=  1;		_coef[2][1]=  1;		_coef[2][2]= -1;
    _coef[3][0]= -1;		_coef[3][1]=  1;		_coef[3][2]= -1;
    _coef[4][0]= -1;		_coef[4][1]= -1;		_coef[4][2]=  1;
    _coef[5][0]=  1;		_coef[5][1]= -1;		_coef[5][2]=  1;
    _coef[6][0]=  1;		_coef[6][1]=  1;		_coef[6][2]=  1;
    _coef[7][0]= -1;		_coef[7][1]=  1;		_coef[7][2]=  1;

    hexahedronHandler = new HFFHexahedronHandler(this,&hexahedronInfo);

    f_poissonRatio.setRequired(true);
    f_youngModulus.setRequired(true);
}

template <class DataTypes>
HexahedralFEMForceField<DataTypes>::~HexahedralFEMForceField()
{
    if(hexahedronHandler) delete hexahedronHandler;
}


template <class DataTypes>
void HexahedralFEMForceField<DataTypes>::init()
{
    this->core::behavior::ForceField<DataTypes>::init();

    this->getContext()->get(_topology);

    if (_topology==NULL)
    {
        serr << "ERROR(HexahedralFEMForceField): object must have a HexahedronSetTopology."<<sendl;
        return;
    }

    this->reinit(); // compute per-element stiffness matrices and other precomputed values
}


template <class DataTypes>
void HexahedralFEMForceField<DataTypes>::reinit()
{
    if (f_method.getValue() == "large")
        this->setMethod(LARGE);
    else if (f_method.getValue()  == "polar")
        this->setMethod(POLAR);

    helper::vector<typename HexahedralFEMForceField<DataTypes>::HexahedronInformation>& hexahedronInf = *(hexahedronInfo.beginEdit());


    hexahedronInf.resize(_topology->getNbHexahedra());

    for (int i=0; i<_topology->getNbHexahedra(); ++i)
    {
        hexahedronHandler->applyCreateFunction(i,hexahedronInf[i],
                _topology->getHexahedron(i),  (const std::vector< unsigned int > )0,
                (const std::vector< double >)0);
    }
    hexahedronInfo.createTopologicalEngine(_topology,hexahedronHandler);
    hexahedronInfo.registerTopologicalData();
    hexahedronInfo.endEdit();
}


template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::addForce (const core::MechanicalParams* /*mparams*/, DataVecDeriv& f, const DataVecCoord& p, const DataVecDeriv& /*v*/)
{
    WDataRefVecDeriv _f = f;
    RDataRefVecCoord _p = p;

    _f.resize(_p.size());

    switch(method)
    {
    case LARGE :
    {
        for(int i = 0 ; i<_topology->getNbHexahedra(); ++i)
        {
            accumulateForceLarge( _f, _p, i);
        }
        break;
    }
    case POLAR :
    {
        for(int i = 0 ; i<_topology->getNbHexahedra(); ++i)
        {
            accumulateForcePolar( _f, _p, i);
        }
        break;
    }
    }

}

template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::addDForce (const core::MechanicalParams *mparams, DataVecDeriv& v, const DataVecDeriv& x)
{
    helper::WriteAccessor< DataVecDeriv > _v = v;
    const VecCoord& _x = x.getValue();
    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    if( _v.size()!=_x.size() ) _v.resize(_x.size());

    const helper::vector<typename HexahedralFEMForceField<DataTypes>::HexahedronInformation>& hexahedronInf = hexahedronInfo.getValue();

    for(int i = 0 ; i<_topology->getNbHexahedra(); ++i)
    {
        Transformation R_0_2;
        R_0_2.transpose(hexahedronInf[i].rotation);

        Displacement X;

        for(int w=0; w<8; ++w)
        {
            Coord x_2;
            x_2 = R_0_2 * _x[_topology->getHexahedron(i)[w]] * kFactor;
            X[w*3] = x_2[0];
            X[w*3+1] = x_2[1];
            X[w*3+2] = x_2[2];
        }

        Displacement F;
        computeForce( F, X, hexahedronInf[i].stiffness );//computeForce( F, X, hexahedronInfo[i].stiffness );

        for(int w=0; w<8; ++w)
            _v[_topology->getHexahedron(i)[w]] -= hexahedronInf[i].rotation * Deriv(F[w*3], F[w*3+1], F[w*3+2]);
    }
}

template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::computeElementStiffness( ElementStiffness &K, const MaterialStiffness &M, const defaulttype::Vec<8,Coord> &nodes)
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
        Mat33 k = vol*integrateStiffness(  _coef[i][0], _coef[i][1],_coef[i][2],  _coef[i][0], _coef[i][1],_coef[i][2], M[0][0], M[0][1],M[3][3], J_1  );

        for(int m=0; m<3; ++m)
        {
            for(int l=0; l<3; ++l)
            {
                K[i*3+m][i*3+l] += k[m][l];
            }
        }

        for(int j=i+1; j<8; ++j)
        {
            Mat33 k = vol*integrateStiffness(  _coef[i][0], _coef[i][1],_coef[i][2],  _coef[j][0], _coef[j][1],_coef[j][2], M[0][0], M[0][1],M[3][3], J_1  );

            for(int m=0; m<3; ++m)
                for(int l=0; l<3; ++l)
                {
                    K[i*3+m][j*3+l] += k[m][l];
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
void HexahedralFEMForceField<DataTypes>::computeMaterialStiffness(MaterialStiffness &m, double youngModulus, double poissonRatio)
{
    m[0][0] = m[1][1] = m[2][2] = 1;
    m[0][1] = m[0][2] = m[1][0]= m[1][2] = m[2][0] =  m[2][1] = (Real)(poissonRatio/(1-poissonRatio));
    m[0][3] = m[0][4] =	m[0][5] = 0;
    m[1][3] = m[1][4] =	m[1][5] = 0;
    m[2][3] = m[2][4] =	m[2][5] = 0;
    m[3][0] = m[3][1] = m[3][2] = m[3][4] =	m[3][5] = 0;
    m[4][0] = m[4][1] = m[4][2] = m[4][3] =	m[4][5] = 0;
    m[5][0] = m[5][1] = m[5][2] = m[5][3] =	m[5][4] = 0;
    m[3][3] = m[4][4] = m[5][5] = (Real)((1-2*poissonRatio)/(2*(1-poissonRatio)));
    m *= (Real)((youngModulus*(1-poissonRatio))/((1+poissonRatio)*(1-2*poissonRatio)));
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
void HexahedralFEMForceField<DataTypes>::initLarge(const int i)
{
    // Rotation matrix (initial Hexahedre/world)
    // first vector on first edge
    // second vector in the plane of the two first edges
    // third vector orthogonal to first and second

    const VecCoord& X0=this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    defaulttype::Vec<8,Coord> nodes;
    for(int w=0; w<8; ++w)
        nodes[w] = (X0)[_topology->getHexahedron(i)[w]];


    Coord horizontal;
    horizontal = (nodes[1]-nodes[0] + nodes[2]-nodes[3] + nodes[5]-nodes[4] + nodes[6]-nodes[7])*.25;
    Coord vertical;
    vertical = (nodes[3]-nodes[0] + nodes[2]-nodes[1] + nodes[7]-nodes[4] + nodes[6]-nodes[5])*.25;
    Transformation R_0_1;
    computeRotationLarge( R_0_1, horizontal,vertical);


    helper::vector<typename HexahedralFEMForceField<DataTypes>::HexahedronInformation>& hexahedronInf = *(hexahedronInfo.beginEdit());

    for(int w=0; w<8; ++w)
        hexahedronInf[i].rotatedInitialElements[w] = R_0_1*nodes[w];

    computeMaterialStiffness( hexahedronInf[i].materialMatrix, f_youngModulus.getValue(), f_poissonRatio.getValue() );
    computeElementStiffness( hexahedronInf[i].stiffness, hexahedronInf[i].materialMatrix, nodes);

    hexahedronInfo.endEdit();
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
}

template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::accumulateForceLarge( WDataRefVecDeriv& f, RDataRefVecCoord & p, const int i)
{
    defaulttype::Vec<8,Coord> nodes;
    for(int w=0; w<8; ++w)
        nodes[w] = p[_topology->getHexahedron(i)[w]];

    Coord horizontal;
    horizontal = (nodes[1]-nodes[0] + nodes[2]-nodes[3] + nodes[5]-nodes[4] + nodes[6]-nodes[7])*.25;
    Coord vertical;
    vertical = (nodes[3]-nodes[0] + nodes[2]-nodes[1] + nodes[7]-nodes[4] + nodes[6]-nodes[5])*.25;

    Transformation R_0_2; // Rotation matrix (deformed and displaced Hexahedron/world)
    computeRotationLarge( R_0_2, horizontal,vertical);

    helper::vector<typename HexahedralFEMForceField<DataTypes>::HexahedronInformation>& hexahedronInf = *(hexahedronInfo.beginEdit());

    hexahedronInf[i].rotation.transpose(R_0_2);

    // positions of the deformed and displaced Hexahedre in its frame
    defaulttype::Vec<8,Coord> deformed;
    for(int w=0; w<8; ++w)
        deformed[w] = R_0_2 * nodes[w];


    // displacement
    Displacement D;
    for(int k=0 ; k<8 ; ++k )
    {
        int indice = k*3;
        for(int j=0 ; j<3 ; ++j )
            D[indice+j] = hexahedronInf[i].rotatedInitialElements[k][j] - deformed[k][j];
    }

    Displacement F; //forces
    computeForce( F, D, hexahedronInf[i].stiffness ); // computeForce( F, D, hexahedronInf[i].stiffness ); // compute force on element

    for(int w=0; w<8; ++w)
        f[_topology->getHexahedron(i)[w]] += hexahedronInf[i].rotation * Deriv( F[w*3],  F[w*3+1],   F[w*3+2]  );

    hexahedronInfo.endEdit();
}



/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
////////////// polar decomposition method


template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::initPolar(const int i)
{
    const VecCoord& X0=this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    defaulttype::Vec<8,Coord> nodes;
    for(int j=0; j<8; ++j)
        nodes[j] = (X0)[_topology->getHexahedron(i)[j]];

    Transformation R_0_1; // Rotation matrix (deformed and displaced Hexahedron/world)
    computeRotationPolar( R_0_1, nodes );


    helper::vector<typename HexahedralFEMForceField<DataTypes>::HexahedronInformation>& hexahedronInf = *(hexahedronInfo.beginEdit());

    for(int j=0; j<8; ++j)
    {
        hexahedronInf[i].rotatedInitialElements[j] = R_0_1 * nodes[j];
    }

    computeMaterialStiffness( hexahedronInf[i].materialMatrix, f_youngModulus.getValue(), f_poissonRatio.getValue() );
    computeElementStiffness( hexahedronInf[i].stiffness, hexahedronInf[i].materialMatrix, nodes );

    hexahedronInfo.endEdit();
}


template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::computeRotationPolar( Transformation &r, defaulttype::Vec<8,Coord> &nodes)
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

    helper::Decompose<Real>::polarDecomposition(HT, r);
}


template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::accumulateForcePolar(WDataRefVecDeriv& f, RDataRefVecCoord & p, const int i)
{
    defaulttype::Vec<8,Coord> nodes;
    for(int j=0; j<8; ++j)
        nodes[j] = p[_topology->getHexahedron(i)[j]];


    Transformation R_0_2; // Rotation matrix (deformed and displaced Hexahedron/world)
    computeRotationPolar( R_0_2, nodes );

    helper::vector<typename HexahedralFEMForceField<DataTypes>::HexahedronInformation>& hexahedronInf = *(hexahedronInfo.beginEdit());

    hexahedronInf[i].rotation.transpose( R_0_2 );

    // positions of the deformed and displaced Hexahedre in its frame
    defaulttype::Vec<8,Coord> deformed;
    for(int j=0; j<8; ++j)
        deformed[j] = R_0_2 * nodes[j];



    // displacement
    Displacement D;
    for(int k=0 ; k<8 ; ++k )
    {
        int indice = k*3;
        for(int j=0 ; j<3 ; ++j )
            D[indice+j] = hexahedronInf[i].rotatedInitialElements[k][j] - deformed[k][j];
    }

    //forces
    Displacement F;

    // compute force on element
    computeForce( F, D, hexahedronInf[i].stiffness );

    for(int j=0; j<8; ++j)
        f[_topology->getHexahedron(i)[j]] += hexahedronInf[i].rotation * Deriv( F[j*3],  F[j*3+1],   F[j*3+2]  );

    hexahedronInfo.endEdit();
}


/////////////////////////////////////////////////
/////////////////////////////////////////////////

template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    // Build Matrix Block for this ForceField
    int i,j,n1, n2, e;

    Index node1, node2;

    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
    const Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());
    const helper::vector<typename HexahedralFEMForceField<DataTypes>::HexahedronInformation>& hexahedronInf = hexahedronInfo.getValue();

    for(e=0 ; e<_topology->getNbHexahedra() ; ++e)
    {
        const ElementStiffness &Ke = hexahedronInf[e].stiffness;

        // find index of node 1
        for (n1=0; n1<8; n1++)
        {
            node1 = _topology->getHexahedron(e)[n1];

            // find index of node 2
            for (n2=0; n2<8; n2++)
            {
                node2 = _topology->getHexahedron(e)[n2];
                Mat33 tmp = hexahedronInf[e].rotation.multTranspose( Mat33(Coord(Ke[3*n1+0][3*n2+0],Ke[3*n1+0][3*n2+1],Ke[3*n1+0][3*n2+2]),
                        Coord(Ke[3*n1+1][3*n2+0],Ke[3*n1+1][3*n2+1],Ke[3*n1+1][3*n2+2]),
                        Coord(Ke[3*n1+2][3*n2+0],Ke[3*n1+2][3*n2+1],Ke[3*n1+2][3*n2+2])) ) * hexahedronInf[e].rotation;
                for(i=0; i<3; i++)
                    for (j=0; j<3; j++)
                        r.matrix->add(r.offset+3*node1+i, r.offset+3*node2+j, - tmp[i][j]*kFactor);
            }
        }
    }
}




template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!this->mstate) return;
    if (!f_drawing.getValue()) return;

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glDisable(GL_LIGHTING);

    for(int i = 0 ; i<_topology->getNbHexahedra(); ++i)
    {
        const core::topology::BaseMeshTopology::Hexahedron &t=_topology->getHexahedron(i);

        Index a = t[0];
        Index b = t[1];
        Index d = t[2];
        Index c = t[3];
        Index e = t[4];
        Index f = t[5];
        Index h = t[6];
        Index g = t[7];

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

    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

#endif /* SOFA_NO_OPENGL */
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_HEXAHEDRALFEMFORCEFIELD_INL
