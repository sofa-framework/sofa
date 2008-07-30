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
#ifndef SOFA_COMPONENT_FORCEFIELD_TETRAHEDRONFEMFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_TETRAHEDRONFEMFORCEFIELD_INL

#include <sofa/core/componentmodel/behavior/ForceField.inl>
#include <sofa/component/forcefield/TetrahedralCorotationalFEMForceField.h>
#include <sofa/component/topology/GridTopology.h>
#include <sofa/helper/PolarDecompose.h>
#include <sofa/helper/gl/template.h>
#include <sofa/component/topology/TetrahedronData.inl>
#include <assert.h>
#include <iostream>
#include <set>

using std::cerr;
using std::endl;
using std::set;


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
void TetrahedralCorotationalFEMForceField<DataTypes>::CFTetrahedronCreationFunction (int tetrahedronIndex, void* param,
        TetrahedronInformation &,
        const Tetrahedron& ,
        const helper::vector< unsigned int > &,
        const helper::vector< double >&)
{
    TetrahedralCorotationalFEMForceField<DataTypes> *ff= (TetrahedralCorotationalFEMForceField<DataTypes> *)param;
    if (ff)
    {

        const Tetrahedron &t=ff->_topology->getTetra(tetrahedronIndex);
        Index a = t[0];
        Index b = t[1];
        Index c = t[2];
        Index d = t[3];

        switch(ff->method)
        {
        case SMALL :
            ff->computeMaterialStiffness(tetrahedronIndex,a,b,c,d);
            ff->initSmall(tetrahedronIndex,a,b,c,d);
            break;
        case LARGE :
            ff->computeMaterialStiffness(tetrahedronIndex,a,b,c,d);
            ff->initLarge(tetrahedronIndex,a,b,c,d);

            break;
        case POLAR :
            ff->computeMaterialStiffness(tetrahedronIndex,a,b,c,d);
            ff->initPolar(tetrahedronIndex,a,b,c,d);
            break;
        }
    }
}

template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::parse(core::objectmodel::BaseObjectDescription* arg)
{
    this->core::componentmodel::behavior::ForceField<DataTypes>::parse(arg);

    this->setComputeGlobalMatrix(std::string(arg->getAttribute("computeGlobalMatrix","false"))=="true");
}

template <class DataTypes> void TetrahedralCorotationalFEMForceField<DataTypes>::handleTopologyChange()
{
    std::list<const TopologyChange *>::const_iterator itBegin=_topology->firstChange();
    std::list<const TopologyChange *>::const_iterator itEnd=_topology->lastChange();


    tetrahedronInfo.handleTopologyEvents(itBegin,itEnd);

}

template <class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::init()
{
    f_poissonRatio.beginEdit();
    f_youngModulus.beginEdit();
    f_localStiffnessFactor.beginEdit();
    f_updateStiffnessMatrix.beginEdit();
    f_assembling.beginEdit();


    this->core::componentmodel::behavior::ForceField<DataTypes>::init();

    _topology = getContext()->getMeshTopology();

    if (_topology->getNbTetras()==0)
    {
        std::cerr << "ERROR(TetrahedralCorotationalFEMForceField): object must have a Tetrahedral Set Topology.\n";
        return;
    }

    reinit(); // compute per-element stiffness matrices and other precomputed values

}


template <class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::reinit()
{

    if (f_method.getValue() == "small")
        this->setMethod(SMALL);
    else if (f_method.getValue() == "polar")
        this->setMethod(POLAR);
    else this->setMethod(LARGE);

    tetrahedronInfo.resize(_topology->getNbTetras());

    for (int i=0; i<_topology->getNbTetras(); ++i)
    {
        CFTetrahedronCreationFunction(i, (void*) this, tetrahedronInfo[i],
                _topology->getTetra(i),  (const std::vector< unsigned int > )0,
                (const std::vector< double >)0);
    }

    tetrahedronInfo.setCreateFunction(CFTetrahedronCreationFunction);
    tetrahedronInfo.setCreateParameter( (void *) this );
    tetrahedronInfo.setDestroyParameter( (void *) this );

}


template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::addForce (VecDeriv& f, const VecCoord& p, const VecDeriv& /*v*/)
{

    switch(method)
    {
    case SMALL :
    {
        for(int i = 0 ; i<_topology->getNbTetras(); ++i)
        {
            accumulateForceSmall( f, p, i );
        }
        break;
    }
    case LARGE :
    {
        for(int i = 0 ; i<_topology->getNbTetras(); ++i)
        {
            accumulateForceLarge( f, p, i );
        }
        break;
    }
    case POLAR :
    {
        for(int i = 0 ; i<_topology->getNbTetras(); ++i)
        {
            accumulateForcePolar( f, p, i );
        }
        break;
    }
    }
}

template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::addDForce (VecDeriv& v, const VecDeriv& x)
{
    switch(method)
    {
    case SMALL :
    {
        for(int i = 0 ; i<_topology->getNbTetras(); ++i)
        {
            const Tetrahedron &t=_topology->getTetra(i);
            Index a = t[0];
            Index b = t[1];
            Index c = t[2];
            Index d = t[3];

            applyStiffnessSmall( v,x, i, a,b,c,d );
        }
        break;
    }
    case LARGE :
    {
        for(int i = 0 ; i<_topology->getNbTetras(); ++i)
        {
            const Tetrahedron &t=_topology->getTetra(i);
            Index a = t[0];
            Index b = t[1];
            Index c = t[2];
            Index d = t[3];

            applyStiffnessLarge( v,x, i, a,b,c,d );
        }
        break;
    }
    case POLAR :
    {
        for(int i = 0 ; i<_topology->getNbTetras(); ++i)
        {
            const Tetrahedron &t=_topology->getTetra(i);
            Index a = t[0];
            Index b = t[1];
            Index c = t[2];
            Index d = t[3];

            applyStiffnessPolar( v,x, i, a,b,c,d );
        }
        break;
    }
    }
}

template <class DataTypes>
double TetrahedralCorotationalFEMForceField<DataTypes>::getPotentialEnergy(const VecCoord&)
{
    cerr<<"TetrahedralCorotationalFEMForceField::getPotentialEnergy-not-implemented !!!"<<endl;
    return 0;
}

template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::computeStrainDisplacement( StrainDisplacement &J, Coord a, Coord b, Coord c, Coord d )
{
    // shape functions matrix
    Mat<2, 3, Real> M;

    M[0][0] = b[1];
    M[0][1] = c[1];
    M[0][2] = d[1];
    M[1][0] = b[2];
    M[1][1] = c[2];
    M[1][2] = d[2];
    J[0][0] = J[1][3] = J[2][5]   = - peudo_determinant_for_coef( M );
    M[0][0] = b[0];
    M[0][1] = c[0];
    M[0][2] = d[0];
    J[0][3] = J[1][1] = J[2][4]   = peudo_determinant_for_coef( M );
    M[1][0] = b[1];
    M[1][1] = c[1];
    M[1][2] = d[1];
    J[0][5] = J[1][4] = J[2][2]   = - peudo_determinant_for_coef( M );

    M[0][0] = c[1];
    M[0][1] = d[1];
    M[0][2] = a[1];
    M[1][0] = c[2];
    M[1][1] = d[2];
    M[1][2] = a[2];
    J[3][0] = J[4][3] = J[5][5]   = peudo_determinant_for_coef( M );
    M[0][0] = c[0];
    M[0][1] = d[0];
    M[0][2] = a[0];
    J[3][3] = J[4][1] = J[5][4]   = - peudo_determinant_for_coef( M );
    M[1][0] = c[1];
    M[1][1] = d[1];
    M[1][2] = a[1];
    J[3][5] = J[4][4] = J[5][2]   = peudo_determinant_for_coef( M );

    M[0][0] = d[1];
    M[0][1] = a[1];
    M[0][2] = b[1];
    M[1][0] = d[2];
    M[1][1] = a[2];
    M[1][2] = b[2];
    J[6][0] = J[7][3] = J[8][5]   = - peudo_determinant_for_coef( M );
    M[0][0] = d[0];
    M[0][1] = a[0];
    M[0][2] = b[0];
    J[6][3] = J[7][1] = J[8][4]   = peudo_determinant_for_coef( M );
    M[1][0] = d[1];
    M[1][1] = a[1];
    M[1][2] = b[1];
    J[6][5] = J[7][4] = J[8][2]   = - peudo_determinant_for_coef( M );

    M[0][0] = a[1];
    M[0][1] = b[1];
    M[0][2] = c[1];
    M[1][0] = a[2];
    M[1][1] = b[2];
    M[1][2] = c[2];
    J[9][0] = J[10][3] = J[11][5]   = peudo_determinant_for_coef( M );
    M[0][0] = a[0];
    M[0][1] = b[0];
    M[0][2] = c[0];
    J[9][3] = J[10][1] = J[11][4]   = - peudo_determinant_for_coef( M );
    M[1][0] = a[1];
    M[1][1] = b[1];
    M[1][2] = c[1];
    J[9][5] = J[10][4] = J[11][2]   = peudo_determinant_for_coef( M );


    // 0
    J[0][1] = J[0][2] = J[0][4] = J[1][0] =  J[1][2] =  J[1][5] =  J[2][0] =  J[2][1] =  J[2][3]  = 0;
    J[3][1] = J[3][2] = J[3][4] = J[4][0] =  J[4][2] =  J[4][5] =  J[5][0] =  J[5][1] =  J[5][3]  = 0;
    J[6][1] = J[6][2] = J[6][4] = J[7][0] =  J[7][2] =  J[7][5] =  J[8][0] =  J[8][1] =  J[8][3]  = 0;
    J[9][1] = J[9][2] = J[9][4] = J[10][0] = J[10][2] = J[10][5] = J[11][0] = J[11][1] = J[11][3] = 0;

    //m_deq( J, 1.2 ); //hack for stability ??
}

template<class DataTypes>
typename TetrahedralCorotationalFEMForceField<DataTypes>::Real TetrahedralCorotationalFEMForceField<DataTypes>::peudo_determinant_for_coef ( const Mat<2, 3, Real>&  M )
{
    return  M[0][1]*M[1][2] - M[1][1]*M[0][2] -  M[0][0]*M[1][2] + M[1][0]*M[0][2] + M[0][0]*M[1][1] - M[1][0]*M[0][1];
}

template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::computeStiffnessMatrix( StiffnessMatrix& S,StiffnessMatrix& SR,const MaterialStiffness &K, const StrainDisplacement &J, const Transformation& Rot )
{
    MatNoInit<6, 12, Real> Jt;
    Jt.transpose( J );

    MatNoInit<12, 12, Real> JKJt;
    JKJt = J*K*Jt;

    MatNoInit<12, 12, Real> RR,RRt;
    RR.clear();
    RRt.clear();
    for(int i=0; i<3; ++i)
        for(int j=0; j<3; ++j)
        {
            RR[i][j]=RR[i+3][j+3]=RR[i+6][j+6]=RR[i+9][j+9]=Rot[i][j];
            RRt[i][j]=RRt[i+3][j+3]=RRt[i+6][j+6]=RRt[i+9][j+9]=Rot[j][i];
        }

    S = RR*JKJt;
    SR = S*RRt;
}

template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::computeMaterialStiffness(int i, Index&a, Index&b, Index&c, Index&d)
{

    const VecReal& localStiffnessFactor = _localStiffnessFactor;
    const Real youngModulus = (localStiffnessFactor.empty() ? 1.0f : localStiffnessFactor[i*localStiffnessFactor.size()/_topology->getNbTetras()])*_youngModulus;
    const Real poissonRatio = _poissonRatio;

    tetrahedronInfo[i].materialMatrix[0][0] = tetrahedronInfo[i].materialMatrix[1][1] = tetrahedronInfo[i].materialMatrix[2][2] = 1;
    tetrahedronInfo[i].materialMatrix[0][1] = tetrahedronInfo[i].materialMatrix[0][2] = tetrahedronInfo[i].materialMatrix[1][0]
            = tetrahedronInfo[i].materialMatrix[1][2] = tetrahedronInfo[i].materialMatrix[2][0] =
                    tetrahedronInfo[i].materialMatrix[2][1] = poissonRatio/(1-poissonRatio);
    tetrahedronInfo[i].materialMatrix[0][3] = tetrahedronInfo[i].materialMatrix[0][4] = tetrahedronInfo[i].materialMatrix[0][5] = 0;
    tetrahedronInfo[i].materialMatrix[1][3] = tetrahedronInfo[i].materialMatrix[1][4] = tetrahedronInfo[i].materialMatrix[1][5] = 0;
    tetrahedronInfo[i].materialMatrix[2][3] = tetrahedronInfo[i].materialMatrix[2][4] = tetrahedronInfo[i].materialMatrix[2][5] = 0;
    tetrahedronInfo[i].materialMatrix[3][0] = tetrahedronInfo[i].materialMatrix[3][1] = tetrahedronInfo[i].materialMatrix[3][2] = tetrahedronInfo[i].materialMatrix[3][4] = tetrahedronInfo[i].materialMatrix[3][5] = 0;
    tetrahedronInfo[i].materialMatrix[4][0] = tetrahedronInfo[i].materialMatrix[4][1] = tetrahedronInfo[i].materialMatrix[4][2] = tetrahedronInfo[i].materialMatrix[4][3] = tetrahedronInfo[i].materialMatrix[4][5] = 0;
    tetrahedronInfo[i].materialMatrix[5][0] = tetrahedronInfo[i].materialMatrix[5][1] = tetrahedronInfo[i].materialMatrix[5][2] = tetrahedronInfo[i].materialMatrix[5][3] = tetrahedronInfo[i].materialMatrix[5][4] = 0;
    tetrahedronInfo[i].materialMatrix[3][3] = tetrahedronInfo[i].materialMatrix[4][4] = tetrahedronInfo[i].materialMatrix[5][5] = (1-2*poissonRatio)/(2*(1-poissonRatio));
    tetrahedronInfo[i].materialMatrix *= (youngModulus*(1-poissonRatio))/((1+poissonRatio)*(1-2*poissonRatio));

    /*Real gamma = (youngModulus*poissonRatio) / ((1+poissonRatio)*(1-2*poissonRatio));
    Real 		mu2 = youngModulus / (1+poissonRatio);
    _materialsStiffnesses[i][0][3] = _materialsStiffnesses[i][0][4] =	_materialsStiffnesses[i][0][5] = 0;
    _materialsStiffnesses[i][1][3] = _materialsStiffnesses[i][1][4] =	_materialsStiffnesses[i][1][5] = 0;
    _materialsStiffnesses[i][2][3] = _materialsStiffnesses[i][2][4] =	_materialsStiffnesses[i][2][5] = 0;
    _materialsStiffnesses[i][3][0] = _materialsStiffnesses[i][3][1] = _materialsStiffnesses[i][3][2] = _materialsStiffnesses[i][3][4] =	_materialsStiffnesses[i][3][5] = 0;
    _materialsStiffnesses[i][4][0] = _materialsStiffnesses[i][4][1] = _materialsStiffnesses[i][4][2] = _materialsStiffnesses[i][4][3] =	_materialsStiffnesses[i][4][5] = 0;
    _materialsStiffnesses[i][5][0] = _materialsStiffnesses[i][5][1] = _materialsStiffnesses[i][5][2] = _materialsStiffnesses[i][5][3] =	_materialsStiffnesses[i][5][4] = 0;
    _materialsStiffnesses[i][0][0] = _materialsStiffnesses[i][1][1] = _materialsStiffnesses[i][2][2] = gamma+mu2;
    _materialsStiffnesses[i][0][1] = _materialsStiffnesses[i][0][2] = _materialsStiffnesses[i][1][0]
    			= _materialsStiffnesses[i][1][2] = _materialsStiffnesses[i][2][0] = _materialsStiffnesses[i][2][1] = gamma;
    _materialsStiffnesses[i][3][3] = _materialsStiffnesses[i][4][4] = _materialsStiffnesses[i][5][5] =	mu2;*/

    // divide by 36 times volumes of the element
    const VecCoord *X0=this->mstate->getX0();

    Coord A = (*X0)[b] - (*X0)[a];
    Coord B = (*X0)[c] - (*X0)[a];
    Coord C = (*X0)[d] - (*X0)[a];
    Coord AB = cross(A, B);
    Real volumes6 = fabs( dot( AB, C ) );
    if (volumes6<0)
    {
        std::cerr << "ERROR: Negative volume for tetra "<<i<<" <"<<a<<','<<b<<','<<c<<','<<d<<"> = "<<volumes6/6<<std::endl;
    }
    tetrahedronInfo[i].materialMatrix  /= volumes6;
}

template<class DataTypes>
inline void TetrahedralCorotationalFEMForceField<DataTypes>::computeForce( Displacement &F, const Displacement &Depl, const MaterialStiffness &K, const StrainDisplacement &J )
{
#if 0
    F = J*(K*(J.multTranspose(Depl)));

#else
    /* We have these zeros
                                  K[0][3]   K[0][4]   K[0][5]
                                  K[1][3]   K[1][4]   K[1][5]
                                  K[2][3]   K[2][4]   K[2][5]
    K[3][0]   K[3][1]   K[3][2]             K[3][4]   K[3][5]
    K[4][0]   K[4][1]   K[4][2]   K[4][3]             K[4][5]
    K[5][0]   K[5][1]   K[5][2]   K[5][3]   K[5][4]


              J[0][1]   J[0][2]             J[0][4]
    J[1][0]             J[1][2]                       J[1][5]
    J[2][0]   J[2][1]             J[2][3]
              J[3][1]   J[3][2]             J[3][4]
    J[4][0]             J[4][2]                       J[4][5]
    J[5][0]   J[5][1]             J[5][3]
              J[6][1]   J[6][2]             J[6][4]
    J[7][0]             J[7][2]                       J[7][5]
    J[8][0]   J[8][1]             J[8][3]
              J[9][1]   J[9][2]             J[9][4]
    J[10][0]            J[10][2]                      J[10][5]
    J[11][0]  J[11][1]            J[11][3]
    */

    VecNoInit<6,Real> JtD;
    JtD[0] =   J[ 0][0]*Depl[ 0]+/*J[ 1][0]*Depl[ 1]+  J[ 2][0]*Depl[ 2]+*/
            J[ 3][0]*Depl[ 3]+/*J[ 4][0]*Depl[ 4]+  J[ 5][0]*Depl[ 5]+*/
            J[ 6][0]*Depl[ 6]+/*J[ 7][0]*Depl[ 7]+  J[ 8][0]*Depl[ 8]+*/
            J[ 9][0]*Depl[ 9] /*J[10][0]*Depl[10]+  J[11][0]*Depl[11]*/;
    JtD[1] = /*J[ 0][1]*Depl[ 0]+*/J[ 1][1]*Depl[ 1]+/*J[ 2][1]*Depl[ 2]+*/
            /*J[ 3][1]*Depl[ 3]+*/J[ 4][1]*Depl[ 4]+/*J[ 5][1]*Depl[ 5]+*/
            /*J[ 6][1]*Depl[ 6]+*/J[ 7][1]*Depl[ 7]+/*J[ 8][1]*Depl[ 8]+*/
            /*J[ 9][1]*Depl[ 9]+*/J[10][1]*Depl[10] /*J[11][1]*Depl[11]*/;
    JtD[2] = /*J[ 0][2]*Depl[ 0]+  J[ 1][2]*Depl[ 1]+*/J[ 2][2]*Depl[ 2]+
            /*J[ 3][2]*Depl[ 3]+  J[ 4][2]*Depl[ 4]+*/J[ 5][2]*Depl[ 5]+
            /*J[ 6][2]*Depl[ 6]+  J[ 7][2]*Depl[ 7]+*/J[ 8][2]*Depl[ 8]+
            /*J[ 9][2]*Depl[ 9]+  J[10][2]*Depl[10]+*/J[11][2]*Depl[11]  ;
    JtD[3] =   J[ 0][3]*Depl[ 0]+  J[ 1][3]*Depl[ 1]+/*J[ 2][3]*Depl[ 2]+*/
            J[ 3][3]*Depl[ 3]+  J[ 4][3]*Depl[ 4]+/*J[ 5][3]*Depl[ 5]+*/
            J[ 6][3]*Depl[ 6]+  J[ 7][3]*Depl[ 7]+/*J[ 8][3]*Depl[ 8]+*/
            J[ 9][3]*Depl[ 9]+  J[10][3]*Depl[10] /*J[11][3]*Depl[11]*/;
    JtD[4] = /*J[ 0][4]*Depl[ 0]+*/J[ 1][4]*Depl[ 1]+  J[ 2][4]*Depl[ 2]+
            /*J[ 3][4]*Depl[ 3]+*/J[ 4][4]*Depl[ 4]+  J[ 5][4]*Depl[ 5]+
            /*J[ 6][4]*Depl[ 6]+*/J[ 7][4]*Depl[ 7]+  J[ 8][4]*Depl[ 8]+
            /*J[ 9][4]*Depl[ 9]+*/J[10][4]*Depl[10]+  J[11][4]*Depl[11]  ;
    JtD[5] =   J[ 0][5]*Depl[ 0]+/*J[ 1][5]*Depl[ 1]*/ J[ 2][5]*Depl[ 2]+
            J[ 3][5]*Depl[ 3]+/*J[ 4][5]*Depl[ 4]*/ J[ 5][5]*Depl[ 5]+
            J[ 6][5]*Depl[ 6]+/*J[ 7][5]*Depl[ 7]*/ J[ 8][5]*Depl[ 8]+
            J[ 9][5]*Depl[ 9]+/*J[10][5]*Depl[10]*/ J[11][5]*Depl[11];
//         cerr<<"TetrahedralCorotationalFEMForceField<DataTypes>::computeForce, D = "<<Depl<<endl;
//         cerr<<"TetrahedralCorotationalFEMForceField<DataTypes>::computeForce, JtD = "<<JtD<<endl;

    Vec<6,Real> KJtD;
    KJtD[0] =   K[0][0]*JtD[0]+  K[0][1]*JtD[1]+  K[0][2]*JtD[2]
            /*K[0][3]*JtD[3]+  K[0][4]*JtD[4]+  K[0][5]*JtD[5]*/;
    KJtD[1] =   K[1][0]*JtD[0]+  K[1][1]*JtD[1]+  K[1][2]*JtD[2]
            /*K[1][3]*JtD[3]+  K[1][4]*JtD[4]+  K[1][5]*JtD[5]*/;
    KJtD[2] =   K[2][0]*JtD[0]+  K[2][1]*JtD[1]+  K[2][2]*JtD[2]
            /*K[2][3]*JtD[3]+  K[2][4]*JtD[4]+  K[2][5]*JtD[5]*/;
    KJtD[3] = /*K[3][0]*JtD[0]+  K[3][1]*JtD[1]+  K[3][2]*JtD[2]+*/
        K[3][3]*JtD[3] /*K[3][4]*JtD[4]+  K[3][5]*JtD[5]*/;
    KJtD[4] = /*K[4][0]*JtD[0]+  K[4][1]*JtD[1]+  K[4][2]*JtD[2]+*/
        /*K[4][3]*JtD[3]+*/K[4][4]*JtD[4] /*K[4][5]*JtD[5]*/;
    KJtD[5] = /*K[5][0]*JtD[0]+  K[5][1]*JtD[1]+  K[5][2]*JtD[2]+*/
        /*K[5][3]*JtD[3]+  K[5][4]*JtD[4]+*/K[5][5]*JtD[5]  ;

    F[ 0] =   J[ 0][0]*KJtD[0]+/*J[ 0][1]*KJtD[1]+  J[ 0][2]*KJtD[2]+*/
            J[ 0][3]*KJtD[3]+/*J[ 0][4]*KJtD[4]+*/J[ 0][5]*KJtD[5]  ;
    F[ 1] = /*J[ 1][0]*KJtD[0]+*/J[ 1][1]*KJtD[1]+/*J[ 1][2]*KJtD[2]+*/
            J[ 1][3]*KJtD[3]+  J[ 1][4]*KJtD[4] /*J[ 1][5]*KJtD[5]*/;
    F[ 2] = /*J[ 2][0]*KJtD[0]+  J[ 2][1]*KJtD[1]+*/J[ 2][2]*KJtD[2]+
            /*J[ 2][3]*KJtD[3]+*/J[ 2][4]*KJtD[4]+  J[ 2][5]*KJtD[5]  ;
    F[ 3] =   J[ 3][0]*KJtD[0]+/*J[ 3][1]*KJtD[1]+  J[ 3][2]*KJtD[2]+*/
            J[ 3][3]*KJtD[3]+/*J[ 3][4]*KJtD[4]+*/J[ 3][5]*KJtD[5]  ;
    F[ 4] = /*J[ 4][0]*KJtD[0]+*/J[ 4][1]*KJtD[1]+/*J[ 4][2]*KJtD[2]+*/
            J[ 4][3]*KJtD[3]+  J[ 4][4]*KJtD[4] /*J[ 4][5]*KJtD[5]*/;
    F[ 5] = /*J[ 5][0]*KJtD[0]+  J[ 5][1]*KJtD[1]+*/J[ 5][2]*KJtD[2]+
            /*J[ 5][3]*KJtD[3]+*/J[ 5][4]*KJtD[4]+  J[ 5][5]*KJtD[5]  ;
    F[ 6] =   J[ 6][0]*KJtD[0]+/*J[ 6][1]*KJtD[1]+  J[ 6][2]*KJtD[2]+*/
            J[ 6][3]*KJtD[3]+/*J[ 6][4]*KJtD[4]+*/J[ 6][5]*KJtD[5]  ;
    F[ 7] = /*J[ 7][0]*KJtD[0]+*/J[ 7][1]*KJtD[1]+/*J[ 7][2]*KJtD[2]+*/
            J[ 7][3]*KJtD[3]+  J[ 7][4]*KJtD[4] /*J[ 7][5]*KJtD[5]*/;
    F[ 8] = /*J[ 8][0]*KJtD[0]+  J[ 8][1]*KJtD[1]+*/J[ 8][2]*KJtD[2]+
            /*J[ 8][3]*KJtD[3]+*/J[ 8][4]*KJtD[4]+  J[ 8][5]*KJtD[5]  ;
    F[ 9] =   J[ 9][0]*KJtD[0]+/*J[ 9][1]*KJtD[1]+  J[ 9][2]*KJtD[2]+*/
            J[ 9][3]*KJtD[3]+/*J[ 9][4]*KJtD[4]+*/J[ 9][5]*KJtD[5]  ;
    F[10] = /*J[10][0]*KJtD[0]+*/J[10][1]*KJtD[1]+/*J[10][2]*KJtD[2]+*/
            J[10][3]*KJtD[3]+  J[10][4]*KJtD[4] /*J[10][5]*KJtD[5]*/;
    F[11] = /*J[11][0]*KJtD[0]+  J[11][1]*KJtD[1]+*/J[11][2]*KJtD[2]+
            /*J[11][3]*KJtD[3]+*/J[11][4]*KJtD[4]+  J[11][5]*KJtD[5]  ;
#endif
}

//////////////////////////////////////////////////////////////////////
////////////////////  small displacements method  ////////////////////
//////////////////////////////////////////////////////////////////////

template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::initSmall(int i, Index&a, Index&b, Index&c, Index&d)
{
    const VecCoord *X0=this->mstate->getX0();

    computeStrainDisplacement(tetrahedronInfo[i].strainDisplacementMatrix, (*X0)[a], (*X0)[b], (*X0)[c], (*X0)[d] );
}

template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::accumulateForceSmall( Vector& f, const Vector & p,Index elementIndex )
{
    //std::cerr<<"TetrahedralCorotationalFEMForceField<DataTypes>::accumulateForceSmall"<<std::endl;

    const Tetrahedron &t=_topology->getTetra(elementIndex);
    const VecCoord *X0=this->mstate->getX0();


    Index a = t[0];
    Index b = t[1];
    Index c = t[2];
    Index d = t[3];

    // displacements
    Displacement D;
    D[0] = 0;
    D[1] = 0;
    D[2] = 0;
    D[3] =  (*X0)[b][0] - (*X0)[a][0] - p[b][0]+p[a][0];
    D[4] =  (*X0)[b][1] - (*X0)[a][1] - p[b][1]+p[a][1];
    D[5] =  (*X0)[b][2] - (*X0)[a][2] - p[b][2]+p[a][2];
    D[6] =  (*X0)[c][0] - (*X0)[a][0] - p[c][0]+p[a][0];
    D[7] =  (*X0)[c][1] - (*X0)[a][1] - p[c][1]+p[a][1];
    D[8] =  (*X0)[c][2] - (*X0)[a][2] - p[c][2]+p[a][2];
    D[9] =  (*X0)[d][0] - (*X0)[a][0] - p[d][0]+p[a][0];
    D[10] = (*X0)[d][1] - (*X0)[a][1] - p[d][1]+p[a][1];
    D[11] = (*X0)[d][2] - (*X0)[a][2] - p[d][2]+p[a][2];
    /*        std::cerr<<"TetrahedralCorotationalFEMForceField<DataTypes>::accumulateForceSmall, displacement"<<D<<std::endl;
            std::cerr<<"TetrahedralCorotationalFEMForceField<DataTypes>::accumulateForceSmall, straindisplacement"<<tetrahedronInfo[elementIndex].strainDisplacementMatrix<<std::endl;
            std::cerr<<"TetrahedralCorotationalFEMForceField<DataTypes>::accumulateForceSmall, material"<<tetrahedronInfo[elementIndex].materialMatrix<<std::endl;*/

    // compute force on element
    Displacement F;

    if(!_assembling)
    {
        computeForce( F, D,tetrahedronInfo[elementIndex].materialMatrix,tetrahedronInfo[elementIndex].strainDisplacementMatrix );
        //std::cerr<<"TetrahedralCorotationalFEMForceField<DataTypes>::accumulateForceSmall, force"<<F<<std::endl;
    }
    else
    {
        Transformation Rot;
        Rot[0][0]=Rot[1][1]=Rot[2][2]=1;
        Rot[0][1]=Rot[0][2]=0;
        Rot[1][0]=Rot[1][2]=0;
        Rot[2][0]=Rot[2][1]=0;


        StiffnessMatrix JKJt,tmp;
        computeStiffnessMatrix(JKJt,tmp,tetrahedronInfo[elementIndex].materialMatrix,tetrahedronInfo[elementIndex].strainDisplacementMatrix,Rot);

        //erase the stiffness matrix at each time step
        if(elementIndex==0)
        {
            for(unsigned int i=0; i<_stiffnesses.size(); ++i)
            {
                _stiffnesses[i].resize(0);
            }
        }

        for(int i=0; i<12; ++i)
        {
            int row = t[i/3]*3+i%3;

            for(int j=0; j<12; ++j)
            {
                if(JKJt[i][j]!=0)
                {

                    int col = t[j/3]*3+j%3;
                    //cerr<<row<<" "<<col<<endl;

                    //typename CompressedValue::iterator result = _stiffnesses[row].find(col);


                    // search if the vertex is already take into account by another element
                    typename CompressedValue::iterator result = _stiffnesses[row].end();
                    for(typename CompressedValue::iterator it=_stiffnesses[row].begin(); it!=_stiffnesses[row].end()&&result==_stiffnesses[row].end(); ++it)
                    {
                        if( (*it).first == col )
                            result = it;
                    }

                    if( result==_stiffnesses[row].end() )
                        _stiffnesses[row].push_back( Col_Value(col,JKJt[i][j] )  );
                    else
                        (*result).second += JKJt[i][j];
                }
            }
        }

        /*for(unsigned int i=0;i<_stiffnesses.size();++i)
        	for(typename CompressedValue::iterator it=_stiffnesses[i].begin();it!=_stiffnesses[i].end();++it)
        		cerr<<i<<" "<<(*it).first<<"   "<<(*it).second<<"   "<<JKJt[i][(*it).first]<<endl;*/

        F = JKJt * D;
    }

    f[a] += Deriv( F[0], F[1], F[2] );
    f[b] += Deriv( F[3], F[4], F[5] );
    f[c] += Deriv( F[6], F[7], F[8] );
    f[d] += Deriv( F[9], F[10], F[11] );
}

template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::applyStiffnessSmall( Vector& f, const Vector& x, int i, Index a, Index b, Index c, Index d )
{
    Displacement X;

    X[0] = x[a][0];
    X[1] = x[a][1];
    X[2] = x[a][2];

    X[3] = x[b][0];
    X[4] = x[b][1];
    X[5] = x[b][2];

    X[6] = x[c][0];
    X[7] = x[c][1];
    X[8] = x[c][2];

    X[9] = x[d][0];
    X[10] = x[d][1];
    X[11] = x[d][2];

    Displacement F;
    computeForce( F, X,tetrahedronInfo[i].materialMatrix,tetrahedronInfo[i].strainDisplacementMatrix);

    f[a] += Deriv( -F[0], -F[1],  -F[2] );
    f[b] += Deriv( -F[3], -F[4],  -F[5] );
    f[c] += Deriv( -F[6], -F[7],  -F[8] );
    f[d] += Deriv( -F[9], -F[10], -F[11] );
}

//////////////////////////////////////////////////////////////////////
////////////////////  large displacements method  ////////////////////
//////////////////////////////////////////////////////////////////////

template<class DataTypes>
inline void TetrahedralCorotationalFEMForceField<DataTypes>::computeRotationLarge( Transformation &r, const Vector &p, const Index &a, const Index &b, const Index &c)
{
    // first vector on first edge
    // second vector in the plane of the two first edges
    // third vector orthogonal to first and second

    Coord edgex = p[b]-p[a];
    edgex.normalize();

    Coord edgey = p[c]-p[a];
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
void TetrahedralCorotationalFEMForceField<DataTypes>::initLarge(int i, Index&a, Index&b, Index&c, Index&d)
{
    // Rotation matrix (initial Tetrahedre/world)
    // first vector on first edge
    // second vector in the plane of the two first edges
    // third vector orthogonal to first and second
    const VecCoord *X0=this->mstate->getX0();

    Transformation R_0_1;
    computeRotationLarge( R_0_1, (*X0), a, b, c);

    tetrahedronInfo[i].rotatedInitialElements[0] = R_0_1*(*X0)[a];
    tetrahedronInfo[i].rotatedInitialElements[1] = R_0_1*(*X0)[b];
    tetrahedronInfo[i].rotatedInitialElements[2] = R_0_1*(*X0)[c];
    tetrahedronInfo[i].rotatedInitialElements[3] = R_0_1*(*X0)[d];

//	cerr<<"a,b,c : "<<a<<" "<<b<<" "<<c<<endl;
//	cerr<<"_initialPoints : "<<_initialPoints<<endl;
//	cerr<<"R_0_1 large : "<<R_0_1<<endl;

    tetrahedronInfo[i].rotatedInitialElements[1] -= tetrahedronInfo[i].rotatedInitialElements[0];
    tetrahedronInfo[i].rotatedInitialElements[2] -= tetrahedronInfo[i].rotatedInitialElements[0];
    tetrahedronInfo[i].rotatedInitialElements[3] -= tetrahedronInfo[i].rotatedInitialElements[0];
    tetrahedronInfo[i].rotatedInitialElements[0] = Coord(0,0,0);

//	cerr<<"_rotatedInitialElements : "<<_rotatedInitialElements<<endl;

    computeStrainDisplacement( tetrahedronInfo[i].strainDisplacementMatrix,tetrahedronInfo[i].rotatedInitialElements[0], tetrahedronInfo[i].rotatedInitialElements[1],tetrahedronInfo[i].rotatedInitialElements[2],tetrahedronInfo[i].rotatedInitialElements[3] );
}

template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::accumulateForceLarge( Vector& f, const Vector & p, Index elementIndex )
{
    const Tetrahedron &t=_topology->getTetra(elementIndex);

    // Rotation matrix (deformed and displaced Tetrahedron/world)
    Transformation R_0_2;
    computeRotationLarge( R_0_2, p, t[0],t[1],t[2]);
    tetrahedronInfo[elementIndex].rotation.transpose(R_0_2);
    //cerr<<"R_0_2 large : "<<R_0_2<<endl;

    // positions of the deformed and displaced Tetrahedron in its frame
    helper::fixed_array<Coord,4> deforme;
    for(int i=0; i<4; ++i)
        deforme[i] = R_0_2*p[t[i]];

    deforme[1][0] -= deforme[0][0];
    deforme[2][0] -= deforme[0][0];
    deforme[2][1] -= deforme[0][1];
    deforme[3] -= deforme[0];

    // displacement
    Displacement D;
    D[0] = 0;
    D[1] = 0;
    D[2] = 0;
    D[3] = tetrahedronInfo[elementIndex].rotatedInitialElements[1][0] - deforme[1][0];
    D[4] = 0;
    D[5] = 0;
    D[6] = tetrahedronInfo[elementIndex].rotatedInitialElements[2][0] - deforme[2][0];
    D[7] = tetrahedronInfo[elementIndex].rotatedInitialElements[2][1] - deforme[2][1];
    D[8] = 0;
    D[9] = tetrahedronInfo[elementIndex].rotatedInitialElements[3][0] - deforme[3][0];
    D[10] = tetrahedronInfo[elementIndex].rotatedInitialElements[3][1] - deforme[3][1];
    D[11] =tetrahedronInfo[elementIndex].rotatedInitialElements[3][2] - deforme[3][2];

    //cerr<<"D : "<<D<<endl;

    Displacement F;
    if(_updateStiffnessMatrix)
    {
        tetrahedronInfo[elementIndex].strainDisplacementMatrix[0][0]   = ( - deforme[2][1]*deforme[3][2] );
        tetrahedronInfo[elementIndex].strainDisplacementMatrix[1][1] = ( deforme[2][0]*deforme[3][2] - deforme[1][0]*deforme[3][2] );
        tetrahedronInfo[elementIndex].strainDisplacementMatrix[2][2]   = ( deforme[2][1]*deforme[3][0] - deforme[2][0]*deforme[3][1] + deforme[1][0]*deforme[3][1] - deforme[1][0]*deforme[2][1] );

        tetrahedronInfo[elementIndex].strainDisplacementMatrix[3][0]   = ( deforme[2][1]*deforme[3][2] );
        tetrahedronInfo[elementIndex].strainDisplacementMatrix[4][1]  = ( - deforme[2][0]*deforme[3][2] );
        tetrahedronInfo[elementIndex].strainDisplacementMatrix[5][2]   = ( - deforme[2][1]*deforme[3][0] + deforme[2][0]*deforme[3][1] );

        tetrahedronInfo[elementIndex].strainDisplacementMatrix[7][1]  = ( deforme[1][0]*deforme[3][2] );
        tetrahedronInfo[elementIndex].strainDisplacementMatrix[8][2]   = ( - deforme[1][0]*deforme[3][1] );

        tetrahedronInfo[elementIndex].strainDisplacementMatrix[11][2] = ( deforme[1][0]*deforme[2][1] );
    }

    if(!_assembling)
    {
        // compute force on element
        computeForce( F, D, tetrahedronInfo[elementIndex].materialMatrix, tetrahedronInfo[elementIndex].strainDisplacementMatrix);
        for(int i=0; i<12; i+=3)
            f[t[i/3]] += tetrahedronInfo[elementIndex].rotation * Deriv( F[i], F[i+1],  F[i+2] );

        //cerr<<"p large : "<<p<<endl;
        //cerr<<"F large : "<<f<<endl;
//		for(int i=0;i<12;i+=3)
//		{
//			Vec tmp;
//			v_eq_Ab( tmp, _rotations[elementIndex], Vec( F[i], F[i+1],  F[i+2] ) );
//			cerr<<tmp<<"\t";
//		}
//		cerr<<endl;
    }
    else
    {
        tetrahedronInfo[elementIndex].strainDisplacementMatrix[6][0] = 0;
        tetrahedronInfo[elementIndex].strainDisplacementMatrix[9][0] = 0;
        tetrahedronInfo[elementIndex].strainDisplacementMatrix[10][1] = 0;

        StiffnessMatrix RJKJt, RJKJtRt;
        computeStiffnessMatrix(RJKJt,RJKJtRt,tetrahedronInfo[elementIndex].materialMatrix, tetrahedronInfo[elementIndex].strainDisplacementMatrix,tetrahedronInfo[elementIndex].rotation);


        //erase the stiffness matrix at each time step
        if(elementIndex==0)
        {
            for(unsigned int i=0; i<_stiffnesses.size(); ++i)
            {
                _stiffnesses[i].resize(0);
            }
        }

        for(int i=0; i<12; ++i)
        {
            int row = t[i/3]*3+i%3;

            for(int j=0; j<12; ++j)
            {
                int col = t[j/3]*3+j%3;

                // search if the vertex is already take into account by another element
                typename CompressedValue::iterator result = _stiffnesses[row].end();
                for(typename CompressedValue::iterator it=_stiffnesses[row].begin(); it!=_stiffnesses[row].end()&&result==_stiffnesses[row].end(); ++it)
                {
                    if( (*it).first == col )
                    {
                        result = it;
                    }
                }

                if( result==_stiffnesses[row].end() )
                {
                    _stiffnesses[row].push_back( Col_Value(col,RJKJtRt[i][j] )  );
                }
                else
                {
                    (*result).second += RJKJtRt[i][j];
                }
            }
        }

        F = RJKJt*D;

        for(int i=0; i<12; i+=3)
            f[t[i/3]] += Deriv( F[i], F[i+1],  F[i+2] );
    }
}

template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::applyStiffnessLarge( Vector& f, const Vector& x, int i, Index a, Index b, Index c, Index d )
{
    Transformation R_0_2;
    R_0_2.transpose(tetrahedronInfo[i].rotation);

    Displacement X;
    Coord x_2;

    x_2 = R_0_2*x[a];
    X[0] = x_2[0];
    X[1] = x_2[1];
    X[2] = x_2[2];

    x_2 = R_0_2*x[b];
    X[3] = x_2[0];
    X[4] = x_2[1];
    X[5] = x_2[2];

    x_2 = R_0_2*x[c];
    X[6] = x_2[0];
    X[7] = x_2[1];
    X[8] = x_2[2];

    x_2 = R_0_2*x[d];
    X[9] = x_2[0];
    X[10] = x_2[1];
    X[11] = x_2[2];

    Displacement F;

    //cerr<<"X : "<<X<<endl;

    computeForce( F, X,tetrahedronInfo[i].materialMatrix, tetrahedronInfo[i].strainDisplacementMatrix);

    //cerr<<"F : "<<F<<endl;

    f[a] += tetrahedronInfo[i].rotation * Deriv( -F[0], -F[1],  -F[2] );
    f[b] += tetrahedronInfo[i].rotation * Deriv( -F[3], -F[4],  -F[5] );
    f[c] += tetrahedronInfo[i].rotation * Deriv( -F[6], -F[7],  -F[8] );
    f[d] += tetrahedronInfo[i].rotation * Deriv( -F[9], -F[10], -F[11] );
}

//////////////////////////////////////////////////////////////////////
////////////////////  polar decomposition method  ////////////////////
//////////////////////////////////////////////////////////////////////

template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::initPolar(int i, Index& a, Index&b, Index&c, Index&d)
{
    const VecCoord *X0=this->mstate->getX0();

    Transformation A;
    A[0] = (*X0)[b]-(*X0)[a];
    A[1] = (*X0)[c]-(*X0)[a];
    A[2] = (*X0)[d]-(*X0)[a];
    tetrahedronInfo[i].initialTransformation = A;

    Transformation R_0_1;
    MatNoInit<3,3,Real> S;
    polar_decomp(A, R_0_1, S);

    tetrahedronInfo[i].rotatedInitialElements[0] = R_0_1*(*X0)[a];
    tetrahedronInfo[i].rotatedInitialElements[1] = R_0_1*(*X0)[b];
    tetrahedronInfo[i].rotatedInitialElements[2] = R_0_1*(*X0)[c];
    tetrahedronInfo[i].rotatedInitialElements[3] = R_0_1*(*X0)[d];

    computeStrainDisplacement( tetrahedronInfo[i].strainDisplacementMatrix,tetrahedronInfo[i].rotatedInitialElements[0], tetrahedronInfo[i].rotatedInitialElements[1],tetrahedronInfo[i].rotatedInitialElements[2],tetrahedronInfo[i].rotatedInitialElements[3] );
}

template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::accumulateForcePolar( Vector& f, const Vector & p, Index elementIndex )
{
    const Tetrahedron &t=_topology->getTetra(elementIndex);

    Transformation A;
    A[0] = p[t[1]]-p[t[0]];
    A[1] = p[t[2]]-p[t[0]];
    A[2] = p[t[3]]-p[t[0]];

    Transformation R_0_2;
    MatNoInit<3,3,Real> S;
    polar_decomp(A, R_0_2, S);

    tetrahedronInfo[elementIndex].rotation.transpose( R_0_2 );

    // positions of the deformed and displaced Tetrahedre in its frame
    helper::fixed_array<Coord, 4>  deforme;
    for(int i=0; i<4; ++i)
        deforme[i] = R_0_2 * p[t[i]];

    // displacement
    Displacement D;
    D[0] = tetrahedronInfo[elementIndex].rotatedInitialElements[0][0] - deforme[0][0];
    D[1] = tetrahedronInfo[elementIndex].rotatedInitialElements[0][1] - deforme[0][1];
    D[2] = tetrahedronInfo[elementIndex].rotatedInitialElements[0][2] - deforme[0][2];
    D[3] = tetrahedronInfo[elementIndex].rotatedInitialElements[1][0] - deforme[1][0];
    D[4] = tetrahedronInfo[elementIndex].rotatedInitialElements[1][1] - deforme[1][1];
    D[5] = tetrahedronInfo[elementIndex].rotatedInitialElements[1][2] - deforme[1][2];
    D[6] = tetrahedronInfo[elementIndex].rotatedInitialElements[2][0] - deforme[2][0];
    D[7] = tetrahedronInfo[elementIndex].rotatedInitialElements[2][1] - deforme[2][1];
    D[8] = tetrahedronInfo[elementIndex].rotatedInitialElements[2][2] - deforme[2][2];
    D[9] = tetrahedronInfo[elementIndex].rotatedInitialElements[3][0] - deforme[3][0];
    D[10] = tetrahedronInfo[elementIndex].rotatedInitialElements[3][1] - deforme[3][1];
    D[11] = tetrahedronInfo[elementIndex].rotatedInitialElements[3][2] - deforme[3][2];
    //cerr<<"D : "<<D<<endl;

    Displacement F;
    if(_updateStiffnessMatrix)
    {
        // shape functions matrix
        computeStrainDisplacement( tetrahedronInfo[elementIndex].strainDisplacementMatrix, deforme[0],deforme[1],deforme[2],deforme[3]  );
    }

    if(!_assembling)
    {
        computeForce( F, D, tetrahedronInfo[elementIndex].materialMatrix, tetrahedronInfo[elementIndex].strainDisplacementMatrix );
        for(int i=0; i<12; i+=3)
            f[t[i/3]] += tetrahedronInfo[elementIndex].rotation * Deriv( F[i], F[i+1],  F[i+2] );
    }
    else
    {
        std::cerr << "TODO(TetrahedralCorotationalFEMForceField): support for assembling system matrix when using polar method.\n";
    }
}

template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::applyStiffnessPolar( Vector& f, const Vector& x, int i, Index a, Index b, Index c, Index d )
{
    Transformation R_0_2;
    R_0_2.transpose( tetrahedronInfo[i].rotation );

    Displacement X;
    Coord x_2;

    x_2 = R_0_2*x[a];
    X[0] = x_2[0];
    X[1] = x_2[1];
    X[2] = x_2[2];

    x_2 = R_0_2*x[b];
    X[3] = x_2[0];
    X[4] = x_2[1];
    X[5] = x_2[2];

    x_2 = R_0_2*x[c];
    X[6] = x_2[0];
    X[7] = x_2[1];
    X[8] = x_2[2];

    x_2 = R_0_2*x[d];
    X[9] = x_2[0];
    X[10] = x_2[1];
    X[11] = x_2[2];

    Displacement F;

    //cerr<<"X : "<<X<<endl;

    computeForce( F, X, tetrahedronInfo[i].materialMatrix, tetrahedronInfo[i].strainDisplacementMatrix);

    //cerr<<"F : "<<F<<endl;

    f[a] -= tetrahedronInfo[i].rotation * Deriv( F[0], F[1],  F[2] );
    f[b] -= tetrahedronInfo[i].rotation * Deriv( F[3], F[4],  F[5] );
    f[c] -= tetrahedronInfo[i].rotation * Deriv( F[6], F[7],  F[8] );
    f[d] -= tetrahedronInfo[i].rotation * Deriv( F[9], F[10], F[11] );
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::draw()
{
    if (!getContext()->getShowForceFields()) return;
    if (!this->mstate) return;

    const VecCoord& x = *this->mstate->getX();

    if (getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glDisable(GL_LIGHTING);

    glBegin(GL_TRIANGLES);

    for(int i = 0 ; i<_topology->getNbTetras(); ++i)
    {
        const Tetrahedron &t=_topology->getTetra(i);

        Index a = t[0];
        Index b = t[1];
        Index c = t[2];
        Index d = t[3];
        Coord center = (x[a]+x[b]+x[c]+x[d])*0.125;
        Coord pa = (x[a]+center)*(Real)0.666667;
        Coord pb = (x[b]+center)*(Real)0.666667;
        Coord pc = (x[c]+center)*(Real)0.666667;
        Coord pd = (x[d]+center)*(Real)0.666667;

        glColor4f(0,0,1,1);
        helper::gl::glVertexT(pa);
        helper::gl::glVertexT(pb);
        helper::gl::glVertexT(pc);

        glColor4f(0,0.5,1,1);
        helper::gl::glVertexT(pb);
        helper::gl::glVertexT(pc);
        helper::gl::glVertexT(pd);

        glColor4f(0,1,1,1);
        helper::gl::glVertexT(pc);
        helper::gl::glVertexT(pd);
        helper::gl::glVertexT(pa);

        glColor4f(0.5,1,1,1);
        helper::gl::glVertexT(pd);
        helper::gl::glVertexT(pa);
        helper::gl::glVertexT(pb);
    }
    glEnd();

    if (getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}


template<class DataTypes>
void TetrahedralCorotationalFEMForceField<DataTypes>::addKToMatrix(sofa::defaulttype::BaseMatrix *mat, SReal /*k*/, unsigned int &offset)
{
    // Build Matrix Block for this ForceField
    unsigned int i,j,n1, n2, row, column, ROW, COLUMN;

    Transformation Rot;
    StiffnessMatrix JKJt,tmp;

    Index noeud1, noeud2;

    Rot[0][0]=Rot[1][1]=Rot[2][2]=1;
    Rot[0][1]=Rot[0][2]=0;
    Rot[1][0]=Rot[1][2]=0;
    Rot[2][0]=Rot[2][1]=0;

    for(int IT=0 ; IT != _topology->getNbTetras() ; ++IT)
    {
        computeStiffnessMatrix(JKJt,tmp,tetrahedronInfo[IT].materialMatrix,tetrahedronInfo[IT].strainDisplacementMatrix,Rot);
        const Tetrahedron &t=_topology->getTetra(IT);

        // find index of node 1
        for (n1=0; n1<4; n1++)
        {
            noeud1 = t[n1];

            for(i=0; i<3; i++)
            {
                ROW = offset+3*noeud1+i;
                row = 3*n1+i;
                // find index of node 2
                for (n2=0; n2<4; n2++)
                {
                    noeud2 = t[n2];

                    for (j=0; j<3; j++)
                    {
                        COLUMN = offset+3*noeud2+j;
                        column = 3*n2+j;
                        mat->add(ROW, COLUMN, tmp[row][column]);
                    }
                }
            }
        }
    }
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
