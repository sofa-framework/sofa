#ifndef SOFA_COMPONENT_FORCEFIELD_BEAMFEMFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_BEAMFEMFORCEFIELD_INL

#include <sofa/core/componentmodel/behavior/ForceField.inl>
#include <sofa/component/forcefield/BeamFEMForceField.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/topology/GridTopology.h>
#include <sofa/helper/PolarDecompose.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/Axis.h>
#include <sofa/helper/rmath.h>
#include <assert.h>
#include <iostream>
#include <set>
#include <GL/gl.h>
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

template <class DataTypes>
void BeamFEMForceField<DataTypes>::init()
{
    this->core::componentmodel::behavior::ForceField<DataTypes>::init();


    _E  = _youngModulus.getValue();
    _nu = _poissonRatio.getValue();
    //_L  = L;
    _r  = _radius.getValue();

    _G  =_E/(2.0*(1.0+_nu));
    _Iz = R_PI*_r*_r*_r*_r/4;
    _Iy = _Iz ;
    _J  = _Iz+_Iy;
    _A  = R_PI*_r*_r;

    if (_timoshenko.getValue())
    {
        _Asy = 10.0/9.0;
        _Asz = 10.0/9.0;
    }
    else
    {
        _Asy = 0.0;
        _Asz = 0.0;
    }

    _mesh = dynamic_cast<sofa::component::topology::MeshTopology*>(this->getContext()->getTopology());
    if (_mesh==NULL || _mesh->getLines().empty())
    {
        std::cerr << "ERROR(BeamFEMForceField): object must have a edge MeshTopology.\n";
        return;
    }
    _indexedElements = & (_mesh->getLines());

    if (_initialPoints.getValue().size() == 0)
    {
        VecCoord& p = *this->mstate->getX();
        _initialPoints.setValue(p);
    }

    //_strainDisplacements.resize( _indexedElements->size() );
    //_stiffnesses.resize( _initialPoints.getValue().size()*3 );
    //_materialsStiffnesses.resize(_indexedElements->size() );
    _stiffnessMatrices.resize(_indexedElements->size() );
    _forces.resize( _initialPoints.getValue().size() );

    //case LARGE :
    {
        _rotations.resize( _indexedElements->size() );
        _initialLength.resize(_indexedElements->size());
        unsigned int i=0;
        typename VecElement::const_iterator it;
        for(it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i)
        {
            Index a = (*it)[0];
            Index b = (*it)[1];
            defaulttype::Vec<3,Real> dp; dp = _initialPoints.getValue()[a].getCenter()-_initialPoints.getValue()[b].getCenter();
            _initialLength[i] = dp.norm();
            //computeMaterialStiffness(i,a,b);
            computeStiffness(i,a,b);
            initLarge(i,a,b);
        }
        //break;
    }

    std::cout << "BeamFEMForceField: init OK, "<<_indexedElements->size()<<" elements."<<std::endl;
}


template<class DataTypes>
void BeamFEMForceField<DataTypes>::addForce (VecDeriv& f, const VecCoord& p, const VecDeriv& /*v*/)
{
    f.resize(p.size());

    // First compute each node rotation
    unsigned int i;

    _nodeRotations.resize(p.size());
    //Mat3x3d R; R.identity();
    for(i=0; i<p.size(); ++i)
    {
        //R = R * MatrixFromEulerXYZ(p[i][3], p[i][4], p[i][5]);
        //_nodeRotations[i] = R;
        p[i].getOrientation().toMatrix(_nodeRotations[i]);
    }

    typename VecElement::const_iterator it;

    for(it=_indexedElements->begin(),i=0; it!=_indexedElements->end(); ++it,++i)
    {
        Index a = (*it)[0];
        Index b = (*it)[1];

        accumulateForceLarge( f, p, i, a, b );
    }

}

template<class DataTypes>
void BeamFEMForceField<DataTypes>::addDForce (VecDeriv& v, const VecDeriv& x)
{
    v.resize(x.size());
    //if(_assembling) applyStiffnessAssembled(v,x);
    //else
    {
        unsigned int i=0;
        typename VecElement::const_iterator it;

        for(it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i)
        {
            Index a = (*it)[0];
            Index b = (*it)[1];

            applyStiffnessLarge( v,x, i, a, b );
        }
    }
}

/*
template <typename V, typename R, typename E>
void BeamFEMForceField<V,R,E>::applyStiffnessAssembled( Vector& v const Vector& x )
{
    for(unsigned int i=0;i<v.size();++i)
    {
        for(int k=0;k<3;++k)
        {
            int row = i*3+k;

            Real val = 0;
            for(typename CompressedValue::iterator it=_stiffnesses[row].begin();it!=_stiffnesses[row].end();++it)
            {
                int col = (*it).first;
                val += ( (*it).second * x[col/3][col%3] );
            }
            v[i][k] += (-val);
        }
    }
}
*/

#if 0
template<class DataTypes>
void BeamFEMForceField<DataTypes>::computeStrainDisplacement( StrainDisplacement &J, Coord a, Coord b, Coord c, Coord d )
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
#endif

template<class DataTypes>
typename BeamFEMForceField<DataTypes>::Real BeamFEMForceField<DataTypes>::peudo_determinant_for_coef ( const Mat<2, 3, Real>&  M )
{
    return  M[0][1]*M[1][2] - M[1][1]*M[0][2] -  M[0][0]*M[1][2] + M[1][0]*M[0][2] + M[0][0]*M[1][1] - M[1][0]*M[0][1];
}

#if 0
template<class DataTypes>
void BeamFEMForceField<DataTypes>::computeStiffnessMatrix( StiffnessMatrix& S,StiffnessMatrix& SR,const MaterialStiffness &K, const StrainDisplacement &J, const Transformation& Rot )
{
    Mat<6, 12, Real> Jt;
    Jt.transpose( J );

    Mat<12, 12, Real> JKJt;
    JKJt = J*K*Jt;

    Mat<12, 12, Real> RR,RRt;
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
#endif

template<class DataTypes>
void BeamFEMForceField<DataTypes>::computeStiffness(int i, Index , Index )
{
    double   phiy, phiz;
    double _L = _initialLength[i];
    double   L2 = _L * _L;
    double   L3 = L2 * _L;
    double   EIy = _E * _Iy;
    double   EIz = _E * _Iz;

    // Find shear-deformation parameters
    if (_Asy == 0)
        phiy = 0.0;
    else
        phiy = 24.0*(1.0+_nu)*_Iz/(_Asy*L2);

    if (_Asz == 0)
        phiz = 0.0;
    else
        phiz = 24.0*(1.0+_nu)*_Iy/(_Asz*L2);

    StiffnessMatrix& k_loc = _stiffnessMatrices[i];

    // Define stiffness matrix 'k' in local coordinates
    k_loc.clear();
    k_loc[6][6]   = k_loc[0][0]   = _E*_A/_L;
    k_loc[7][7]   = k_loc[1][1]   = 12.0*EIz/(L3*(1.0+phiy));
    k_loc[8][8]   = k_loc[2][2]   = 12.0*EIy/(L3*(1.0+phiz));
    k_loc[9][9]   = k_loc[3][3]   = _G*_J/_L;
    k_loc[10][10] = k_loc[4][4]   = (4.0+phiz)*EIy/(_L*(1.0+phiz));
    k_loc[11][11] = k_loc[5][5]   = (4.0+phiy)*EIz/(_L*(1.0+phiy));

    k_loc[4][2]   = -6.0*EIy/(L2*(1.0+phiz));
    k_loc[5][1]   =  6.0*EIz/(L2*(1.0+phiy));
    k_loc[6][0]   = -k_loc[0][0];
    k_loc[7][1]   = -k_loc[1][1];
    k_loc[7][5]   = -k_loc[5][1];
    k_loc[8][2]   = -k_loc[2][2];
    k_loc[8][4]   = -k_loc[4][2];
    k_loc[9][3]   = -k_loc[3][3];
    k_loc[10][2]  = k_loc[4][2];
    k_loc[10][4]  = (2.0-phiz)*EIy/(_L*(1.0+phiz));
    k_loc[10][8]  = -k_loc[4][2];
    k_loc[11][1]  = k_loc[5][1];
    k_loc[11][5]  = (2.0-phiy)*EIz/(_L*(1.0+phiy));
    k_loc[11][7]  = -k_loc[5][1];

    for (int i=0; i<=10; i++)
        for (int j=i+1; j<12; j++)
            k_loc[i][j] = k_loc[j][i];
}

#if 0
template<class DataTypes>
void BeamFEMForceField<DataTypes>::computeForce( Displacement &F, const Displacement &Depl, const MaterialStiffness &K, const StrainDisplacement &J )
{
    //Mat<6, 12, Real> Jt;
    //Jt.transpose(J);
    //F = J*(K*(Jt*Depl));
    F = J*(K*(J.multTranspose(Depl)));
    return;

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

    Vec<6,Real> JtD;
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
//         cerr<<"BeamFEMForceField<DataTypes>::computeForce, D = "<<Depl<<endl;
//         cerr<<"BeamFEMForceField<DataTypes>::computeForce, JtD = "<<JtD<<endl;

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
}
#endif

////////////// large displacements method
template<class DataTypes>
void BeamFEMForceField<DataTypes>::initLarge(int i, Index , Index )
{
    _rotations[i].identity();
    // Rotation matrix (initial Tetrahedre/world)
    // first vector on first edge
    // second vector in the plane of the two first edges
    // third vector orthogonal to first and second
    /*

            Transformation R_0_1;
            computeRotationLarge( R_0_1, _initialPoints.getValue(), a, b, c);

            _rotatedInitialElements[i][0] = R_0_1*_initialPoints.getValue()[a];
            _rotatedInitialElements[i][1] = R_0_1*_initialPoints.getValue()[b];
            _rotatedInitialElements[i][2] = R_0_1*_initialPoints.getValue()[c];
            _rotatedInitialElements[i][3] = R_0_1*_initialPoints.getValue()[d];

    //         cerr<<"a,b,c : "<<a<<" "<<b<<" "<<c<<endl;
    //         cerr<<"_initialPoints : "<<_initialPoints<<endl;
    //         cerr<<"R_0_1 large : "<<R_0_1<<endl;

            _rotatedInitialElements[i][1] -= _rotatedInitialElements[i][0];
            _rotatedInitialElements[i][2] -= _rotatedInitialElements[i][0];
            _rotatedInitialElements[i][3] -= _rotatedInitialElements[i][0];
            _rotatedInitialElements[i][0] = Coord(0,0,0);


    //         cerr<<"_rotatedInitialElements : "<<_rotatedInitialElements<<endl;

            computeStrainDisplacement( _strainDisplacements[i],_rotatedInitialElements[i][0], _rotatedInitialElements[i][1],_rotatedInitialElements[i][2],_rotatedInitialElements[i][3] );
    */
}
/*
template<class DataTypes>
void BeamFEMForceField<DataTypes>::computeRotationLarge( Transformation &r, const Vector &p, const Index &a, const Index &b, const Index &c)
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
}*/

template<class DataTypes>
void BeamFEMForceField<DataTypes>::accumulateForceLarge( VecDeriv& f, const VecCoord & x, int i, Index a, Index b )
{
    Transformation& lambda = _rotations[i];

    Vec3d locX1 =  _nodeRotations[a].col(0);
    Vec3d locY1 =  _nodeRotations[a].col(1);
    Vec3d locZ1 =  _nodeRotations[a].col(2);

    //Vec3d locX2 =  _nodeRotations[b].col(0);
    //Vec3d locY2 =  _nodeRotations[b].col(1);
    //Vec3d locZ2 =  _nodeRotations[b].col(2);

    locX1 = (x[b].getCenter()-x[a].getCenter());
    // Make orthonormal
    locZ1 = cross(locX1,locY1);
    locY1 = cross(locZ1,locX1);
    locX1.normalize();
    //locY1 -= locX1 * dot(locX1, locY1);
    locY1.normalize();
    //locZ1 -= locX1 * dot(locX1, locZ1) + locY1 * dot(locY1, locZ1);
    locZ1.normalize();

    lambda[0][0] = locX1[0]; lambda[1][0] = locX1[1]; lambda[2][0] = locX1[2];
    lambda[0][1] = locY1[0]; lambda[1][1] = locY1[1]; lambda[2][1] = locY1[2];
    lambda[0][2] = locZ1[0]; lambda[1][2] = locZ1[1]; lambda[2][2] = locZ1[2];
    //lambda[0] = locX1;
    //lambda[1] = locY1;
    //lambda[2] = locZ1;

    // Apply lambda

    Displacement depl;
    // U = R1*p1p2 - p1p2_init
    Vec<3,Real> U; U = x[b].getCenter()-x[a].getCenter();
    U = lambda.multTranspose(U);
    U[0] -= _initialLength[i];

//      double dthetaY2 = -asin(dot(locZ1,_nodeRotations[b].col(0)));
//      double param2 = 1.0 / cos(dthetaY2);
//      double dthetaZ2 =  asin( dot(locY1,_nodeRotations[b].col(0)) * param2);
//      double dthetaX2 =  asin( dot(locZ1,_nodeRotations[b].col(1)) * param2);

//      double dthetaY1 = -asin(dot(locZ1,_nodeRotations[a].col(0)));
//      double param1 = 1.0 / cos(dthetaY1);
//      double dthetaZ1 =  asin( dot(locY1,_nodeRotations[a].col(0)) * param1);
//      double dthetaX1 =  asin( dot(locZ1,_nodeRotations[a].col(1)) * param1);
//
//     std::cout << "dtheta1 = "<<dthetaX1<<" "<<dthetaY1<<" "<<dthetaZ1<<"\n";
//
//     depl[3] = dthetaX1;
//     depl[4] = dthetaY1;
//     depl[5] = dthetaZ1;
    depl[6] = U[0];
    depl[7] = U[1];
    depl[8] = U[2];
//     depl[9] = dthetaX2;
//     depl[10] = dthetaY2;
//     depl[11] = dthetaZ2;

    Quat q0; q0.fromMatrix(lambda); q0[3] = -q0[3]; //q0 = q0.inverse();

    //Quat qa_inv = x[a].getOrientation().inverse(); //qa.normalize();
    Quat qa = x[a].getOrientation(); qa.normalize();
    //qa[3] = -qa[3];
    if (q0[0]*qa[0]+q0[1]*qa[1]+q0[2]*qa[2]+q0[3]*qa[3] < 0) { qa[0] = -qa[0]; qa[1] = -qa[1]; qa[2] = -qa[2]; qa[3] = -qa[3]; }
    Quat qb = x[b].getOrientation(); qb.normalize();
    //qb[3] = -qb[3];
    if (q0[0]*qb[0]+q0[1]*qb[1]+q0[2]*qb[2]+q0[3]*qb[3] < 0) { qb[0] = -qb[0]; qb[1] = -qb[1]; qb[2] = -qb[2]; qb[3] = -qb[3]; }

    //std::cout << "qa = "<<qa<<" qb = "<<qb<<" q0 = "<<q0<<std::endl;

    Quat dq2 = qb * q0;
    if (dq2[3] < 0) { dq2[0] = -dq2[0]; dq2[1] = -dq2[1]; dq2[2] = -dq2[2]; dq2[3] = -dq2[3]; }
    //std::cout << "dq2 = "<<dq2<<std::endl;
    Vec<3,Real> V2;
    double half_theta2 = acos(dq2[3]);
    //std::cout << i<<" qa_inv = "<<qa_inv<<" qb = "<<qb<<" dq = "<<dq<<" theta = "<<2*half_theta<<std::endl;
    if (half_theta2 > 0.0000001) // || half_theta2 < -0.0000001)
        V2 = Vec<3,Real>(dq2[0],dq2[1],dq2[2])*(2*half_theta2/sin(half_theta2));
    //std::cout << "V2 = "<<V2<<std::endl;
    //std::cout << "V2 = "<<V2<<std::endl;

    Quat dq1 = qa * q0;
    if (dq1[3] < 0) { dq1[0] = -dq1[0]; dq1[1] = -dq1[1]; dq1[2] = -dq1[2]; dq1[3] = -dq1[3]; }
    //std::cout << "dq1 = "<<dq1<<std::endl;
    Vec<3,Real> V1;
    double half_theta1 = acos(dq1[3]);
    //std::cout << i<<" qa_inv = "<<qa_inv<<" qb = "<<qb<<" dq = "<<dq<<" theta = "<<2*half_theta<<std::endl;
    if (half_theta1 > 0.0000001) // || half_theta1 < -0.0000001)
        V1 = Vec<3,Real>(dq1[0],dq1[1],dq1[2])*(2*half_theta1/sin(half_theta1));
    //std::cout << "V1 = "<<V1<<std::endl;

    V2 = lambda.multTranspose(V2);
    V1 = lambda.multTranspose(V1);

    depl[3] = V1[0];
    depl[4] = V1[1];
    depl[5] = V1[2];
    depl[9] = V2[0];
    depl[10] = V2[1];
    depl[11] = V2[2];

    Displacement force = _stiffnessMatrices[i] * depl;

    // Apply lambda transpose

    Vec3d fa1 = lambda*(Vec3d(force[0],force[1],force[2]));
    Vec3d fa2 = lambda*(Vec3d(force[3],force[4],force[5]));
    //Vec3d fa2 = Vec3d(force[3],force[4],force[5]);
    Vec3d fb1 = lambda*(Vec3d(force[6],force[7],force[8]));
    Vec3d fb2 = lambda*(Vec3d(force[9],force[10],force[11]));
    //Vec3d fb2 = Vec3d(force[9],force[10],force[11]);

    f[a] += Deriv(-fa1,-fa2);
    f[b] += Deriv(-fb1,-fb2);

}

template<class DataTypes>
void BeamFEMForceField<DataTypes>::applyStiffnessLarge( VecDeriv& f, const VecDeriv& x, int i, Index a, Index b )
{
    Transformation& lambda = _rotations[i];

    // Apply lambda

    Displacement depl;

    Vec3d U;
    U = lambda.multTranspose(x[a].getVCenter()); //Vec3d(x[a][0],x[a][1], x[a][2]);
    depl[0] = U[0];
    depl[1] = U[1];
    depl[2] = U[2];
    U = lambda.multTranspose(x[a].getVOrientation()); //Vec3d(x[a][3],x[a][4], x[a][5]);
    //U = Vec3d(x[a][3],x[a][4], x[a][5]);
    depl[3] = U[0];
    depl[4] = U[1];
    depl[5] = U[2];
    U = lambda.multTranspose(x[b].getVCenter()); //Vec3d(x[b][0],x[b][1], x[b][2]);
    depl[6] = U[0];
    depl[7] = U[1];
    depl[8] = U[2];
    U = lambda.multTranspose(x[b].getVOrientation()); //Vec3d(x[b][3],x[b][4], x[b][5]);
    //U = Vec3d(x[b][3],x[b][4], x[b][5]);
    depl[9] = U[0];
    depl[10] = U[1];
    depl[11] = U[2];

    Displacement force = _stiffnessMatrices[i] * depl;

    // Apply lambda transpose

    Vec3d fa1 = lambda*(Vec3d(force[0],force[1],force[2]));
    Vec3d fa2 = lambda*(Vec3d(force[3],force[4],force[5]));
    //Vec3d fa2 = Vec3d(force[3],force[4],force[5]);
    Vec3d fb1 = lambda*(Vec3d(force[6],force[7],force[8]));
    Vec3d fb2 = lambda*(Vec3d(force[9],force[10],force[11]));
    //Vec3d fb2 = Vec3d(force[9],force[10],force[11]);

    f[a] += Deriv(-fa1,-fa2);
    f[b] += Deriv(-fb1,-fb2);
}

template<class DataTypes>
void BeamFEMForceField<DataTypes>::draw()
{
    if (!getContext()->getShowForceFields()) return;
    if (!this->mstate) return;

    const VecCoord& x = *this->mstate->getX();

    glDisable(GL_LIGHTING);

    glBegin(GL_LINES);
    typename VecElement::const_iterator it;
    int i;
    for(it = _indexedElements->begin(), i = 0 ; it != _indexedElements->end() ; ++it, ++i)
    {
        Index a = (*it)[0];
        Index b = (*it)[1];
        defaulttype::Vec3d p; p = (x[a].getCenter()+x[b].getCenter())*0.5;
        //defaulttype::Quat quat; quat.fromMatrix(_rotations[i]);
        double len = _initialLength[i]*0.5;
        const Transformation& R = _rotations[i];
        glColor3f(1,0,0);
        helper::gl::glVertexT(p - R.col(0)*len);
        helper::gl::glVertexT(p + R.col(0)*len);
        glColor3f(0,1,0);
        helper::gl::glVertexT(p); // - R.col(1)*len);
        helper::gl::glVertexT(p + R.col(1)*len);
        glColor3f(0,0,1);
        helper::gl::glVertexT(p); // - R.col(2)*len);
        helper::gl::glVertexT(p + R.col(2)*len);
    }
    glEnd();
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
