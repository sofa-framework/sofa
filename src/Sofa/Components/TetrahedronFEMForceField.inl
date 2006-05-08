#ifndef SOFA_COMPONENTS_TETRAHEDRONFEMFORCEFIELD_INL
#define SOFA_COMPONENTS_TETRAHEDRONFEMFORCEFIELD_INL

#include "TetrahedronFEMForceField.h"
#include "Scene.h"
#include "MeshTopology.h"
#include "Common/PolarDecompose.h"
#include "GL/template.h"
#include <assert.h>
#include <iostream>

#include <GL/gl.h>

namespace Sofa
{

namespace Components
{

using namespace Common;

template <class DataTypes>
void TetrahedronFEMForceField<DataTypes>::init()
{

    _mesh = dynamic_cast<Sofa::Components::MeshTopology*>(this->object->getTopology());
    if (_mesh==NULL || (_mesh->getTetras().empty() && _mesh->getNbCubes()<=0))
    {
        std::cerr << "ERROR(TetrahedronFEMForceField): object must have a tetrahedric MeshTopology.\n";
        return;
    }
    if (!_mesh->getTetras().empty())
    {
        _indexedElements = & (_mesh->getTetras());
    }
    else
    {
        MeshTopology::SeqTetras* tetras = new MeshTopology::SeqTetras;
        int nbcubes = _mesh->getNbCubes();
        tetras->reserve(nbcubes*6);
        for (int i=0; i<nbcubes; i++)
        {
            MeshTopology::Cube c = _mesh->getCube(i);
            tetras->push_back(make_array(c[0],c[5],c[1],c[7]));
            tetras->push_back(make_array(c[0],c[1],c[2],c[7]));
            tetras->push_back(make_array(c[1],c[2],c[3],c[7]));
            tetras->push_back(make_array(c[7],c[2],c[6],c[0]));
            tetras->push_back(make_array(c[7],c[6],c[5],c[0]));
            tetras->push_back(make_array(c[6],c[5],c[4],c[0]));
        }
        _indexedElements = tetras;
    }

    VecCoord& p = *this->object->getX();
    _initialPoints = p;

    _strainDisplacements.resize( _indexedElements->size() );
    _stiffnesses.resize( _initialPoints.size()*3 );
    _materialsStiffnesses.resize(_indexedElements->size() );
    _forces.resize( _initialPoints.size() );

    switch(_method)
    {
    case SMALL :
    {
        unsigned int i=0;
        typename VecElement::const_iterator it;
        for(it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i)
        {
            Index a = (*it)[0];
            Index b = (*it)[1];
            Index c = (*it)[2];
            Index d = (*it)[3];
            this->computeMaterialStiffness(i,a,b,c,d);
            this->initSmall(i,a,b,c,d);
        }
        break;
    }
    case LARGE :
    {
        _rotations.resize( _indexedElements->size() );
        _rotatedInitialElements.resize(_indexedElements->size());
        unsigned int i=0;
        typename VecElement::const_iterator it;
        for(it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i)
        {
            Index a = (*it)[0];
            Index b = (*it)[1];
            Index c = (*it)[2];
            Index d = (*it)[3];
            computeMaterialStiffness(i,a,b,c,d);
            initLarge(i,a,b,c,d);
        }
        break;
    }
    case POLAR :
    {
        _rotations.resize( _indexedElements->size() );
        _rotatedInitialElements.resize(_indexedElements->size());
        _initialTransformation.resize(_indexedElements->size());
        unsigned int i=0;
        typename VecElement::const_iterator it;
        for(it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i)
        {
            Index a = (*it)[0];
            Index b = (*it)[1];
            Index c = (*it)[2];
            Index d = (*it)[3];
            computeMaterialStiffness(i,a,b,c,d);
            initPolar(i,a,b,c,d);
        }
        break;
    }
    }
    std::cout << "TetrahedronFEMForceField: init OK, "<<_indexedElements->size()<<" tetra."<<std::endl;
}


template<class DataTypes>
void TetrahedronFEMForceField<DataTypes>::addForce()
{
    assert(this->object);
    VecDeriv& f = *this->object->getF();
    VecCoord& p = *this->object->getX();
    //VecDeriv& v = *this->object->getV();
    f.resize(p.size());

    unsigned int i=0;
    typename VecElement::const_iterator it;

    if(_dampingRatio!=0)
        if(_method==SMALL)
            for(it=_indexedElements->begin(); it!=_indexedElements->end(); ++it,++i)
            {
                accumulateForceSmall( f, p, it, i );
                //accumulateDampingSmall( f, i );
            }
        else if(_method==LARGE)
            for(it=_indexedElements->begin(); it!=_indexedElements->end(); ++it,++i)
            {
                accumulateForceLarge( f, p, it, i );
                //accumulateDampingLarge( f, i );
            }
        else
            for(it=_indexedElements->begin(); it!=_indexedElements->end(); ++it,++i)
            {
                accumulateForcePolar( f, p, it, i );
                //accumulateDampingPolar( f, i );
            }
    else if(_method==SMALL)
        for(it=_indexedElements->begin(); it!=_indexedElements->end(); ++it,++i)
            accumulateForceSmall( f, p, it, i );
    else if(_method==LARGE)
        for(it=_indexedElements->begin(); it!=_indexedElements->end(); ++it,++i)
            accumulateForceLarge( f, p, it, i );
    else
        for(it=_indexedElements->begin(); it!=_indexedElements->end(); ++it,++i)
            accumulateForcePolar( f, p, it, i );

//		_forces = f;
//		cerr<<"----------------\nf : "<<f<<endl;

}

template<class DataTypes>
void TetrahedronFEMForceField<DataTypes>::addDForce()
{
    Real h=1;
    assert(this->object);
    VecDeriv& v = *this->object->getF();
    VecDeriv& x = *this->object->getDx();
    v.resize(x.size());
    //if(_assembling) applyStiffnessAssembled(v,h,x);
    //else
    {
        unsigned int i=0;
        typename VecElement::const_iterator it;

        switch(_method)
        {
        case SMALL :
        {
            for(it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i)
            {
                Index a = (*it)[0];
                Index b = (*it)[1];
                Index c = (*it)[2];
                Index d = (*it)[3];

                applyStiffnessSmall( v,h,x, i, a,b,c,d );
            }
            break;
        }
        case LARGE :
        {
            for(it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i)
            {
                Index a = (*it)[0];
                Index b = (*it)[1];
                Index c = (*it)[2];
                Index d = (*it)[3];

                applyStiffnessLarge( v,h,x, i, a,b,c,d );
            }
            break;
        }
        case POLAR :
        {
            for(it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i)
            {
                Index a = (*it)[0];
                Index b = (*it)[1];
                Index c = (*it)[2];
                Index d = (*it)[3];

                applyStiffnessPolar( v,h,x, i, a,b,c,d );
            }
            break;
        }
        }
    }
}



/*
template <typename V, typename R, typename E>
void TetrahedronFEMForceField<V,R,E>::applyStiffnessAssembled( Vector& v, Real h, const Vector& x )
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
			v[i][k] += (-h*val);
		}
	}
}
*/

template<class DataTypes>
void TetrahedronFEMForceField<DataTypes>::computeStrainDisplacement( StrainDisplacement &J, Coord a, Coord b, Coord c, Coord d )
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
typename TetrahedronFEMForceField<DataTypes>::Real TetrahedronFEMForceField<DataTypes>::peudo_determinant_for_coef ( const Mat<2, 3, Real>&  M )
{
    return  M[0][1]*M[1][2] - M[1][1]*M[0][2] -  M[0][0]*M[1][2] + M[1][0]*M[0][2] + M[0][0]*M[1][1] - M[1][0]*M[0][1];
}

template<class DataTypes>
void TetrahedronFEMForceField<DataTypes>::computeStiffnessMatrix( StiffnessMatrix& S,StiffnessMatrix& SR,const MaterialStiffness &K, const StrainDisplacement &J, const Transformation& Rot )
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

template<class DataTypes>
void TetrahedronFEMForceField<DataTypes>::computeMaterialStiffness(int i, Index&a, Index&b, Index&c, Index&d)
{

    _materialsStiffnesses[i][0][0] = _materialsStiffnesses[i][1][1] = _materialsStiffnesses[i][2][2] = 1;
    _materialsStiffnesses[i][0][1] = _materialsStiffnesses[i][0][2] = _materialsStiffnesses[i][1][0]
            = _materialsStiffnesses[i][1][2] = _materialsStiffnesses[i][2][0] =
                    _materialsStiffnesses[i][2][1] = _poissonRatio/(1-_poissonRatio);
    _materialsStiffnesses[i][0][3] = _materialsStiffnesses[i][0][4] =	_materialsStiffnesses[i][0][5] = 0;
    _materialsStiffnesses[i][1][3] = _materialsStiffnesses[i][1][4] =	_materialsStiffnesses[i][1][5] = 0;
    _materialsStiffnesses[i][2][3] = _materialsStiffnesses[i][2][4] =	_materialsStiffnesses[i][2][5] = 0;
    _materialsStiffnesses[i][3][0] = _materialsStiffnesses[i][3][1] = _materialsStiffnesses[i][3][2] = _materialsStiffnesses[i][3][4] =	_materialsStiffnesses[i][3][5] = 0;
    _materialsStiffnesses[i][4][0] = _materialsStiffnesses[i][4][1] = _materialsStiffnesses[i][4][2] = _materialsStiffnesses[i][4][3] =	_materialsStiffnesses[i][4][5] = 0;
    _materialsStiffnesses[i][5][0] = _materialsStiffnesses[i][5][1] = _materialsStiffnesses[i][5][2] = _materialsStiffnesses[i][5][3] =	_materialsStiffnesses[i][5][4] = 0;
    _materialsStiffnesses[i][3][3] = _materialsStiffnesses[i][4][4] = _materialsStiffnesses[i][5][5] = (1-2*_poissonRatio)/(2*(1-_poissonRatio));
    _materialsStiffnesses[i] *= (_youngModulus*(1-_poissonRatio))/((1+_poissonRatio)*(1-2*_poissonRatio));



    /*Real gamma = (_youngModulus*_poissonRatio) / ((1+_poissonRatio)*(1-2*_poissonRatio));
    Real 		mu2 = _youngModulus / (1+_poissonRatio);
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

    Coord A = _initialPoints[b] - _initialPoints[a];
    Coord B = _initialPoints[c] - _initialPoints[a];
    Coord C = _initialPoints[d] - _initialPoints[a];
    Coord AB = cross(A, B);
    //v_eq_cross( AB, A, B );
    Real volumes6 = fabs( dot( AB, C ) );
    if (volumes6<0)
    {
        std::cerr << "ERROR: Negative volume for tetra "<<i<<" <"<<a<<','<<b<<','<<c<<','<<d<<"> = "<<volumes6/6<<std::endl;
    }
    _materialsStiffnesses[i] /= volumes6*volumes6;
}

template<class DataTypes>
void TetrahedronFEMForceField<DataTypes>::computeForce( Displacement &F, const Displacement &Depl, const MaterialStiffness &K, const StrainDisplacement &J )
{
    //Mat<6, 12, Real> Jt;
    //Jt.transpose(J);
    //F = J*(K*(Jt*Depl));

    /* We have:
    K[0][3] = K[0][4] =	K[0][5] = 0;
    K[1][3] = K[1][4] =	K[1][5] = 0;
    K[2][3] = K[2][4] =	K[2][5] = 0;
    K[3][0] = K[3][1] = K[3][2] = K[3][4] =	K[3][5] = 0;
    K[4][0] = K[4][1] = K[4][2] = K[4][3] =	K[4][5] = 0;
    K[5][0] = K[5][1] = K[5][2] = K[5][3] =	K[5][4] = 0;

    J[0][1] = J[0][2] = J[0][4] =
    J[1][0] = J[1][2] = J[1][5] =
    J[2][0] = J[2][1] = J[2][3] = 0;
    J[3][1] = J[3][2] = J[3][4] =
    J[4][0] = J[4][2] = J[4][5] =
    J[5][0] = J[5][1] = J[5][3] = 0;
    J[6][1] = J[6][2] = J[6][4] =
    J[7][0] = J[7][2] = J[7][5] =
    J[8][0] = J[8][1] = J[8][3] = 0;
    J[9][1] = J[9][2] = J[9][4] =
    J[10][0] = J[10][2] = J[10][5] =
    J[11][0] = J[11][1] = J[11][3] = 0;
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
    JtD[5] =   J[ 0][5]*Depl[ 0]+  J[ 1][5]*Depl[ 1]+/*J[ 2][5]*Depl[ 2]+*/
            J[ 3][5]*Depl[ 3]+  J[ 4][5]*Depl[ 4]+/*J[ 5][5]*Depl[ 5]+*/
            J[ 6][5]*Depl[ 6]+  J[ 7][5]*Depl[ 7]+/*J[ 8][5]*Depl[ 8]+*/
            J[ 9][5]*Depl[ 9]+  J[10][5]*Depl[10] /*J[11][5]*Depl[11]*/;

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

////////////// small displacements method
template<class DataTypes>
void TetrahedronFEMForceField<DataTypes>::initSmall(int i, Index&a, Index&b, Index&c, Index&d)
{
    computeStrainDisplacement( _strainDisplacements[i], _initialPoints[a], _initialPoints[b], _initialPoints[c], _initialPoints[d] );
}

template<class DataTypes>
void TetrahedronFEMForceField<DataTypes>::accumulateForceSmall( Vector& f, const Vector & p, typename VecElement::const_iterator elementIt, Index elementIndex )
{
    Element index = *elementIt;
    Index a = index[0];
    Index b = index[1];
    Index c = index[2];
    Index d = index[3];

    // displacements
    Displacement D;
    D[0] = 0;
    D[1] = 0;
    D[2] = 0;
    D[3] =  _initialPoints[b][0] - _initialPoints[a][0] - p[b][0]+p[a][0];
    D[4] =  _initialPoints[b][1] - _initialPoints[a][1] - p[b][1]+p[a][1];
    D[5] =  _initialPoints[b][2] - _initialPoints[a][2] - p[b][2]+p[a][2];
    D[6] =  _initialPoints[c][0] - _initialPoints[a][0] - p[c][0]+p[a][0];
    D[7] =  _initialPoints[c][1] - _initialPoints[a][1] - p[c][1]+p[a][1];
    D[8] =  _initialPoints[c][2] - _initialPoints[a][2] - p[c][2]+p[a][2];
    D[9] =  _initialPoints[d][0] - _initialPoints[a][0] - p[d][0]+p[a][0];
    D[10] = _initialPoints[d][1] - _initialPoints[a][1] - p[d][1]+p[a][1];
    D[11] = _initialPoints[d][2] - _initialPoints[a][2] - p[d][2]+p[a][2];

    // compute force on element
    Displacement F;

    if(!_assembling)
    {
        computeForce( F, D, _materialsStiffnesses[elementIndex], _strainDisplacements[elementIndex] );
    }
    else
    {
        Transformation Rot;
        Rot[0][0]=Rot[1][1]=Rot[2][2]=1;
        Rot[0][1]=Rot[0][2]=0;
        Rot[1][0]=Rot[1][2]=0;
        Rot[2][0]=Rot[2][1]=0;


        StiffnessMatrix JKJt,tmp;
        computeStiffnessMatrix(JKJt,tmp,_materialsStiffnesses[elementIndex], _strainDisplacements[elementIndex],Rot);

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
            int row = index[i/3]*3+i%3;
            /// \todo PAS CLAIR
            //_stiffnesses[row].resize(0);

            for(int j=0; j<12; ++j)
            {
                if(JKJt[i][j]!=0)
                {

                    int col = index[j/3]*3+j%3;
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
void TetrahedronFEMForceField<DataTypes>::accumulateDampingSmall( Vector& /*f*/, Index /*elementIndex*/ )
{
    std::cerr << "TODO\n";
}

template<class DataTypes>
void TetrahedronFEMForceField<DataTypes>::applyStiffnessSmall( Vector& f, Real h, const Vector& x, int i, Index a, Index b, Index c, Index d )
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
    computeForce( F, X, _materialsStiffnesses[i], _strainDisplacements[i] );

    f[a] += Deriv( -h*F[0], -h*F[1],  -h*F[2] );
    f[b] += Deriv( -h*F[3], -h*F[4],  -h*F[5] );
    f[c] += Deriv( -h*F[6], -h*F[7],  -h*F[8] );
    f[d] += Deriv( -h*F[9], -h*F[10], -h*F[11] );
}

////////////// large displacements method
template<class DataTypes>
void TetrahedronFEMForceField<DataTypes>::initLarge(int i, Index&a, Index&b, Index&c, Index&d)
{
    // Rotation matrix (initial Tetrahedre/world)
    // first vector on first edge
    // second vector in the plane of the two first edges
    // third vector orthogonal to first and second
    Transformation R_0_1;
    computeRotationLarge( R_0_1, _initialPoints, a, b, c);

    _rotatedInitialElements[i][0] = R_0_1*_initialPoints[a];
    _rotatedInitialElements[i][1] = R_0_1*_initialPoints[b];
    _rotatedInitialElements[i][2] = R_0_1*_initialPoints[c];
    _rotatedInitialElements[i][3] = R_0_1*_initialPoints[d];

// 		cerr<<"a,b,c : "<<a<<" "<<b<<" "<<c<<endl;
// 		cerr<<"_initialPoints : "<<_initialPoints<<endl;
// 		cerr<<"R_0_1 large : "<<R_0_1<<endl;

    _rotatedInitialElements[i][1] -= _rotatedInitialElements[i][0];
    _rotatedInitialElements[i][2] -= _rotatedInitialElements[i][0];
    _rotatedInitialElements[i][3] -= _rotatedInitialElements[i][0];
    _rotatedInitialElements[i][0] = Coord(0,0,0);


// 		cerr<<"_rotatedInitialElements : "<<_rotatedInitialElements<<endl;

    computeStrainDisplacement( _strainDisplacements[i],_rotatedInitialElements[i][0], _rotatedInitialElements[i][1],_rotatedInitialElements[i][2],_rotatedInitialElements[i][3] );
}

template<class DataTypes>
void TetrahedronFEMForceField<DataTypes>::computeRotationLarge( Transformation &r, const Vector &p, const Index &a, const Index &b, const Index &c)
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
void TetrahedronFEMForceField<DataTypes>::accumulateForceLarge( Vector& f, const Vector & p, typename VecElement::const_iterator elementIt, Index elementIndex )
{
    Element index = *elementIt;

    // Rotation matrix (deformed and displaced Tetrahedron/world)
    Transformation R_0_2;
    computeRotationLarge( R_0_2, p, index[0],index[1],index[2]);
    _rotations[elementIndex].transpose(R_0_2);
    //cerr<<"R_0_2 large : "<<R_0_2<<endl;

    // positions of the deformed and displaced Tetrahedre in its frame
    fixed_array<Coord,4> deforme;
    for(int i=0; i<4; ++i)
        deforme[i] = R_0_2*p[index[i]];

    deforme[1][0] -= deforme[0][0];
    deforme[2][0] -= deforme[0][0];
    deforme[2][1] -= deforme[0][1];
    deforme[3] -= deforme[0];

    // displacement
    Displacement D;
    D[0] = 0;
    D[1] = 0;
    D[2] = 0;
    D[3] = _rotatedInitialElements[elementIndex][1][0] - deforme[1][0];
    D[4] = 0;
    D[5] = 0;
    D[6] = _rotatedInitialElements[elementIndex][2][0] - deforme[2][0];
    D[7] = _rotatedInitialElements[elementIndex][2][1] - deforme[2][1];
    D[8] = 0;
    D[9] = _rotatedInitialElements[elementIndex][3][0] - deforme[3][0];
    D[10] = _rotatedInitialElements[elementIndex][3][1] - deforme[3][1];
    D[11] =_rotatedInitialElements[elementIndex][3][2] - deforme[3][2];

    //cerr<<"D : "<<D<<endl;


    Displacement F;
    if(_updateStiffnessMatrix)
    {
        _strainDisplacements[elementIndex][0][0]   = ( - deforme[2][1]*deforme[3][2] );
        _strainDisplacements[elementIndex][1][1] = ( deforme[2][0]*deforme[3][2] - deforme[1][0]*deforme[3][2] );
        _strainDisplacements[elementIndex][2][2]   = ( deforme[2][1]*deforme[3][0] - deforme[2][0]*deforme[3][1] + deforme[1][0]*deforme[3][1] - deforme[1][0]*deforme[2][1] );

        _strainDisplacements[elementIndex][3][0]   = ( deforme[2][1]*deforme[3][2] );
        _strainDisplacements[elementIndex][4][1]  = ( - deforme[2][0]*deforme[3][2] );
        _strainDisplacements[elementIndex][5][2]   = ( - deforme[2][1]*deforme[3][0] + deforme[2][0]*deforme[3][1] );

        _strainDisplacements[elementIndex][7][1]  = ( deforme[1][0]*deforme[3][2] );
        _strainDisplacements[elementIndex][8][2]   = ( - deforme[1][0]*deforme[3][1] );

        _strainDisplacements[elementIndex][11][2] = ( deforme[1][0]*deforme[2][1] );


    }


    if(!_assembling)
    {
        // compute force on element
        computeForce( F, D, _materialsStiffnesses[elementIndex], _strainDisplacements[elementIndex]);

        //computeForceOptimized( F, D, _materialsStiffnesses[elementIndex], _strainDisplacements[elementIndex]);

        for(int i=0; i<12; i+=3)
            f[index[i/3]] += _rotations[elementIndex] * Deriv( F[i], F[i+1],  F[i+2] );

        //cerr<<"p large : "<<p<<endl;
        //cerr<<"F large : "<<f<<endl;
// 		for(int i=0;i<12;i+=3)
// 		{
// 			Vec tmp;
// 			v_eq_Ab( tmp, _rotations[elementIndex], Vec( F[i], F[i+1],  F[i+2] ) );
// 			cerr<<tmp<<"\t";
// 		}
// 		cerr<<endl;
    }
    else
    {
        _strainDisplacements[elementIndex][6][0] = 0;
        _strainDisplacements[elementIndex][9][0] = 0;
        _strainDisplacements[elementIndex][10][1] = 0;

        StiffnessMatrix RJKJt, RJKJtRt;
        computeStiffnessMatrix(RJKJt,RJKJtRt,_materialsStiffnesses[elementIndex], _strainDisplacements[elementIndex],_rotations[elementIndex]);


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
            int row = index[i/3]*3+i%3;

            for(int j=0; j<12; ++j)
            {
                int col = index[j/3]*3+j%3;

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
            f[index[i/3]] += Deriv( F[i], F[i+1],  F[i+2] );
    }
}

template<class DataTypes>
void TetrahedronFEMForceField<DataTypes>::accumulateDampingLarge( Vector& /*f*/, Index /*elementIndex*/ )
{
    std::cerr << "TODO\n";
}

template<class DataTypes>
void TetrahedronFEMForceField<DataTypes>::applyStiffnessLarge( Vector& f, Real h, const Vector& x, int i, Index a, Index b, Index c, Index d )
{
    Transformation R_0_2;
    R_0_2.transpose(_rotations[i]);

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

    computeForce( F, X, _materialsStiffnesses[i], _strainDisplacements[i] );
    //computeForce( F, X, _materialsStiffnesses[i],  _strainDisplacements[i][0][0],_strainDisplacements[i][1][1],_strainDisplacements[i][2][2],_strainDisplacements[i][3][0],_strainDisplacements[i][4][1],_strainDisplacements[i][5][2],_strainDisplacements[i][7][1],_strainDisplacements[i][8][2],_strainDisplacements[i][11][2]);

    //cerr<<"F : "<<F<<endl;

    f[a] += _rotations[i] * Deriv( -h*F[0], -h*F[1],  -h*F[2] );
    f[b] += _rotations[i] * Deriv( -h*F[3], -h*F[4],  -h*F[5] );
    f[c] += _rotations[i] * Deriv( -h*F[6], -h*F[7],  -h*F[8] );
    f[d] += _rotations[i] * Deriv( -h*F[9], -h*F[10], -h*F[11] );
}

////////////// polar decomposition method

template<class DataTypes>
void TetrahedronFEMForceField<DataTypes>::initPolar(int i, Index& a, Index&b, Index&c, Index&d)
{
    Transformation A;
    A[0] = _initialPoints[b]-_initialPoints[a];
    A[1] = _initialPoints[c]-_initialPoints[a];
    A[2] = _initialPoints[d]-_initialPoints[a];
    _initialTransformation[i] = A;

    Transformation R_0_1;
    Mat<3,3,Real> S;
    polar_decomp(A, R_0_1, S);

    _rotatedInitialElements[i][0] = R_0_1*_initialPoints[a];
    _rotatedInitialElements[i][1] = R_0_1*_initialPoints[b];
    _rotatedInitialElements[i][2] = R_0_1*_initialPoints[c];
    _rotatedInitialElements[i][3] = R_0_1*_initialPoints[d];

    computeStrainDisplacement( _strainDisplacements[i],_rotatedInitialElements[i][0], _rotatedInitialElements[i][1],_rotatedInitialElements[i][2],_rotatedInitialElements[i][3] );
}

template<class DataTypes>
void TetrahedronFEMForceField<DataTypes>::accumulateForcePolar( Vector& f, const Vector & p, typename VecElement::const_iterator elementIt, Index elementIndex )
{
    Element index = *elementIt;

    Transformation A;
    A[0] = p[index[1]]-p[index[0]];
    A[1] = p[index[2]]-p[index[0]];
    A[2] = p[index[3]]-p[index[0]];

    Transformation R_0_2;
    Mat<3,3,Real> S;
    polar_decomp(A, R_0_2, S);

    _rotations[elementIndex].transpose( R_0_2 );

    // positions of the deformed and displaced Tetrahedre in its frame
    fixed_array<Coord, 4>  deforme;
    for(int i=0; i<4; ++i)
        deforme[i] = R_0_2 * p[index[i]];

    // displacement
    Displacement D;
    D[0] = _rotatedInitialElements[elementIndex][0][0] - deforme[0][0];
    D[1] = _rotatedInitialElements[elementIndex][0][1] - deforme[0][1];
    D[2] = _rotatedInitialElements[elementIndex][0][2] - deforme[0][2];
    D[3] = _rotatedInitialElements[elementIndex][1][0] - deforme[1][0];
    D[4] = _rotatedInitialElements[elementIndex][1][1] - deforme[1][1];
    D[5] = _rotatedInitialElements[elementIndex][1][2] - deforme[1][2];
    D[6] = _rotatedInitialElements[elementIndex][2][0] - deforme[2][0];
    D[7] = _rotatedInitialElements[elementIndex][2][1] - deforme[2][1];
    D[8] = _rotatedInitialElements[elementIndex][2][2] - deforme[2][2];
    D[9] = _rotatedInitialElements[elementIndex][3][0] - deforme[3][0];
    D[10] = _rotatedInitialElements[elementIndex][3][1] - deforme[3][1];
    D[11] = _rotatedInitialElements[elementIndex][3][2] - deforme[3][2];
    //cerr<<"D : "<<D<<endl;


    Displacement F;
    if(_updateStiffnessMatrix)
    {
        // shape functions matrix
        computeStrainDisplacement( _strainDisplacements[elementIndex], deforme[0],deforme[1],deforme[2],deforme[3]  );
        //computeForceOptimized( F, D, _materialsStiffnesses[elementIndex], _strainDisplacements[elementIndex] );
    }


    if(!_assembling)
    {
        computeForce( F, D, _materialsStiffnesses[elementIndex], _strainDisplacements[elementIndex] );

        for(int i=0; i<12; i+=3)
            f[index[i/3]] += _rotations[elementIndex] * Deriv( F[i], F[i+1],  F[i+2] );
    }
    else
    {

    }
}

template<class DataTypes>
void TetrahedronFEMForceField<DataTypes>::applyStiffnessPolar( Vector& f, Real h, const Vector& x, int i, Index a, Index b, Index c, Index d )
{
    Transformation R_0_2;
    R_0_2.transpose( _rotations[i] );

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

    //computeForceOptimized( F, X, _materialsStiffnesses[i], _strainDisplacements[i] );
    computeForce( F, X, _materialsStiffnesses[i], _strainDisplacements[i] );

    //cerr<<"F : "<<F<<endl;

    f[a] += _rotations[i] * Deriv( -h*F[0], -h*F[1],  -h*F[2] );
    f[b] += _rotations[i] * Deriv( -h*F[3], -h*F[4],  -h*F[5] );
    f[c] += _rotations[i] * Deriv( -h*F[6], -h*F[7],  -h*F[8] );
    f[d] += _rotations[i] * Deriv( -h*F[9], -h*F[10], -h*F[11] );
}

template<class DataTypes>
void TetrahedronFEMForceField<DataTypes>::draw()
{
    if (!Scene::getInstance()->getShowForceFields()) return;
    if (!this->object) return;
    VecCoord& x = *this->object->getX();

    glDisable(GL_LIGHTING);

    glBegin(GL_TRIANGLES);
    typename VecElement::const_iterator it;
    for(it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it)
    {
        Index a = (*it)[0];
        Index b = (*it)[1];
        Index c = (*it)[2];
        Index d = (*it)[3];
        Coord center = (x[a]+x[b]+x[c]+x[d])*0.125;
        Coord pa = (x[a]+center)*0.666667;
        Coord pb = (x[b]+center)*0.666667;
        Coord pc = (x[c]+center)*0.666667;
        Coord pd = (x[d]+center)*0.666667;

        glColor4f(0,0,1,1);
        GL::glVertexT(pa);
        GL::glVertexT(pb);
        GL::glVertexT(pc);

        glColor4f(0,0.5,1,1);
        GL::glVertexT(pb);
        GL::glVertexT(pc);
        GL::glVertexT(pd);

        glColor4f(0,1,1,1);
        GL::glVertexT(pc);
        GL::glVertexT(pd);
        GL::glVertexT(pa);

        glColor4f(0.5,1,1,1);
        GL::glVertexT(pd);
        GL::glVertexT(pa);
        GL::glVertexT(pb);
    }
    glEnd();
}

} // namespace Components

} // namespace Sofa

#endif
