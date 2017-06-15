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
#ifndef SOFA_COMPONENT_FORCEFIELD_HEXAHEDRONFEMFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_HEXAHEDRONFEMFORCEFIELD_INL

#include "HexahedronFEMForceField.h"
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/helper/decompose.h>
#include <sofa/helper/gl/template.h>
#include <assert.h>
#include <iostream>
#include <set>







// WARNING: indices ordering is different than in topology node
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

using std::set;
using namespace sofa::defaulttype;

#ifndef SOFA_NEW_HEXA
template<class DataTypes> const int HexahedronFEMForceField<DataTypes>::_indices[8] = {0,1,3,2,4,5,7,6};
// template<class DataTypes> const int HexahedronFEMForceField<DataTypes>::_indices[8] = {4,5,7,6,0,1,3,2};
#endif


template <class DataTypes>
void HexahedronFEMForceField<DataTypes>::init()
{
    if(_alreadyInit)return;
    else _alreadyInit=true;

    this->core::behavior::ForceField<DataTypes>::init();
    if( this->getContext()->getMeshTopology()==NULL )
    {
        serr << "ERROR(HexahedronFEMForceField): object must have a Topology."<<sendl;
        return;
    }

    _mesh = dynamic_cast<sofa::core::topology::BaseMeshTopology*>(this->getContext()->getMeshTopology());
    if ( _mesh==NULL)
    {
        serr << "ERROR(HexahedronFEMForceField): object must have a MeshTopology."<<sendl;
        return;
    }
#ifdef SOFA_NEW_HEXA
    else if( _mesh->getNbHexahedra()<=0 )
#else
    else if( _mesh->getNbCubes()<=0 )
#endif
    {
        serr << "ERROR(HexahedronFEMForceField): object must have a hexahedric MeshTopology."<<sendl;
        serr << _mesh->getName()<<sendl;
        serr << _mesh->getTypeName()<<sendl;
        serr<<_mesh->getNbPoints()<<sendl;
        return;
    }
    _sparseGrid = dynamic_cast<topology::SparseGridTopology*>(_mesh);
    m_potentialEnergy = 0;



// 	if( _elementStiffnesses.getValue().empty() )
// 		_elementStiffnesses.beginEdit()->resize(this->getIndexedElements()->size());
    // 	_stiffnesses.resize( _initialPoints.getValue().size()*3 ); // assembly ?


    reinit();



// 	unsigned int i=0;
// 	typename VecElement::const_iterator it;
// 	for(it = this->getIndexedElements()->begin() ; it != this->getIndexedElements()->end() ; ++it, ++i)
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
void HexahedronFEMForceField<DataTypes>::reinit()
{

    //if (_initialPoints.getValue().size() == 0)
    //{
        const VecCoord& p = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
        _initialPoints.setValue(p);
    //}

    _materialsStiffnesses.resize(this->getIndexedElements()->size() );
    _rotations.resize( this->getIndexedElements()->size() );
    _rotatedInitialElements.resize(this->getIndexedElements()->size());
    _initialrotations.resize( this->getIndexedElements()->size() );

    if (f_method.getValue() == "large")
        this->setMethod(LARGE);
    else if (f_method.getValue() == "polar")
        this->setMethod(POLAR);
    else if (f_method.getValue() == "small")
        this->setMethod(SMALL);

    switch(method)
    {
    case LARGE :
    {
        unsigned int i=0;
        typename VecElement::const_iterator it;
        for(it = this->getIndexedElements()->begin() ; it != this->getIndexedElements()->end() ; ++it, ++i)
        {
            computeMaterialStiffness(i);
            initLarge(i,*it);
        }
        break;
    }
    case POLAR :
    {
        unsigned int i=0;
        typename VecElement::const_iterator it;
        for(it = this->getIndexedElements()->begin() ; it != this->getIndexedElements()->end() ; ++it, ++i)
        {
            computeMaterialStiffness(i);
            initPolar(i,*it);
        }
        break;
    }
    case SMALL :
    {
        unsigned int i=0;
        typename VecElement::const_iterator it;
        for(it = this->getIndexedElements()->begin() ; it != this->getIndexedElements()->end() ; ++it, ++i)
        {
            computeMaterialStiffness(i);
            initSmall(i,*it);
        }
        break;
    }
    }
}



/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////




template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::addForce (const core::MechanicalParams* /*mparams*/, DataVecDeriv& f, const DataVecCoord& p, const DataVecDeriv& /*v*/)
{
    WDataRefVecDeriv _f = f;
    RDataRefVecCoord _p = p;

    _f.resize(_p.size());

    if (needUpdateTopology)
    {
        reinit();
        needUpdateTopology = false;
    }

    unsigned int i=0;
    typename VecElement::const_iterator it;

    switch(method)
    {
    case LARGE :
    {
        m_potentialEnergy = 0;
        for(it=this->getIndexedElements()->begin(); it!=this->getIndexedElements()->end(); ++it,++i)
        {
            accumulateForceLarge( _f, _p, i, *it );
        }
        m_potentialEnergy/=-2.0;
        break;
    }
    case POLAR :
    {
        m_potentialEnergy = 0;
        for(it=this->getIndexedElements()->begin(); it!=this->getIndexedElements()->end(); ++it,++i)
        {
            accumulateForcePolar( _f, _p, i, *it );
        }
        m_potentialEnergy/=-2.0;
        break;
    }
    case SMALL :
    {
        m_potentialEnergy = 0;
        for(it=this->getIndexedElements()->begin(); it!=this->getIndexedElements()->end(); ++it,++i)
        {
            accumulateForceSmall( _f, _p, i, *it );
        }
        m_potentialEnergy/=-2.0;
        break;
    }
    }


}

template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::addDForce (const core::MechanicalParams *mparams, DataVecDeriv& v, const DataVecDeriv& x)
{
    WDataRefVecDeriv _df = v;
    RDataRefVecCoord _dx = x;
    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    if (_df.size() != _dx.size())
        _df.resize(_dx.size());

    unsigned int i = 0;
    typename VecElement::const_iterator it;

    for(it = this->getIndexedElements()->begin() ; it != this->getIndexedElements()->end() ; ++it, ++i)
    {
        // Transformation R_0_2;
        // R_0_2.transpose(_rotations[i]);

        Displacement X;

        for(int w=0; w<8; ++w)
        {
            Coord x_2;
#ifndef SOFA_NEW_HEXA
            x_2 = _rotations[i] * _dx[(*it)[_indices[w]]];
#else
            x_2 = _rotations[i] * _dx[(*it)[w]];
#endif
            X[w*3] = x_2[0];
            X[w*3+1] = x_2[1];
            X[w*3+2] = x_2[2];
        }

        Displacement F;
        computeForce( F, X, _elementStiffnesses.getValue()[i] );

        for(int w=0; w<8; ++w)
        {
#ifndef SOFA_NEW_HEXA
            _df[(*it)[_indices[w]]] -= _rotations[i].multTranspose(Deriv(F[w*3], F[w*3+1], F[w*3+2])) * kFactor;
#else
            _df[(*it)[w]] -= _rotations[i].multTranspose(Deriv(F[w*3], F[w*3+1], F[w*3+2])) * kFactor;
#endif
        }
    }
}

template <class DataTypes>
const typename HexahedronFEMForceField<DataTypes>::Transformation& HexahedronFEMForceField<DataTypes>::getElementRotation(const unsigned elemidx)
{
    return _rotations[elemidx];
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////




// enable to use generic matrix computing code instead of the original optimized code specific to parallelepipeds
#define GENERIC_STIFFNESS_MATRIX
// enable to use the full content of the MaterialStiffness matrix, instead of only the 3x3 upper bloc
#define MAT_STIFFNESS_USE_W
// enable to use J when computing qx/qy/qz, instead of computing the matrix relative to (x1,x2,x3) and pre/post multiply by J^-1 afterward.
// note that this does not matter if the element is a cube.
#define DN_USE_J


template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::computeElementStiffness( ElementStiffness &K, const MaterialStiffness &M, const helper::fixed_array<Coord,8> &nodes, const int elementIndice, double stiffnessFactor)
{
// 	serr<<"HexahedronFEMForceField<DataTypes>::computeElementStiffnessAsFinest"<<sendl;

    const bool verbose = elementIndice==0;
    // X = n0 (1-x1)(1-x2)(1-x3)/8 + n1 (1+x1)(1-x2)(1-x3)/8 + n2 (1+x1)(1+x2)(1-x3)/8 + n3 (1-x1)(1+x2)(1-x3)/8 + n4 (1-x1)(1-x2)(1+x3)/8 + n5 (1+x1)(1-x2)(1+x3)/8 + n6 (1+x1)(1+x2)(1+x3)/8 + n7 (1-x1)(1+x2)(1+x3)/8
    // J = [ DXi / xj ] = [ (n1i-n0i)(1-x2)(1-x3)/8+(n2i-n3i)(1+x2)(1-x3)/8+(n5i-n4i)(1-x2)(1+x3)/8+(n6i-n7i)(1+x2)(1+x3)/8  (n3i-n0i)(1-x1)(1-x3)/8+(n2i-n1i)(1+x1)(1-x3)/8+(n7i-n4i)(1-x1)(1+x3)/8+(n6i-n5i)(1+x1)(1+x3)/8  (n4i-n0i)(1-x1)(1-x2)/8+(n5i-n1i)(1+x1)(1-x2)/8+(n6i-n2i)(1+x1)(1+x2)/8+(n7i-n3i)(1-x1)(1+x2)/8 ]
    // if it is an orthogonal regular hexahedra: J = [ [ l0/2 0 0 ] [ 0 l1/2 0 ] [ 0 0 l2/2 ] ] det(J) = l0l1l2/8 = vol/8
    //
    // Ke = integralV(BtEBdv) = integral(BtEB det(J) dx1 dx2 dx3)
    // B = DN = [ qx 0  0  ]
    //          [ 0  qy 0  ]
    //          [ 0  0  qz ]
    //          [ qy qx 0  ]
    //          [ 0  qz qy ]
    //          [ qz 0  qx ]
    // with qx = [ dN1/dx ... dN8/dx ] qy = [ dN1/dy ... dN8/dy ] qz = [ dN1/dz ... dN8/dz ]
    // The submatrix Kij of K linking nodes i and j can then be computed as: Kij = integralV(Bjt E Bi det(J) dx1 dx2 dx3)
    // with Bi = part of B related to node i: Bi = [ [ dNi/dx 0 0 ] [ 0 dNi/dy 0 ] [ 0 0 dNi/dz ] [ dNi/dy dNi/dx 0 ] [ 0 dNi/dz dNi/dy ] [ dNi/dz 0 dNi/dx ] ]
    // This integral can be estimated using 8 gauss quadrature points (x1,x2,x3)=(+-1/sqrt(3),+-1/sqrt(3),+-sqrt(3))
    K.fill( 0.0 );
    Mat33 J; // J[i][j] = dXi/dxj
    Mat33 J_1; // J_1[i][j] = dxi/dXj
    Mat33 J_1t;
    Real detJ = (Real)1.0;
    // check if the hexaedra is a parallelepiped
    Coord lx = nodes[1]-nodes[0];
    Coord ly = nodes[3]-nodes[0];
    Coord lz = nodes[4]-nodes[0];
    bool isParallel = false;
    if ((nodes[3]+lx-nodes[2]).norm() < lx.norm()*0.001 && (nodes[0]+lz-nodes[4]).norm() < lz.norm()*0.001 && (nodes[1]+lz-nodes[5]).norm() < lz.norm()*0.001 && (nodes[2]+lz-nodes[6]).norm() < lz.norm()*0.001 && (nodes[3]+lz-nodes[7]).norm() < lz.norm()*0.001)
    {
        isParallel = true;
        for (int c=0; c<3; ++c)
        {
            J[c][0] = lx[c]/2;
            J[c][1] = ly[c]/2;
            J[c][2] = lz[c]/2;
        }
        detJ = defaulttype::determinant(J);
        J_1.invert(J);
        J_1t.transpose(J_1);

        dmsg_info_when(verbose) << "J = " << J << msgendl
                                << "invJ = "  << J_1 << msgendl
                                << "detJ = " << detJ << msgendl;
    }
//     else
//         sout << "Hexa "<<elementIndice<<" is NOT a parallelepiped."<<sendl;

    const Real U = M[0][0];
    const Real V = M[0][1];
#ifdef MAT_STIFFNESS_USE_W
    const Real W = M[3][3];
#else
    const Real W = M[2][2];
#endif
    const double inv_sqrt3 = 1.0/sqrt(3.0);
    for (int gx1=-1; gx1<=1; gx1+=2)
        for (int gx2=-1; gx2<=1; gx2+=2)
            for (int gx3=-1; gx3<=1; gx3+=2)
            {
                double x1 = gx1*inv_sqrt3;
                double x2 = gx2*inv_sqrt3;
                double x3 = gx3*inv_sqrt3;
                // compute jacobian matrix
                //Mat33 J; // J[i][j] = dXi/dxj
                //Mat33 J_1; // J_1[i][j] = dxi/dXj
                if (!isParallel)
                {
                    for (int c=0; c<3; ++c)
                    {
                        J[c][0] = (Real)( (nodes[1][c]-nodes[0][c])*(1-x2)*(1-x3)/8+(nodes[2][c]-nodes[3][c])*(1+x2)*(1-x3)/8+(nodes[5][c]-nodes[4][c])*(1-x2)*(1+x3)/8+(nodes[6][c]-nodes[7][c])*(1+x2)*(1+x3)/8);
                        J[c][1] =(Real)( (nodes[3][c]-nodes[0][c])*(1-x1)*(1-x3)/8+(nodes[2][c]-nodes[1][c])*(1+x1)*(1-x3)/8+(nodes[7][c]-nodes[4][c])*(1-x1)*(1+x3)/8+(nodes[6][c]-nodes[5][c])*(1+x1)*(1+x3)/8);
                        J[c][2] =(Real)( (nodes[4][c]-nodes[0][c])*(1-x1)*(1-x2)/8+(nodes[5][c]-nodes[1][c])*(1+x1)*(1-x2)/8+(nodes[6][c]-nodes[2][c])*(1+x1)*(1+x2)/8+(nodes[7][c]-nodes[3][c])*(1-x1)*(1+x2)/8);
                    }
                    detJ = defaulttype::determinant(J);
                    J_1.invert(J);
                    J_1t.transpose(J_1);

                    dmsg_info_when(verbose) << "J = " << J << msgendl
                                            << "invJ = "  << J_1 << msgendl
                                            << "detJ = " << detJ << msgendl;
                }
                Real qx[8];
                Real qy[8];
                Real qz[8];
                for(int i=0; i<8; ++i)
                {
                    // Ni = 1/8 (1+_coef[i][0]x1)(1+_coef[i][1]x2)(1+_coef[i][2]x3)
                    // qxi = dNi/dx = dNi/dx1 dx1/dx + dNi/dx2 dx2/dx + dNi/dx3 dx3/dx
                    Real dNi_dx1 =(Real)( (_coef[i][0])*(1+_coef[i][1]*x2)*(1+_coef[i][2]*x3)/8.0);
                    Real dNi_dx2 =(Real)((1+_coef[i][0]*x1)*(_coef[i][1])*(1+_coef[i][2]*x3)/8.0);
                    Real dNi_dx3 =(Real)((1+_coef[i][0]*x1)*(1+_coef[i][1]*x2)*(_coef[i][2])/8.0);
                    dmsg_info_when(verbose) << "dN"<<i<<"/dxi = "<<dNi_dx1<<" "<<dNi_dx2<<" "<<dNi_dx3;
#ifdef DN_USE_J
                    qx[i] = dNi_dx1*J_1[0][0] + dNi_dx2*J_1[1][0] + dNi_dx3*J_1[2][0];
                    qy[i] = dNi_dx1*J_1[0][1] + dNi_dx2*J_1[1][1] + dNi_dx3*J_1[2][1];
                    qz[i] = dNi_dx1*J_1[0][2] + dNi_dx2*J_1[1][2] + dNi_dx3*J_1[2][2];
                    dmsg_info_when(verbose) << "q"<<i<<" = "<<qx[i]<<" "<<qy[i]<<" "<<qz[i]<<"";
#else
                    qx[i] = dNi_dx1;
                    qy[i] = dNi_dx2;
                    qz[i] = dNi_dx3;
#endif
                }
                for(int i=0; i<8; ++i)
                {
                    defaulttype::Mat<6,3,Real> MBi;
                    //MBi[0][0] = M[0][0] * qx[i]; MBi[0][1] = M[0][1] * qy[i]; MBi[0][2] = M[0][2] * qz[i];
                    //MBi[1][0] = M[1][0] * qx[i]; MBi[1][1] = M[1][1] * qy[i]; MBi[1][2] = M[1][2] * qz[i];
                    //MBi[2][0] = M[2][0] * qx[i]; MBi[2][1] = M[2][1] * qy[i]; MBi[2][2] = M[2][2] * qz[i];
                    //MBi[3][0] = M[3][3] * qy[i]; MBi[3][1] = M[3][3] * qx[i]; MBi[3][2] = 0;
                    //MBi[4][0] = 0;               MBi[4][1] = M[4][4] * qz[i]; MBi[4][2] = M[4][4] * qy[i];
                    //MBi[5][0] = M[5][5] * qz[i]; MBi[5][1] = 0;               MBi[5][2] = M[5][5] * qx[i];
                    MBi[0][0] = U * qx[i]; MBi[0][1] = V * qy[i]; MBi[0][2] = V * qz[i];
                    MBi[1][0] = V * qx[i]; MBi[1][1] = U * qy[i]; MBi[1][2] = V * qz[i];
                    MBi[2][0] = V * qx[i]; MBi[2][1] = V * qy[i]; MBi[2][2] = U * qz[i];
                    MBi[3][0] = W * qy[i]; MBi[3][1] = W * qx[i]; MBi[3][2] = (Real)0;
                    MBi[4][0] = (Real)0;   MBi[4][1] = W * qz[i]; MBi[4][2] = W * qy[i];
                    MBi[5][0] = W * qz[i]; MBi[5][1] = (Real)0;   MBi[5][2] = W * qx[i];

                    dmsg_info_when(verbose) << "MB"<<i<<" = "<<MBi<<"";

                    for(int j=i; j<8; ++j)
                    {
                        Mat33 k; // k = BjtMBi
                        k[0][0] = qx[j]*MBi[0][0]   + qy[j]*MBi[3][0]   + qz[j]*MBi[5][0];
                        k[0][1] = qx[j]*MBi[0][1]   + qy[j]*MBi[3][1] /*+ qz[j]*MBi[5][1]*/;
                        k[0][2] = qx[j]*MBi[0][2] /*+ qy[j]*MBi[3][2]*/ + qz[j]*MBi[5][2];

                        k[1][0] = qy[j]*MBi[1][0]   + qx[j]*MBi[3][0] /*+ qz[j]*MBi[4][0]*/;
                        k[1][1] = qy[j]*MBi[1][1]   + qx[j]*MBi[3][1]   + qz[j]*MBi[4][1];
                        k[1][2] = qy[j]*MBi[1][2] /*+ qx[j]*MBi[3][2]*/ + qz[j]*MBi[4][2];

                        k[2][0] = qz[j]*MBi[2][0] /*+ qy[j]*MBi[4][0]*/ + qx[j]*MBi[5][0];
                        k[2][1] = qz[j]*MBi[2][1]   + qy[j]*MBi[4][1] /*+ qx[j]*MBi[5][1]*/;
                        k[2][2] = qz[j]*MBi[2][2]   + qy[j]*MBi[4][2]   + qx[j]*MBi[5][2];
#ifndef DN_USE_J
                        k = J_1t*k*J_1;
#endif
                        if (verbose) sout << "K"<<i<<j<<" += "<<k<<" * "<<detJ<<""<<sendl;
                        k *= detJ;
                        for(int m=0; m<3; ++m)
                            for(int l=0; l<3; ++l)
                            {
                                K[i*3+m][j*3+l] += k[l][m];
                            }
                    }
                }
            }
    for(int i=0; i<24; ++i)
        for(int j=i+1; j<24; ++j)
        {
            K[j][i] = K[i][j];
        }
    ElementStiffness K1 = K;
    K.fill( 0.0 );





    //Mat33 J_1; // only accurate for orthogonal regular hexa
    J_1.fill( 0.0 );
    Coord l = nodes[6] - nodes[0];
    J_1[0][0]=2.0f / l[0];
    J_1[1][1]=2.0f / l[1];
    J_1[2][2]=2.0f / l[2];


    Real vol = ((nodes[1]-nodes[0]).norm()*(nodes[3]-nodes[0]).norm()*(nodes[4]-nodes[0]).norm());
    //vol *= vol;
    vol /= 8.0; // ???

    K.clear();


    for(int i=0; i<8; ++i)
    {
        Mat33 k = integrateStiffness(  _coef[i][0], _coef[i][1],_coef[i][2],  _coef[i][0], _coef[i][1],_coef[i][2], M[0][0], M[0][1],M[3][3], J_1  )*vol;


        for(int m=0; m<3; ++m)
            for(int l=0; l<3; ++l)
            {
                K[i*3+m][i*3+l] += k[m][l];
            }



        for(int j=i+1; j<8; ++j)
        {
            Mat33 k = integrateStiffness(  _coef[i][0], _coef[i][1],_coef[i][2],  _coef[j][0], _coef[j][1],_coef[j][2], M[0][0], M[0][1],M[3][3], J_1  )*vol;


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
// 	if (elementIndice==0)
// 	{
// 		sout << "nodes = "<<nodes[0]<<"  "<<nodes[1]<<"  "<<nodes[2]<<"  "<<nodes[3]<<"  "<<nodes[4]<<"  "<<nodes[5]<<"  "<<nodes[6]<<"  "<<nodes[7]<<sendl;
// 		sout << "M = "<<M<<sendl;
// 		sout << "K = "<<sendl;
// 		for (int i=0;i<24;i++)
// 			sout << K[i] << sendl;
// 		sout << "K1 = "<<sendl;
// 		for (int i=0;i<24;i++)
// 			sout << K1[i] << sendl;
// 	}
#ifdef GENERIC_STIFFNESS_MATRIX
    K=K1;
#endif

    K *= (Real)stiffnessFactor;

    if (verbose)
    {
        std::stringstream tmp;
        tmp <<"============================ computeElementStiffness:  Element "<<"   ===STIFNESSMATRIX====" << msgendl;
        for(int inode=0; inode<8; inode++)
        {
            for(int icomp=0; icomp<3; icomp++)
            {
                int imatrix=inode*3+icomp;

                for(int jnode=0; jnode<8; jnode++)
                {
                    tmp<<"| ";
                    for(int jcomp=0; jcomp<3; jcomp++)
                    {
                        int jmatrix=jnode*3+jcomp;
                        tmp<<K[imatrix][jmatrix]<<" ";
                    }
                }
                tmp<<" |  "<<std::endl;
            }
            tmp<<"  "<<sendl;
        }
        tmp<<"===============================================================" << msgendl;
        msg_info() << tmp.str() ;
    }
}




template<class DataTypes>
typename HexahedronFEMForceField<DataTypes>::Mat33 HexahedronFEMForceField<DataTypes>::integrateStiffness( int signx0, int signy0, int signz0, int signx1, int signy1, int signz1, const Real u, const Real v, const Real w, const Mat33& J_1  )
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

    Real t1 = J_1[0][0]*J_1[0][0];                // m^-2            (J0J0             )
    Real t2 = t1*signx0;                          // m^-2            (J0J0    sx0      )
    Real t3 = (Real)(signy0*signz0);              //                 (           sy0sz0)
    Real t4 = t2*t3;                              // m^-2            (J0J0    sx0sy0sz0)
    Real t5 = w*signx1;                           // kg m^-4 s^-2    (W       sx1      )
    Real t6 = (Real)(signy1*signz1);              //                 (           sy1sz1)
    Real t7 = t5*t6;                              // kg m^-4 s^-2    (W       sx1sy1sz1)
    Real t10 = t1*signy0;                         // m^-2            (J0J0       sy0   )
    Real t12 = w*signy1;                          // kg m^-4 s^-2    (W          sy1   )
    Real t13 = t12*signz1;                        // kg m^-4 s^-2    (W          sy1sz1)
    Real t16 = t2*signz0;                         // m^-2            (J0J0    sx0   sz0)
    Real t17 = u*signx1;                          // kg m^-4 s^-2    (U       sx1      )
    Real t18 = t17*signz1;                        // kg m^-4 s^-2    (U       sx1   sz1)
    Real t21 = t17*t6;                            // kg m^-4 s^-2    (U       sx1sy1sz1)
    Real t24 = t2*signy0;                         // m^-2            (J0J0    sx0sy0   )
    Real t25 = t17*signy1;                        // kg m^-4 s^-2    (U       sx1sy1   )
    Real t28 = t5*signy1;                         // kg m^-4 s^-2    (W       sx1sy1   )
    Real t32 = w*signz1;                          // kg m^-4 s^-2    (W             sz1)
    Real t37 = t5*signz1;                         // kg m^-4 s^-2    (W       sx1   sz1)
    Real t43 = J_1[0][0]*signx0;                  // m^-1            (J0      sx0      )
    Real t45 = v*J_1[1][1];                       // kg m^-5 s^-2    (VJ1              )
    Real t49 = J_1[0][0]*signy0;                  // m^-1            (J0         sy0   )
    Real t50 = t49*signz0;                        // m^-1            (J0         sy0sz0)
    Real t51 = w*J_1[1][1];                       // kg m^-5 s^-2    (WJ1              )
    Real t52 = (Real)(signx1*signz1);             //                 (        sx1   sz1)
    Real t53 = t51*t52;                           // kg m^-5 s^-2    (WJ1     sx1   sz1)
    Real t56 = t45*signy1;                        // kg m^-5 s^-2    (VJ1        sy1   )
    Real t64 = v*J_1[2][2];                       // kg m^-5 s^-2    (VJ2              )
    Real t68 = w*J_1[2][2];                       // kg m^-5 s^-2    (WJ2              )
    Real t69 = (Real)(signx1*signy1);             //                 (        sx1sy1   )
    Real t70 = t68*t69;                           // kg m^-5 s^-2    (WJ2     sx1sy1   )
    Real t73 = t64*signz1;                        // kg m^-5 s^-2    (VJ2           sz1)
    Real t81 = J_1[1][1]*signy0;                  // m^-1            (J1         sy0   )
    Real t83 = v*J_1[0][0];                       // kg m^-5 s^-2    (VJ0              )
    Real t87 = J_1[1][1]*signx0;                  // m^-1            (J1      sx0      )
    Real t88 = t87*signz0;                        // m^-1            (J1      sx0   sz0)
    Real t89 = w*J_1[0][0];                       // kg m^-5 s^-2    (WJ0              )
    Real t90 = t89*t6;                            // kg m^-5 s^-2    (WJ0        sy1sz1)
    Real t93 = t83*signx1;                        // kg m^-5 s^-2    (VJ0     sx1      )
    Real t100 = J_1[1][1]*J_1[1][1];              // m^-2            (J1J1             )
    Real t101 = t100*signx0;                      // m^-2            (J1J1    sx0      )
    Real t102 = t101*t3;                          // m^-2            (J1J1    sx0sy0sz0)
    Real t110 = t100*signy0;                      // m^-2            (J1J1       sy0   )
    Real t111 = t110*signz0;                      // m^-2            (J1J1       sy0sz0)
    Real t112 = u*signy1;                         // kg m^-4 s^-2    (U          sy1   )
    Real t113 = t112*signz1;                      // kg m^-4 s^-2    (U          sy1sz1)
    Real t116 = t101*signy0;                      // m^-2            (J1J1    sx0sy0   )
    Real t144 = J_1[2][2]*signy0;                 // m^-1            (J2         sy0   )
    Real t149 = J_1[2][2]*signx0;                 // m^-1            (J2      sx0      )
    Real t150 = t149*signy0;                      // m^-1            (J2      sx0sy0   )
    Real t153 = J_1[2][2]*signz0;                 // m^-1            (J2            sz0)
    Real t172 = J_1[2][2]*J_1[2][2];              // m^-2            (J2J2             )
    Real t173 = t172*signx0;                      // m^-2            (J2J2    sx0      )
    Real t174 = t173*t3;                          // m^-2            (J2J2    sx0sy0sz0)
    Real t177 = t173*signz0;                      // m^-2            (J2J2    sx0   sz0)
    Real t180 = t172*signy0;                      // m^-2            (J2J2       sy0   )
    Real t181 = t180*signz0;                      // m^-2            (J2J2       sy0sz0)
    // kg m^-6 s^-2
    K[0][0] = (float)(t4*t7/36.0+t10*signz0*t13/12.0+t16*t18/24.0+t4*t21/72.0+
            t24*t25/24.0+t24*t28/24.0+t1*signz0*t32/8.0+t10*t12/8.0+t16*t37/24.0+t2*t17/8.0);
    // K00 = (J0J0    sx0sy0sz0)(W       sx1sy1sz1)/36
    //     + (J0J0       sy0sz0)(W          sy1sz1)/12
    //     + (J0J0    sx0   sz0)(U       sx1   sz1)/24
    //     + (J0J0    sx0sy0sz0)(U       sx1sy1sz1)/72
    //     + (J0J0    sx0sy0   )(U       sx1sy1   )/24
    //     + (J0J0    sx0sy0   )(W       sx1sy1   )/24
    //     + (J0J0          sz0)(W             sz1)/8
    //     + (J0J0       sy0   )(W          sy1   )/8
    //     + (J0J0    sx0   sz0)(W       sx1   sz1)/24
    //     + (J0J0    sx0      )(U       sx1      )/8
    K[0][1] = (float)(t43*signz0*t45*t6/24.0+t50*t53/24.0+t43*t56/8.0+t49*t51*
            signx1/8.0);
    // K01 = (J0      sx0   sz0)(VJ1        sy1sz1)/24
    //     + (J0         sy0sz0)(WJ1     sx1   sz1)/24
    //     + (J0      sx0      )(VJ1        sy1   )/8
    //     + (J0         sy0   )(WJ1     sx1      )/8
    K[0][2] = (float)(t43*signy0*t64*t6/24.0+t50*t70/24.0+t43*t73/8.0+J_1[0][0]*signz0
            *t68*signx1/8.0);
    // K02 = (J0      sx0sy0   )(VJ2        sy1sz1)/24
    //     + (J0         sy0sz0)(WJ2     sx1sy1   )/24
    //     + (J0      sx0      )(VJ2           sz1)/8
    //     + (J0            sz0)(WJ2     sx1      )/8
    K[1][0] = (float)(t81*signz0*t83*t52/24.0+t88*t90/24.0+t81*t93/8.0+t87*t89*
            signy1/8.0);
    // K10 = (J1         sy0sz0)(VJ0     sx1   sz1)/24
    //     + (J1      sx0   sz0)(WJ0        sy1sz1)/24
    //     + (J1         sy0   )(VJ0     sx1      )/8
    //     + (J1      sx0      )(WJ0        sy1   )/8
    K[1][1] = (float)(t102*t7/36.0+t102*t21/72.0+t101*signz0*t37/12.0+t111*t113
            /24.0+t116*t28/24.0+t100*signz0*t32/8.0+t111*t13/24.0+t116*t25/24.0+t110*t112/
            8.0+t101*t5/8.0);
    // K11 = (J1J1    sx0sy0sz0)(W       sx1sy1sz1)/36
    //     + (J1J1    sx0sy0sz0)(U       sx1sy1sz1)/72
    //     + (J1J1    sx0   sz0)(W       sx1   sz1)/12
    //     + (J1J1       sy0sz0)(U          sy1sz1)/24
    //     + (J1J1    sx0sy0   )(W       sx1sy1   )/24
    //     + (J1J1          sz0)(W             sz1)/8
    //     + (J1J1       sy0sz0)(W          sy1sz1)/24
    //     + (J1J1    sx0sy0   )(U       sx1sy1   )/24
    //     + (J1J1       sy0   )(U          sy1   )/8
    //     + (J1J1    sx0      )(W       sx1      )/8
    K[1][2] = (float)(t87*signy0*t64*t52/24.0+t88*t70/24.0+t81*t73/8.0+J_1[1][1]*
            signz0*t68*signy1/8.0);
    // K12 = (J1      sx0sy0   )(VJ2     sx1   sz1)/24
    //     + (J1      sx0   sz0)(WJ2     sx1sy1   )/24
    //     + (J1         sy0   )(VJ2           sz1)/8
    //     + (J1            sz0)(WJ2        sy1   )/8
    K[2][0] = (float)(t144*signz0*t83*t69/24.0+t150*t90/24.0+t153*t93/8.0+t149*
            t89*signz1/8.0);
    // K20 = (J2         sy0sz0)(VJ0     sx1sy1   )/24
    //     + (J2      sx0sy0   )(WJ0        sy1sz1)/24
    //     + (J2            sz0)(VJ0     sx1      )/8
    //     + (J2      sx0      )(WJ0           sz1)/8
    K[2][1] = (float)(t149*signz0*t45*t69/24.0+t150*t53/24.0+t153*t56/8.0+t144*
            t51*signz1/8.0);
    // K21 = (J2      sx0   sz0)(VJ1     sx1sy1   )/24
    //     + (J2      sx0sy0   )(WJ1     sx1   sz1)/24
    //     + (J2            sz0)(VJ1        sy1   )/8
    //     + (J2         sy0   )(WJ1           sz1)/8
    K[2][2] = (float)(t174*t7/36.0+t177*t37/24.0+t181*t13/24.0+t174*t21/72.0+
            t173*signy0*t28/12.0+t180*t12/8.0+t181*t113/24.0+t177*t18/24.0+t172*signz0*u*
            signz1/8.0+t173*t5/8.0);
    // K22 = (J2J2    sx0sy0sz0)(W       sx1sy1sz1)/36
    //     + (J2J2    sx0   sz0)(W       sx1   sz1)/24
    //     + (J2J2       sy0sz0)(W          sy1sz1)/24
    //     + (J2J2    sx0sy0sz0)(U       sx1sy1sz1)/72
    //     + (J2J2    sx0sy0   )(W       sx1sy1   )/12
    //     + (J2J2       sy0   )(W          sy1   )/8
    //     + (J2J2       sy0sz0)(U          sy1sz1)/24
    //     + (J2J2    sx0   sz0)(U       sx1   sz1)/24
    //     + (J2J2          sz0)(U             sz1)/8
    //     + (J2J2    sx0      )(W       sx1      )/8

    return K /*/(J_1[0][0]*J_1[1][1]*J_1[2][2])*/;

    // K = J_1t E J_1
    // E00 = Usx0sx1(sy0sz0sy1sz1/9 + sz0sz1/3 + sy0sy1/3 + 1)/8 + W(2sx0sy0sz0sx1sy1sz1/9 + 2sy0sz0sy1sz1/3 + sx0sy0sx1sy1/3 + sx0sz0sx1sz1/3 + sz0sz1 + sy0sy1)/8
    // E00 = Usx0sx1(1+sz0sz1/3)(1+sy0sy1/3)/8 + W(1+sx0sx1/3)(2sy0sy1sz0sz1/3+sz0sz1+sy0sy1)/8
    // E01 = Vsx0sy1(1+sz0sz1/3)/8             + Wsy0sx1(1+sz0sz1/3)/8
    // E02 = Vsx0sz1(1+sy0sy1/3)/8             + Wsz0sx1(1+sy0sy1/3)/8
    // E10 = Vsy0sx1(1+sz0sz1/3)/8             + Wsx0sy1(1+sz0sz1/3)/8
    // E11 = Usy0sy1(1+sz0sz1/3)(1+sx0sx1/3)/8 + W(1+sy0sy1/3)(2sx0sx1sz0sz1/3+sx0sx1+sz0sz1)/8
    // E12 = Vsy0sz1(1+sx0sx1/3)/8             + Wsz0sy1(1+sx0sx1/3)/8
    // E20 = Vsz0sx1(1+sy0sy1/3)/8             + Wsx0sz1(1+sy0sy1/3)/8
    // E21 = Vsz0sy1(1+sx0sx1/3)/8             + Wsy0sz1(1+sx0sx1/3)/8
    // E22 = Usz0sz1(1+sy0sy1/3)(1+sx0sx1/3)/8 + W(1+sz0sz1/3)(2sx0sx1sy0sy1/3+sx0sx1+sy0sy1)/8

}



template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::computeMaterialStiffness(int i)
{
    _materialsStiffnesses[i][0][0] = _materialsStiffnesses[i][1][1] = _materialsStiffnesses[i][2][2] = 1;
    _materialsStiffnesses[i][0][1] = _materialsStiffnesses[i][0][2] = _materialsStiffnesses[i][1][0]
            = _materialsStiffnesses[i][1][2] = _materialsStiffnesses[i][2][0] =
                    _materialsStiffnesses[i][2][1] = f_poissonRatio.getValue()/(1-f_poissonRatio.getValue());
    _materialsStiffnesses[i][0][3] = _materialsStiffnesses[i][0][4] =	_materialsStiffnesses[i][0][5] = 0;
    _materialsStiffnesses[i][1][3] = _materialsStiffnesses[i][1][4] =	_materialsStiffnesses[i][1][5] = 0;
    _materialsStiffnesses[i][2][3] = _materialsStiffnesses[i][2][4] =	_materialsStiffnesses[i][2][5] = 0;
    _materialsStiffnesses[i][3][0] = _materialsStiffnesses[i][3][1] = _materialsStiffnesses[i][3][2] = _materialsStiffnesses[i][3][4] =	_materialsStiffnesses[i][3][5] = 0;
    _materialsStiffnesses[i][4][0] = _materialsStiffnesses[i][4][1] = _materialsStiffnesses[i][4][2] = _materialsStiffnesses[i][4][3] =	_materialsStiffnesses[i][4][5] = 0;
    _materialsStiffnesses[i][5][0] = _materialsStiffnesses[i][5][1] = _materialsStiffnesses[i][5][2] = _materialsStiffnesses[i][5][3] =	_materialsStiffnesses[i][5][4] = 0;
    _materialsStiffnesses[i][3][3] = _materialsStiffnesses[i][4][4] = _materialsStiffnesses[i][5][5] = (1-2*f_poissonRatio.getValue())/(2*(1-f_poissonRatio.getValue()));
    _materialsStiffnesses[i] *= (f_youngModulus.getValue()*(1-f_poissonRatio.getValue()))/((1+f_poissonRatio.getValue())*(1-2*f_poissonRatio.getValue()));
    // S = [ U V V 0 0 0 ]
    //     [ V U V 0 0 0 ]
    //     [ V V U 0 0 0 ]
    //     [ 0 0 0 W 0 0 ]
    //     [ 0 0 0 0 W 0 ]
    //     [ 0 0 0 0 0 W ]
    // with U = y * (1-p)/( (1+p)(1-2p))
    //      V = y *    p /( (1+p)(1-2p))
    //      W = y *  1   /(2(1+p)) = (U-V)/2
}

template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::computeForce( Displacement &F, const Displacement &Depl, const ElementStiffness &K )
{
    // 576*multiplications+1176*subscripts+552*additions+24*storage+24*assignments
    F = K*Depl;
}


/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
////////////// small displacements method

template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::initSmall(int i, const Element &elem)
{
    // Rotation matrix identity
    Transformation t; t.identity();
    _rotations[i] = t;

    for(int w=0; w<8; ++w)
#ifndef SOFA_NEW_HEXA
        _rotatedInitialElements[i][w] =  _rotations[i]*_initialPoints.getValue()[elem[_indices[w]]];
#else
        _rotatedInitialElements[i][w] =  _rotations[i]*_initialPoints.getValue()[elem[w]];
#endif


    if( _elementStiffnesses.getValue().size() <= (unsigned)i )
    {
        _elementStiffnesses.beginEdit()->resize( _elementStiffnesses.getValue().size()+1 );
    }

    computeElementStiffness( (*_elementStiffnesses.beginEdit())[i], _materialsStiffnesses[i], _rotatedInitialElements[i], i, _sparseGrid?_sparseGrid->getStiffnessCoef(i):1.0 );
}

template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::accumulateForceSmall ( WDataRefVecDeriv &f, RDataRefVecCoord &p, int i, const Element&elem )
{
    Vec<8,Coord> nodes;
    for(int w=0; w<8; ++w)
#ifndef SOFA_NEW_HEXA
        nodes[w] = p[elem[_indices[w]]];
#else
        nodes[w] = p[elem[w]];
#endif

    // positions of the deformed and displaced Tetrahedron in its frame
    Vec<8,Coord> deformed;
    for(int w=0; w<8; ++w)
        deformed[w] = nodes[w];

    // displacement
    Displacement D;
    for(int k=0 ; k<8 ; ++k )
    {
        int indice = k*3;
        for(int j=0 ; j<3 ; ++j )
            D[indice+j] = _rotatedInitialElements[i][k][j] - nodes[k][j];
    }


    if(f_updateStiffnessMatrix.getValue())
        computeElementStiffness( (*_elementStiffnesses.beginEdit())[i], _materialsStiffnesses[i], deformed, i, _sparseGrid?_sparseGrid->getStiffnessCoef(i):1.0 );


    Displacement F; //forces
    computeForce( F, D, _elementStiffnesses.getValue()[i] ); // compute force on element

    for(int w=0; w<8; ++w)
#ifndef SOFA_NEW_HEXA
        f[elem[_indices[w]]] += Deriv( F[w*3],  F[w*3+1],   F[w*3+2]  ) ;
#else
        f[elem[w]] += Deriv( F[w*3],  F[w*3+1],   F[w*3+2]  ) ;
#endif

    m_potentialEnergy += dot(Deriv( F[0], F[1], F[2] ) ,-Deriv( D[0], D[1], D[2]));
    m_potentialEnergy += dot(Deriv( F[3], F[4], F[5] ) ,-Deriv( D[3], D[4], D[5] ));
    m_potentialEnergy += dot(Deriv( F[6], F[7], F[8] ) ,-Deriv( D[6], D[7], D[8] ));
    m_potentialEnergy += dot(Deriv( F[9], F[10], F[11]),-Deriv( D[9], D[10], D[11] ));
    m_potentialEnergy += dot(Deriv( F[12], F[13], F[14]),-Deriv( D[12], D[13], D[14] ));
    m_potentialEnergy += dot(Deriv( F[15], F[16], F[17]),-Deriv( D[15], D[16], D[17] ));
    m_potentialEnergy += dot(Deriv( F[18], F[19], F[20]),-Deriv( D[18], D[19], D[20] ));
    m_potentialEnergy += dot(Deriv( F[21], F[22], F[23]),-Deriv( D[21], D[22], D[23] ));
}


/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
////////////// large displacements method


template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::initLarge(int i, const Element &elem)
{
    // Rotation matrix (initial Tetrahedre/world)
    // edges mean on 3 directions


    defaulttype::Vec<8,Coord> nodes;
    for(int w=0; w<8; ++w)
#ifndef SOFA_NEW_HEXA
        nodes[w] = _initialPoints.getValue()[elem[_indices[w]]];
#else
        nodes[w] = _initialPoints.getValue()[elem[w]];
#endif

    Coord horizontal;
    horizontal = (nodes[1]-nodes[0] + nodes[2]-nodes[3] + nodes[5]-nodes[4] + nodes[6]-nodes[7])*.25;
    Coord vertical;
    vertical = (nodes[3]-nodes[0] + nodes[2]-nodes[1] + nodes[7]-nodes[4] + nodes[6]-nodes[5])*.25;
    computeRotationLarge( _rotations[i], horizontal,vertical);
    _initialrotations[i] = _rotations[i];

    for(int w=0; w<8; ++w)
#ifndef SOFA_NEW_HEXA
        _rotatedInitialElements[i][w] =  _rotations[i]*_initialPoints.getValue()[elem[_indices[w]]];
#else
        _rotatedInitialElements[i][w] =  _rotations[i]*_initialPoints.getValue()[elem[w]];
#endif

    if( _elementStiffnesses.getValue().size() <= (unsigned)i )
    {
        _elementStiffnesses.beginEdit()->resize( _elementStiffnesses.getValue().size()+1 );
    }

    computeElementStiffness( (*_elementStiffnesses.beginEdit())[i], _materialsStiffnesses[i], _rotatedInitialElements[i], i, _sparseGrid?_sparseGrid->getStiffnessCoef(i):1.0 );
// 	computeElementStiffness( (*_elementStiffnesses.beginEdit())[i], _materialsStiffnesses[i], _rotatedInitialElements[i], i, i==1?10.0:1.0 );

// 	printMatlab( serr,this->_elementStiffnesses.getValue()[0] );

}

template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::computeRotationLarge( Transformation &r, Coord &edgex, Coord &edgey)
{

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
void HexahedronFEMForceField<DataTypes>::accumulateForceLarge( WDataRefVecDeriv &f, RDataRefVecCoord &p, int i, const Element&elem )
{
    defaulttype::Vec<8,Coord> nodes;
    for(int w=0; w<8; ++w)
#ifndef SOFA_NEW_HEXA
        nodes[w] = p[elem[_indices[w]]];
#else
        nodes[w] = p[elem[w]];
#endif

    Coord horizontal;
    horizontal = (nodes[1]-nodes[0] + nodes[2]-nodes[3] + nodes[5]-nodes[4] + nodes[6]-nodes[7])*.25;
    Coord vertical;
    vertical = (nodes[3]-nodes[0] + nodes[2]-nodes[1] + nodes[7]-nodes[4] + nodes[6]-nodes[5])*.25;

// 	Transformation R_0_2; // Rotation matrix (deformed and displaced Tetrahedron/world)
    computeRotationLarge( _rotations[i], horizontal,vertical);

// 	_rotations[i].transpose(R_0_2);

    // positions of the deformed and displaced Tetrahedron in its frame
    defaulttype::Vec<8,Coord> deformed;
    for(int w=0; w<8; ++w)
        deformed[w] = _rotations[i] * nodes[w];


    // displacement
    Displacement D;
    for(int k=0 ; k<8 ; ++k )
    {
        int indice = k*3;
        for(int j=0 ; j<3 ; ++j )
            D[indice+j] = _rotatedInitialElements[i][k][j] - deformed[k][j];
    }


    if(f_updateStiffnessMatrix.getValue())
// 		computeElementStiffness( _elementStiffnesses[i], _materialsStiffnesses[i], deformed );
        computeElementStiffness( (*_elementStiffnesses.beginEdit())[i], _materialsStiffnesses[i], deformed, i, _sparseGrid?_sparseGrid->getStiffnessCoef(i):1.0 );


    Displacement F; //forces
    computeForce( F, D, _elementStiffnesses.getValue()[i] ); // compute force on element

    for(int w=0; w<8; ++w)
#ifndef SOFA_NEW_HEXA
        f[elem[_indices[w]]] += _rotations[i].multTranspose( Deriv( F[w*3],  F[w*3+1],   F[w*3+2]  ) );
#else
        f[elem[w]] += _rotations[i].multTranspose( Deriv( F[w*3],  F[w*3+1],   F[w*3+2]  ) );
#endif

    m_potentialEnergy += dot(Deriv( F[0], F[1], F[2] ) ,-Deriv( D[0], D[1], D[2]));
    m_potentialEnergy += dot(Deriv( F[3], F[4], F[5] ) ,-Deriv( D[3], D[4], D[5] ));
    m_potentialEnergy += dot(Deriv( F[6], F[7], F[8] ) ,-Deriv( D[6], D[7], D[8] ));
    m_potentialEnergy += dot(Deriv( F[9], F[10], F[11]),-Deriv( D[9], D[10], D[11] ));
    m_potentialEnergy += dot(Deriv( F[12], F[13], F[14]),-Deriv( D[12], D[13], D[14] ));
    m_potentialEnergy += dot(Deriv( F[15], F[16], F[17]),-Deriv( D[15], D[16], D[17] ));
    m_potentialEnergy += dot(Deriv( F[18], F[19], F[20]),-Deriv( D[18], D[19], D[20] ));
    m_potentialEnergy += dot(Deriv( F[21], F[22], F[23]),-Deriv( D[21], D[22], D[23] ));
}







/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
////////////// polar decomposition method



template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::initPolar(int i, const Element& elem)
{
    defaulttype::Vec<8,Coord> nodes;
    for(int j=0; j<8; ++j)
#ifndef SOFA_NEW_HEXA
        nodes[j] = _initialPoints.getValue()[elem[_indices[j]]];
#else
        nodes[j] = _initialPoints.getValue()[elem[j]];
#endif

    computeRotationPolar( _rotations[i], nodes );

    _initialrotations[i] = _rotations[i];

    for(int j=0; j<8; ++j)
    {
        _rotatedInitialElements[i][j] =  _rotations[i] * nodes[j];
    }


    if( _elementStiffnesses.getValue().size() <= (unsigned)i )
    {
        _elementStiffnesses.beginEdit()->resize( _elementStiffnesses.getValue().size()+1 );
    }


    computeElementStiffness( (*_elementStiffnesses.beginEdit())[i], _materialsStiffnesses[i], _rotatedInitialElements[i], i, _sparseGrid?_sparseGrid->getStiffnessCoef(i):1.0);
}



template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::computeRotationPolar( Transformation &r, defaulttype::Vec<8,Coord> &nodes)
{
    Transformation A;
    Coord Edge =(nodes[1]-nodes[0] + nodes[2]-nodes[3] + nodes[5]-nodes[4] + nodes[6]-nodes[7])*.25;
    //Edge.normalize();
    A[0][0] = Edge[0];
    A[0][1] = Edge[1];
    A[0][2] = Edge[2];
    Edge = (nodes[3]-nodes[0] + nodes[2]-nodes[1] + nodes[7]-nodes[4] + nodes[6]-nodes[5])*.25;
    //Edge.normalize();
    A[1][0] = Edge[0];
    A[1][1] = Edge[1];
    A[1][2] = Edge[2];
    Edge = (nodes[4]-nodes[0] + nodes[5]-nodes[1] + nodes[7]-nodes[3] + nodes[6]-nodes[2])*.25;
    //Edge.normalize();
    A[2][0] = Edge[0];
    A[2][1] = Edge[1];
    A[2][2] = Edge[2];

    Mat33 HT;
    for(int k=0; k<3; ++k)
        for(int j=0; j<3; ++j)
            HT[k][j]=A[k][j];
    //HT[3][0] = HT[3][1] = HT[3][2] = HT[0][3] = HT[1][3] = HT[2][3] = 0;
    //HT[3][3] = 1;

    helper::Decompose<Real>::polarDecomposition(HT, r);
}


template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::accumulateForcePolar( WDataRefVecDeriv &f, RDataRefVecCoord &p, int i, const Element&elem )
{
    defaulttype::Vec<8,Coord> nodes;
    for(int j=0; j<8; ++j)
#ifndef SOFA_NEW_HEXA
        nodes[j] = p[elem[_indices[j]]];
#else
        nodes[j] = p[elem[j]];
#endif

// 	Transformation R_0_2; // Rotation matrix (deformed and displaced Tetrahedron/world)
    computeRotationPolar( _rotations[i], nodes );

// 	_rotations[i].transpose( R_0_2 );


    // positions of the deformed and displaced Tetrahedre in its frame
    defaulttype::Vec<8,Coord> deformed;
    for(int j=0; j<8; ++j)
        deformed[j] = _rotations[i] * nodes[j];



    // displacement
    Displacement D;
    for(int k=0 ; k<8 ; ++k )
    {
        int indice = k*3;
        for(int j=0 ; j<3 ; ++j )
            D[indice+j] = _rotatedInitialElements[i][k][j] - deformed[k][j];
    }

    //forces
    Displacement F;

    if(f_updateStiffnessMatrix.getValue())
// 		computeElementStiffness( _elementStiffnesses[i], _materialsStiffnesses[i], deformed );
        computeElementStiffness( (*_elementStiffnesses.beginEdit())[i], _materialsStiffnesses[i], deformed, i, _sparseGrid?_sparseGrid->getStiffnessCoef(i):1.0);


    // compute force on element
    computeForce( F, D, _elementStiffnesses.getValue()[i] );


    for(int j=0; j<8; ++j)
    #ifndef SOFA_NEW_HEXA
            f[elem[_indices[j]]] += _rotations[i].multTranspose( Deriv( F[j*3],  F[j*3+1],   F[j*3+2]  ) );
    #else
            f[elem[j]] += _rotations[i].multTranspose( Deriv( F[j*3],  F[j*3+1],   F[j*3+2]  ) );
    #endif

    m_potentialEnergy += dot(Deriv( F[0], F[1], F[2] ) ,-Deriv( D[0], D[1], D[2]));
    m_potentialEnergy += dot(Deriv( F[3], F[4], F[5] ) ,-Deriv( D[3], D[4], D[5] ));
    m_potentialEnergy += dot(Deriv( F[6], F[7], F[8] ) ,-Deriv( D[6], D[7], D[8] ));
    m_potentialEnergy += dot(Deriv( F[9], F[10], F[11]),-Deriv( D[9], D[10], D[11] ));
    m_potentialEnergy += dot(Deriv( F[12], F[13], F[14]),-Deriv( D[12], D[13], D[14] ));
    m_potentialEnergy += dot(Deriv( F[15], F[16], F[17]),-Deriv( D[15], D[16], D[17] ));
    m_potentialEnergy += dot(Deriv( F[18], F[19], F[20]),-Deriv( D[18], D[19], D[20] ));
    m_potentialEnergy += dot(Deriv( F[21], F[22], F[23]),-Deriv( D[21], D[22], D[23] ));
}

template<class DataTypes>
inline SReal HexahedronFEMForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams*) const
{
    return m_potentialEnergy;
}



/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////


template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    // Build Matrix Block for this ForceField
    int i,j,n1, n2, e;

    typename VecElement::const_iterator it;

    Index node1, node2;

    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);

    for(it = this->getIndexedElements()->begin(), e=0 ; it != this->getIndexedElements()->end() ; ++it,++e)
    {
        const ElementStiffness &Ke = _elementStiffnesses.getValue()[e];
//         const Transformation& Rt = _rotations[e];
//         Transformation R; R.transpose(Rt);

        Transformation Rot = getElementRotation(e);

        Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());
        // find index of node 1
        for (n1=0; n1<8; n1++)
        {
#ifndef SOFA_NEW_HEXA
            node1 = (*it)[_indices[n1]];
#else
            node1 = (*it)[n1];
#endif
            // find index of node 2
            for (n2=0; n2<8; n2++)
            {
#ifndef SOFA_NEW_HEXA
                node2 = (*it)[_indices[n2]];
#else
                node2 = (*it)[n2];
#endif
                Mat33 tmp = Rot.multTranspose( Mat33(Coord(Ke[3*n1+0][3*n2+0],Ke[3*n1+0][3*n2+1],Ke[3*n1+0][3*n2+2]),
                        Coord(Ke[3*n1+1][3*n2+0],Ke[3*n1+1][3*n2+1],Ke[3*n1+1][3*n2+2]),
                        Coord(Ke[3*n1+2][3*n2+0],Ke[3*n1+2][3*n2+1],Ke[3*n1+2][3*n2+2])) ) * Rot;
                for(i=0; i<3; i++)
                    for (j=0; j<3; j++)
                        r.matrix->add(r.offset+3*node1+i, r.offset+3*node2+j, - tmp[i][j]*kFactor);
            }
        }
    }
}




template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
//    if ( !vparams->isSupported(sofa::core::visual::API_OpenGL) )
//    {
//        this->sout << "WARNING in : " << this->getClassName() << " in draw(VisualParams*) method :\n" <<
//                "Cannot display this component debug info beacause of using GL render instructions"<<
//                this->sendl;
//        return;
//    }

// 	serr<<"HexahedronFEMForceField<DataTypes>::draw()"<<sendl;
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!this->mstate) return;
    if (!f_drawing.getValue()) return;


    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0,true);



    typename VecElement::const_iterator it;
    int i;
    for(it = this->getIndexedElements()->begin(), i = 0 ; it != this->getIndexedElements()->end() ; ++it, ++i)
    {


        std::vector< defaulttype::Vector3 > points[6];

        Index a = (*it)[0];
        Index b = (*it)[1];
#ifndef SOFA_NEW_HEXA
        Index d = (*it)[2];
        Index c = (*it)[3];
#else
        Index d = (*it)[3];
        Index c = (*it)[2];
#endif
        Index e = (*it)[4];
        Index f = (*it)[5];
#ifndef SOFA_NEW_HEXA
        Index h = (*it)[6];
        Index g = (*it)[7];
#else
        Index h = (*it)[7];
        Index g = (*it)[6];
#endif

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
        Real percentage = f_drawPercentageOffset.getValue();
        Coord pa = x[a]-(x[a]-center)*percentage;
        Coord pb = x[b]-(x[b]-center)*percentage;
        Coord pc = x[c]-(x[c]-center)*percentage;
        Coord pd = x[d]-(x[d]-center)*percentage;
        Coord pe = x[e]-(x[e]-center)*percentage;
        Coord pf = x[f]-(x[f]-center)*percentage;
        Coord pg = x[g]-(x[g]-center)*percentage;
        Coord ph = x[h]-(x[h]-center)*percentage;


        if(_sparseGrid )
        {
            vparams->drawTool()->enableBlending();
        }




// 		glColor4f(0.7f, 0.7f, 0.1f, (_sparseGrid?_sparseGrid->getStiffnessCoef(i):1.0f));
        points[0].push_back(pa);
        points[0].push_back(pb);
        points[0].push_back(pc);
        points[0].push_back(pa);
        points[0].push_back(pc);
        points[0].push_back(pd);
// 		glColor4f(0.7f, 0, 0, (_sparseGrid?_sparseGrid->getStiffnessCoef(i):1.0f));

        points[1].push_back(pe);
        points[1].push_back(pf);
        points[1].push_back(pg);
        points[1].push_back(pe);
        points[1].push_back(pg);
        points[1].push_back(ph);

// 		glColor4f(0, 0.7f, 0, (_sparseGrid?_sparseGrid->getStiffnessCoef(i):1.0f));

        points[2].push_back(pc);
        points[2].push_back(pd);
        points[2].push_back(ph);
        points[2].push_back(pc);
        points[2].push_back(ph);
        points[2].push_back(pg);

// 		glColor4f(0, 0, 0.7f, (_sparseGrid?_sparseGrid->getStiffnessCoef(i):1.0f));

        points[3].push_back(pa);
        points[3].push_back(pb);
        points[3].push_back(pf);
        points[3].push_back(pa);
        points[3].push_back(pf);
        points[3].push_back(pe);

// 		glColor4f(0.1f, 0.7f, 0.7f, (_sparseGrid?_sparseGrid->getStiffnessCoef(i):1.0f));

        points[4].push_back(pa);
        points[4].push_back(pd);
        points[4].push_back(ph);
        points[4].push_back(pa);
        points[4].push_back(ph);
        points[4].push_back(pe);

// 		glColor4f(0.7f, 0.1f, 0.7f, (_sparseGrid?_sparseGrid->getStiffnessCoef(i):1.0f));

        points[5].push_back(pb);
        points[5].push_back(pc);
        points[5].push_back(pg);
        points[5].push_back(pb);
        points[5].push_back(pg);
        points[5].push_back(pf);


        vparams->drawTool()->setLightingEnabled(false);
        vparams->drawTool()->drawTriangles(points[0], defaulttype::Vec<4,float>(0.7f,0.7f,0.1f,(_sparseGrid?_sparseGrid->getStiffnessCoef(i):1.0f)));
        vparams->drawTool()->drawTriangles(points[1], defaulttype::Vec<4,float>(0.7f,0.0f,0.0f,(_sparseGrid?_sparseGrid->getStiffnessCoef(i):1.0f)));
        vparams->drawTool()->drawTriangles(points[2], defaulttype::Vec<4,float>(0.0f,0.7f,0.0f,(_sparseGrid?_sparseGrid->getStiffnessCoef(i):1.0f)));
        vparams->drawTool()->drawTriangles(points[3], defaulttype::Vec<4,float>(0.0f,0.0f,0.7f,(_sparseGrid?_sparseGrid->getStiffnessCoef(i):1.0f)));
        vparams->drawTool()->drawTriangles(points[4], defaulttype::Vec<4,float>(0.1f,0.7f,0.7f,(_sparseGrid?_sparseGrid->getStiffnessCoef(i):1.0f)));
        vparams->drawTool()->drawTriangles(points[5], defaulttype::Vec<4,float>(0.7f,0.1f,0.7f,(_sparseGrid?_sparseGrid->getStiffnessCoef(i):1.0f)));

    }


    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0,false);

    if(_sparseGrid )
       vparams->drawTool()->disableBlending();
}


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_HEXAHEDRONFEMFORCEFIELD_INL
