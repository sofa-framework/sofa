/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/component/solidmechanics/fem/elastic/HexahedronFEMForceField.h>
#include <sofa/component/solidmechanics/fem/elastic/BaseLinearElasticityFEMForceField.inl>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <sofa/linearalgebra/RotationMatrix.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/helper/decompose.h>
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>

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

namespace sofa::component::solidmechanics::fem::elastic
{

using std::set;
using namespace sofa::type;
using namespace sofa::defaulttype;

template <class DataTypes>
HexahedronFEMForceField<DataTypes>::HexahedronFEMForceField()
    : d_method(initData(&d_method, std::string("large"), "method", "\"large\" or \"polar\" or \"small\" displacements" ))
    , d_updateStiffnessMatrix(initData(&d_updateStiffnessMatrix, false, "updateStiffnessMatrix", ""))
    , d_gatherPt(initData(&d_gatherPt, "gatherPt", "number of dof accumulated per threads during the gather operation (Only use in GPU version)"))
    , d_gatherBsize(initData(&d_gatherBsize, "gatherBsize", "number of dof accumulated per threads during the gather operation (Only use in GPU version)"))
    , d_drawing(initData(&d_drawing, true, "drawing", "draw the forcefield if true"))
    , d_drawPercentageOffset(initData(&d_drawPercentageOffset, (Real)0.15, "drawPercentageOffset", "size of the hexa"))
    , needUpdateTopology(false)
    , d_elementStiffnesses(initData(&d_elementStiffnesses, "stiffnessMatrices", "Stiffness matrices per element (K_i)"))
    , _sparseGrid(nullptr)
    , d_initialPoints(initData(&d_initialPoints, "initialPoints", "Initial Position"))
    , data(new HexahedronFEMForceFieldInternalData<DataTypes>())
{
    data->initPtrData(this);
    _coef(0,0)=-1;
    _coef(1,0)=1;
    _coef(2,0)=1;
    _coef(3,0)=-1;
    _coef(4,0)=-1;
    _coef(5,0)=1;
    _coef(6,0)=1;
    _coef(7,0)=-1;
    _coef(0,1)=-1;
    _coef(1,1)=-1;
    _coef(2,1)=1;
    _coef(3,1)=1;
    _coef(4,1)=-1;
    _coef(5,1)=-1;
    _coef(6,1)=1;
    _coef(7,1)=1;
    _coef(0,2)=-1;
    _coef(1,2)=-1;
    _coef(2,2)=-1;
    _coef(3,2)=-1;
    _coef(4,2)=1;
    _coef(5,2)=1;
    _coef(6,2)=1;
    _coef(7,2)=1;

    _alreadyInit=false;
}


template <class DataTypes>
void HexahedronFEMForceField<DataTypes>::init()
{
    if (_alreadyInit)
    {
        return;
    }
    _alreadyInit=true;

    BaseLinearElasticityFEMForceField<DataTypes>::init();

    if (this->d_componentState.getValue() == sofa::core::objectmodel::ComponentState::Invalid)
    {
        return;
    }

    if ( this->l_topology->getNbHexahedra() <= 0 )
    {
        msg_error() << "Object must have a hexahedric MeshTopology." << msgendl
                    << " name: " << this->l_topology->getName() << msgendl
                    << " typename: " << this->l_topology->getTypeName() << msgendl
                    << " nbPoints:" << this->l_topology->getNbPoints() << msgendl;
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    _sparseGrid = dynamic_cast<topology::container::grid::SparseGridTopology*>(this->l_topology.get());
    m_potentialEnergy = 0;

    this->d_componentState.setValue(core::objectmodel::ComponentState::Valid);
    reinit();
}


template <class DataTypes>
void HexahedronFEMForceField<DataTypes>::reinit()
{
    const VecCoord& p = this->mstate->read(core::vec_id::read_access::restPosition)->getValue();
    d_initialPoints.setValue(p);

    _materialsStiffnesses.resize(this->getIndexedElements()->size() );
    _rotations.resize( this->getIndexedElements()->size() );
    _rotatedInitialElements.resize(this->getIndexedElements()->size());
    _initialrotations.resize( this->getIndexedElements()->size() );

    if (d_method.getValue() == "large")
        this->setMethod(LARGE);
    else if (d_method.getValue() == "polar")
        this->setMethod(POLAR);
    else if (d_method.getValue() == "small")
        this->setMethod(SMALL);

    switch(method)
    {
    case LARGE :
    {
        sofa::Index i=0;
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
        sofa::Index i=0;
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
        sofa::Index i=0;
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

    const auto* indexedElements = this->getIndexedElements();

    switch(method)
    {
    case LARGE :
    {
        m_potentialEnergy = 0;
        for(it=indexedElements->begin(); it!=indexedElements->end(); ++it,++i)
        {
            accumulateForceLarge( _f, _p, i, *it );
        }
        m_potentialEnergy/=-2.0;
        break;
    }
    case POLAR :
    {
        m_potentialEnergy = 0;
        for(it=indexedElements->begin(); it!=indexedElements->end(); ++it,++i)
        {
            accumulateForcePolar( _f, _p, i, *it );
        }
        m_potentialEnergy/=-2.0;
        break;
    }
    case SMALL :
    {
        m_potentialEnergy = 0;
        for(it=indexedElements->begin(); it!=indexedElements->end(); ++it,++i)
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
    Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    if (_df.size() != _dx.size())
        _df.resize(_dx.size());

    unsigned int i = 0;
    typename VecElement::const_iterator it;

    const auto* indexedElements = this->getIndexedElements();

    for(it = indexedElements->begin() ; it != indexedElements->end() ; ++it, ++i)
    {
        // Transformation R_0_2;
        // R_0_2.transpose(_rotations[i]);

        Displacement X;

        for(int w=0; w<8; ++w)
        {
            Coord x_2;
            x_2 = _rotations[i] * _dx[(*it)[w]];

            X[w*3] = x_2[0];
            X[w*3+1] = x_2[1];
            X[w*3+2] = x_2[2];
        }

        Displacement F;
        computeForce(F, X, d_elementStiffnesses.getValue()[i] );

        for(int w=0; w<8; ++w)
        {
            _df[(*it)[w]] -= _rotations[i].multTranspose(Deriv(F[w*3], F[w*3+1], F[w*3+2])) * kFactor;
        }
    }
}

template <class DataTypes>
const typename HexahedronFEMForceField<DataTypes>::Transformation& HexahedronFEMForceField<DataTypes>::getElementRotation(const sofa::Index elemidx)
{
    return _rotations[elemidx];
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
// enable to use generic matrix computing code instead of the original optimized code specific to parallelepipeds
#define GENERIC_STIFFNESS_MATRIX
// enable to use the full content of the MaterialStiffness matrix, instead of only the 3x3 upper block
#define MAT_STIFFNESS_USE_W
// enable to use J when computing qx/qy/qz, instead of computing the matrix relative to (x1,x2,x3) and pre/post multiply by J^-1 afterward.
// note that this does not matter if the element is a cube.
#define DN_USE_J


template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::computeElementStiffness( ElementStiffness &K, const MaterialStiffness &M, const type::Vec<8, Coord> &nodes, const sofa::Index elementIndice, double stiffnessFactor) const
{
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
    Mat33 J; // J(i,j) = dXi/dxj
    Mat33 J_1; // J_1(i,j) = dxi/dXj
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
            J(c,0) = lx[c]/2;
            J(c,1) = ly[c]/2;
            J(c,2) = lz[c]/2;
        }
        detJ = type::determinant(J);
        const bool canInvert = J_1.invert(J);
        assert(canInvert);
        SOFA_UNUSED(canInvert);
        J_1t.transpose(J_1);

        dmsg_info_when(verbose) << "J = " << J << msgendl
                                << "invJ = "  << J_1 << msgendl
                                << "detJ = " << detJ << msgendl;
    }
    const Real U = M(0,0);
    const Real V = M(0,1);
#ifdef MAT_STIFFNESS_USE_W
    const Real W = M(3,3);
#else
    const Real W = M(2,2);
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
                //Mat33 J; // J(i,j) = dXi/dxj
                //Mat33 J_1; // J_1(i,j) = dxi/dXj
                if (!isParallel)
                {
                    for (int c=0; c<3; ++c)
                    {
                        J(c,0) = (Real)( (nodes[1][c]-nodes[0][c])*(1-x2)*(1-x3)/8+(nodes[2][c]-nodes[3][c])*(1+x2)*(1-x3)/8+(nodes[5][c]-nodes[4][c])*(1-x2)*(1+x3)/8+(nodes[6][c]-nodes[7][c])*(1+x2)*(1+x3)/8);
                        J(c,1) =(Real)( (nodes[3][c]-nodes[0][c])*(1-x1)*(1-x3)/8+(nodes[2][c]-nodes[1][c])*(1+x1)*(1-x3)/8+(nodes[7][c]-nodes[4][c])*(1-x1)*(1+x3)/8+(nodes[6][c]-nodes[5][c])*(1+x1)*(1+x3)/8);
                        J(c,2) =(Real)( (nodes[4][c]-nodes[0][c])*(1-x1)*(1-x2)/8+(nodes[5][c]-nodes[1][c])*(1+x1)*(1-x2)/8+(nodes[6][c]-nodes[2][c])*(1+x1)*(1+x2)/8+(nodes[7][c]-nodes[3][c])*(1-x1)*(1+x2)/8);
                    }
                    detJ = type::determinant(J);
                    const bool canInvert = J_1.invert(J);
                    assert(canInvert);
                    SOFA_UNUSED(canInvert);
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
                    // Ni = 1/8 (1+_coef(i,0)x1)(1+_coef(i,1)x2)(1+_coef(i,2)x3)
                    // qxi = dNi/dx = dNi/dx1 dx1/dx + dNi/dx2 dx2/dx + dNi/dx3 dx3/dx
                    Real dNi_dx1 =(Real)( (_coef(i,0))*(1+_coef(i,1)*x2)*(1+_coef(i,2)*x3)/8.0);
                    Real dNi_dx2 =(Real)((1+_coef(i,0)*x1)*(_coef(i,1))*(1+_coef(i,2)*x3)/8.0);
                    Real dNi_dx3 =(Real)((1+_coef(i,0)*x1)*(1+_coef(i,1)*x2)*(_coef(i,2))/8.0);
                    dmsg_info_when(verbose) << "dN"<<i<<"/dxi = "<<dNi_dx1<<" "<<dNi_dx2<<" "<<dNi_dx3;
#ifdef DN_USE_J
                    qx[i] = dNi_dx1*J_1(0,0) + dNi_dx2*J_1(1,0) + dNi_dx3*J_1(2,0);
                    qy[i] = dNi_dx1*J_1(0,1) + dNi_dx2*J_1(1,1) + dNi_dx3*J_1(2,1);
                    qz[i] = dNi_dx1*J_1(0,2) + dNi_dx2*J_1(1,2) + dNi_dx3*J_1(2,2);
                    dmsg_info_when(verbose) << "q"<<i<<" = "<<qx[i]<<" "<<qy[i]<<" "<<qz[i]<<"";
#else
                    qx[i] = dNi_dx1;
                    qy[i] = dNi_dx2;
                    qz[i] = dNi_dx3;
#endif
                }
                for(int i=0; i<8; ++i)
                {
                    type::Mat<6,3,Real> MBi;
                    MBi(0,0) = U * qx[i]; MBi(0,1) = V * qy[i]; MBi(0,2) = V * qz[i];
                    MBi(1,0) = V * qx[i]; MBi(1,1) = U * qy[i]; MBi(1,2) = V * qz[i];
                    MBi(2,0) = V * qx[i]; MBi(2,1) = V * qy[i]; MBi(2,2) = U * qz[i];
                    MBi(3,0) = W * qy[i]; MBi(3,1) = W * qx[i]; MBi(3,2) = (Real)0;
                    MBi(4,0) = (Real)0;   MBi(4,1) = W * qz[i]; MBi(4,2) = W * qy[i];
                    MBi(5,0) = W * qz[i]; MBi(5,1) = (Real)0;   MBi(5,2) = W * qx[i];

                    dmsg_info_when(verbose) << "MB"<<i<<" = "<<MBi<<"";

                    for(int j=i; j<8; ++j)
                    {
                        Mat33 k; // k = BjtMBi
                        k(0,0) = qx[j]*MBi(0,0)   + qy[j]*MBi(3,0)   + qz[j]*MBi(5,0);
                        k(0,1) = qx[j]*MBi(0,1)   + qy[j]*MBi(3,1) /*+ qz[j]*MBi(5,1)*/;
                        k(0,2) = qx[j]*MBi(0,2) /*+ qy[j]*MBi(3,2)*/ + qz[j]*MBi(5,2);

                        k(1,0) = qy[j]*MBi(1,0)   + qx[j]*MBi(3,0) /*+ qz[j]*MBi(4,0)*/;
                        k(1,1) = qy[j]*MBi(1,1)   + qx[j]*MBi(3,1)   + qz[j]*MBi(4,1);
                        k(1,2) = qy[j]*MBi(1,2) /*+ qx[j]*MBi(3,2)*/ + qz[j]*MBi(4,2);

                        k(2,0) = qz[j]*MBi(2,0) /*+ qy[j]*MBi(4,0)*/ + qx[j]*MBi(5,0);
                        k(2,1) = qz[j]*MBi(2,1)   + qy[j]*MBi(4,1) /*+ qx[j]*MBi(5,1)*/;
                        k(2,2) = qz[j]*MBi(2,2)   + qy[j]*MBi(4,2)   + qx[j]*MBi(5,2);
#ifndef DN_USE_J
                        k = J_1t*k*J_1;
#endif
                        dmsg_info_when(verbose) << "K"<<i<<j<<" += "<<k<<" * "<<detJ<<"";
                        k *= detJ;
                        for(int m=0; m<3; ++m)
                            for(int l=0; l<3; ++l)
                            {
                                K(i*3+m,j*3+l) += k(l,m);
                            }
                    }
                }
            }
    for(int i=0; i<24; ++i)
        for(int j=i+1; j<24; ++j)
        {
            K(j,i) = K(i,j);
        }
    ElementStiffness K1 = K;
    K.fill( 0.0 );

    //Mat33 J_1; // only accurate for orthogonal regular hexa
    J_1.fill( 0.0 );
    Coord l = nodes[6] - nodes[0];
    J_1(0,0)=2.0f / l[0];
    J_1(1,1)=2.0f / l[1];
    J_1(2,2)=2.0f / l[2];


    Real vol = ((nodes[1]-nodes[0]).norm()*(nodes[3]-nodes[0]).norm()*(nodes[4]-nodes[0]).norm());
    //vol *= vol;
    vol /= 8.0; // ???

    K.clear();


    for(int i=0; i<8; ++i)
    {
        Mat33 k = integrateStiffness(  _coef(i,0), _coef(i,1),_coef(i,2),  _coef(i,0), _coef(i,1),_coef(i,2), M(0,0), M(0,1),M(3,3), J_1  )*vol;


        for(int m=0; m<3; ++m)
            for(int l=0; l<3; ++l)
            {
                K(i*3+m,i*3+l) += k(m,l);
            }

        for(int j=i+1; j<8; ++j)
        {
            Mat33 k = integrateStiffness(  _coef(i,0), _coef(i,1),_coef(i,2),  _coef(j,0), _coef(j,1),_coef(j,2), M(0,0), M(0,1),M(3,3), J_1  )*vol;


            for(int m=0; m<3; ++m)
                for(int l=0; l<3; ++l)
                {
                    K(i*3+m,j*3+l) += k(m,l);
                }
        }
    }

    for(int i=0; i<24; ++i)
        for(int j=i+1; j<24; ++j)
        {
            K(j,i) = K(i,j);
        }

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
                        tmp<<K(imatrix,jmatrix)<<" ";
                    }
                }
                tmp<<" |  "<<std::endl;
            }
            tmp<<"  "<< std::endl;
        }
        tmp<<"===============================================================" << msgendl;
        msg_info() << tmp.str() ;
    }
}




template<class DataTypes>
typename HexahedronFEMForceField<DataTypes>::Mat33 HexahedronFEMForceField<DataTypes>::integrateStiffness( int signx0, int signy0, int signz0, int signx1, int signy1, int signz1, const Real u, const Real v, const Real w, const Mat33& J_1  )
{
    Mat33 K;

    Real t1 = J_1(0,0)*J_1(0,0);                // m^-2            (J0J0             )
    Real t2 = t1*signx0;                          // m^-2            (J0J0    sx0      )
    Real t3 = Real(signy0)* Real(signz0);              //                 (           sy0sz0)
    Real t4 = t2*t3;                              // m^-2            (J0J0    sx0sy0sz0)
    Real t5 = w*signx1;                           // kg m^-4 s^-2    (W       sx1      )
    Real t6 = Real(signy1)*Real(signz1);              //                 (           sy1sz1)
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
    Real t43 = J_1(0,0)*signx0;                  // m^-1            (J0      sx0      )
    Real t45 = v*J_1(1,1);                       // kg m^-5 s^-2    (VJ1              )
    Real t49 = J_1(0,0)*signy0;                  // m^-1            (J0         sy0   )
    Real t50 = t49*signz0;                        // m^-1            (J0         sy0sz0)
    Real t51 = w*J_1(1,1);                       // kg m^-5 s^-2    (WJ1              )
    Real t52 = Real(signx1)* Real(signz1);             //                 (        sx1   sz1)
    Real t53 = t51*t52;                           // kg m^-5 s^-2    (WJ1     sx1   sz1)
    Real t56 = t45*signy1;                        // kg m^-5 s^-2    (VJ1        sy1   )
    Real t64 = v*J_1(2,2);                       // kg m^-5 s^-2    (VJ2              )
    Real t68 = w*J_1(2,2);                       // kg m^-5 s^-2    (WJ2              )
    Real t69 = Real(signx1)* Real(signy1);             //                 (        sx1sy1   )
    Real t70 = t68*t69;                           // kg m^-5 s^-2    (WJ2     sx1sy1   )
    Real t73 = t64*signz1;                        // kg m^-5 s^-2    (VJ2           sz1)
    Real t81 = J_1(1,1)*signy0;                  // m^-1            (J1         sy0   )
    Real t83 = v*J_1(0,0);                       // kg m^-5 s^-2    (VJ0              )
    Real t87 = J_1(1,1)*signx0;                  // m^-1            (J1      sx0      )
    Real t88 = t87*signz0;                        // m^-1            (J1      sx0   sz0)
    Real t89 = w*J_1(0,0);                       // kg m^-5 s^-2    (WJ0              )
    Real t90 = t89*t6;                            // kg m^-5 s^-2    (WJ0        sy1sz1)
    Real t93 = t83*signx1;                        // kg m^-5 s^-2    (VJ0     sx1      )
    Real t100 = J_1(1,1)*J_1(1,1);              // m^-2            (J1J1             )
    Real t101 = t100*signx0;                      // m^-2            (J1J1    sx0      )
    Real t102 = t101*t3;                          // m^-2            (J1J1    sx0sy0sz0)
    Real t110 = t100*signy0;                      // m^-2            (J1J1       sy0   )
    Real t111 = t110*signz0;                      // m^-2            (J1J1       sy0sz0)
    Real t112 = u*signy1;                         // kg m^-4 s^-2    (U          sy1   )
    Real t113 = t112*signz1;                      // kg m^-4 s^-2    (U          sy1sz1)
    Real t116 = t101*signy0;                      // m^-2            (J1J1    sx0sy0   )
    Real t144 = J_1(2,2)*signy0;                 // m^-1            (J2         sy0   )
    Real t149 = J_1(2,2)*signx0;                 // m^-1            (J2      sx0      )
    Real t150 = t149*signy0;                      // m^-1            (J2      sx0sy0   )
    Real t153 = J_1(2,2)*signz0;                 // m^-1            (J2            sz0)
    Real t172 = J_1(2,2)*J_1(2,2);              // m^-2            (J2J2             )
    Real t173 = t172*signx0;                      // m^-2            (J2J2    sx0      )
    Real t174 = t173*t3;                          // m^-2            (J2J2    sx0sy0sz0)
    Real t177 = t173*signz0;                      // m^-2            (J2J2    sx0   sz0)
    Real t180 = t172*signy0;                      // m^-2            (J2J2       sy0   )
    Real t181 = t180*signz0;                      // m^-2            (J2J2       sy0sz0)
    // kg m^-6 s^-2
    K(0,0) = (float)(t4*t7/36.0+t10*signz0*t13/12.0+t16*t18/24.0+t4*t21/72.0+
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
    K(0,1) = (float)(t43*signz0*t45*t6/24.0+t50*t53/24.0+t43*t56/8.0+t49*t51*
            signx1/8.0);
    // K01 = (J0      sx0   sz0)(VJ1        sy1sz1)/24
    //     + (J0         sy0sz0)(WJ1     sx1   sz1)/24
    //     + (J0      sx0      )(VJ1        sy1   )/8
    //     + (J0         sy0   )(WJ1     sx1      )/8
    K(0,2) = (float)(t43*signy0*t64*t6/24.0+t50*t70/24.0+t43*t73/8.0+J_1(0,0)*signz0
            *t68*signx1/8.0);
    // K02 = (J0      sx0sy0   )(VJ2        sy1sz1)/24
    //     + (J0         sy0sz0)(WJ2     sx1sy1   )/24
    //     + (J0      sx0      )(VJ2           sz1)/8
    //     + (J0            sz0)(WJ2     sx1      )/8
    K(1,0) = (float)(t81*signz0*t83*t52/24.0+t88*t90/24.0+t81*t93/8.0+t87*t89*
            signy1/8.0);
    // K10 = (J1         sy0sz0)(VJ0     sx1   sz1)/24
    //     + (J1      sx0   sz0)(WJ0        sy1sz1)/24
    //     + (J1         sy0   )(VJ0     sx1      )/8
    //     + (J1      sx0      )(WJ0        sy1   )/8
    K(1,1) = (float)(t102*t7/36.0+t102*t21/72.0+t101*signz0*t37/12.0+t111*t113
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
    K(1,2) = (float)(t87*signy0*t64*t52/24.0+t88*t70/24.0+t81*t73/8.0+J_1(1,1)*
            signz0*t68*signy1/8.0);
    // K12 = (J1      sx0sy0   )(VJ2     sx1   sz1)/24
    //     + (J1      sx0   sz0)(WJ2     sx1sy1   )/24
    //     + (J1         sy0   )(VJ2           sz1)/8
    //     + (J1            sz0)(WJ2        sy1   )/8
    K(2,0) = (float)(t144*signz0*t83*t69/24.0+t150*t90/24.0+t153*t93/8.0+t149*
            t89*signz1/8.0);
    // K20 = (J2         sy0sz0)(VJ0     sx1sy1   )/24
    //     + (J2      sx0sy0   )(WJ0        sy1sz1)/24
    //     + (J2            sz0)(VJ0     sx1      )/8
    //     + (J2      sx0      )(WJ0           sz1)/8
    K(2,1) = (float)(t149*signz0*t45*t69/24.0+t150*t53/24.0+t153*t56/8.0+t144*
            t51*signz1/8.0);
    // K21 = (J2      sx0   sz0)(VJ1     sx1sy1   )/24
    //     + (J2      sx0sy0   )(WJ1     sx1   sz1)/24
    //     + (J2            sz0)(VJ1        sy1   )/8
    //     + (J2         sy0   )(WJ1           sz1)/8
    K(2,2) = (float)(t174*t7/36.0+t177*t37/24.0+t181*t13/24.0+t174*t21/72.0+
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

    return K /*/(J_1(0,0)*J_1(1,1)*J_1(2,2))*/;
}



template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::computeMaterialStiffness(sofa::Index i)
{
    const auto poissonRatio = this->getPoissonRatioInElement(i);
    _materialsStiffnesses[i](0,0) = _materialsStiffnesses[i](1,1) = _materialsStiffnesses[i](2,2) = 1;
    _materialsStiffnesses[i](0,1) = _materialsStiffnesses[i](0,2) = _materialsStiffnesses[i](1,0)
            = _materialsStiffnesses[i](1,2) = _materialsStiffnesses[i](2,0) =
                    _materialsStiffnesses[i](2,1) = poissonRatio / (1 - poissonRatio);
    _materialsStiffnesses[i](0,3) = _materialsStiffnesses[i](0,4) =	_materialsStiffnesses[i](0,5) = 0;
    _materialsStiffnesses[i](1,3) = _materialsStiffnesses[i](1,4) =	_materialsStiffnesses[i](1,5) = 0;
    _materialsStiffnesses[i](2,3) = _materialsStiffnesses[i](2,4) =	_materialsStiffnesses[i](2,5) = 0;
    _materialsStiffnesses[i](3,0) = _materialsStiffnesses[i](3,1) = _materialsStiffnesses[i](3,2) = _materialsStiffnesses[i](3,4) =	_materialsStiffnesses[i](3,5) = 0;
    _materialsStiffnesses[i](4,0) = _materialsStiffnesses[i](4,1) = _materialsStiffnesses[i](4,2) = _materialsStiffnesses[i](4,3) =	_materialsStiffnesses[i](4,5) = 0;
    _materialsStiffnesses[i](5,0) = _materialsStiffnesses[i](5,1) = _materialsStiffnesses[i](5,2) = _materialsStiffnesses[i](5,3) =	_materialsStiffnesses[i](5,4) = 0;
    _materialsStiffnesses[i](3,3) = _materialsStiffnesses[i](4,4) = _materialsStiffnesses[i](5,5) = (1- 2 * poissonRatio) / (2 * (1 - poissonRatio));
    _materialsStiffnesses[i] *= (this->getYoungModulusInElement(i) * (1 - poissonRatio)) / ((1 + poissonRatio) * (1 - 2 * poissonRatio));
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
    F = K*Depl;
}


/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
////////////// small displacements method

template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::initSmall(sofa::Index i, const Element &elem)
{
    // Rotation matrix identity
    Transformation t; t.identity();
    _rotations[i] = t;

    for(int w=0; w<8; ++w)
        _rotatedInitialElements[i][w] = _rotations[i] * d_initialPoints.getValue()[elem[w]];

    auto& stiffnesses = *sofa::helper::getWriteOnlyAccessor(d_elementStiffnesses);
    if( stiffnesses.size() <= i )
    {
        stiffnesses.resize( i + 1 );
    }

    computeElementStiffness( stiffnesses[i], _materialsStiffnesses[i], _rotatedInitialElements[i], i, _sparseGrid?_sparseGrid->getStiffnessCoef(i):1.0 );
}

template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::accumulateForceSmall ( WDataRefVecDeriv &f, RDataRefVecCoord &p, sofa::Index i, const Element&elem )
{
    Vec<8,Coord> nodes;
    for(int w=0; w<8; ++w)
        nodes[w] = p[elem[w]];

    // positions of the deformed and displaced Tetrahedron in its frame
    sofa::type::fixed_array<Coord, 8> deformed;
    for(int w=0; w<8; ++w)
        deformed[w] = nodes[w];

    // displacement
    Displacement D;
    for(int k=0 ; k<8 ; ++k )
    {
        const int index = k*3;
        for(int j=0 ; j<3 ; ++j )
            D[index+j] = _rotatedInitialElements[i][k][j] - nodes[k][j];
    }

    auto& stiffnesses = *sofa::helper::getWriteOnlyAccessor(d_elementStiffnesses);
    if(d_updateStiffnessMatrix.getValue())
        computeElementStiffness( stiffnesses[i], _materialsStiffnesses[i], deformed, i, _sparseGrid?_sparseGrid->getStiffnessCoef(i):1.0 );

    Displacement F; //forces
    computeForce( F, D, stiffnesses[i] ); // compute force on element

    for(int w=0; w<8; ++w)
        f[elem[w]] += Deriv( F[w*3],  F[w*3+1],   F[w*3+2]  ) ;

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
void HexahedronFEMForceField<DataTypes>::initLarge(sofa::Index i, const Element &elem)
{
    // Rotation matrix (initial Tetrahedre/world)
    // edges mean on 3 directions
    type::Vec<8,Coord> nodes;
    for(int w=0; w<8; ++w)
        nodes[w] = d_initialPoints.getValue()[elem[w]];

    Coord horizontal;
    horizontal = (nodes[1]-nodes[0] + nodes[2]-nodes[3] + nodes[5]-nodes[4] + nodes[6]-nodes[7])*.25;
    Coord vertical;
    vertical = (nodes[3]-nodes[0] + nodes[2]-nodes[1] + nodes[7]-nodes[4] + nodes[6]-nodes[5])*.25;
    computeRotationLarge( _rotations[i], horizontal,vertical);
    _initialrotations[i] = _rotations[i];

    for(int w=0; w<8; ++w)
        _rotatedInitialElements[i][w] = _rotations[i] * d_initialPoints.getValue()[elem[w]];

    auto& stiffnesses = *sofa::helper::getWriteOnlyAccessor(d_elementStiffnesses);
    if( stiffnesses.size() <= i )
    {
        stiffnesses.resize( i + 1 );
    }

    computeElementStiffness( stiffnesses[i], _materialsStiffnesses[i], _rotatedInitialElements[i], i, _sparseGrid?_sparseGrid->getStiffnessCoef(i):1.0 );
}

template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::computeRotationLarge( Transformation &r, Coord &edgex, Coord &edgey)
{
    edgex.normalize();

    const Coord edgez = cross( edgex, edgey ).normalized();

    edgey = cross( edgez, edgex ); //edgey is unit vector because edgez and edgex are orthogonal unit vectors

    r(0,0) = edgex[0];
    r(0,1) = edgex[1];
    r(0,2) = edgex[2];
    r(1,0) = edgey[0];
    r(1,1) = edgey[1];
    r(1,2) = edgey[2];
    r(2,0) = edgez[0];
    r(2,1) = edgez[1];
    r(2,2) = edgez[2];
}

template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::accumulateForceLarge( WDataRefVecDeriv &f, RDataRefVecCoord &p, sofa::Index i, const Element&elem )
{
    type::Vec<8,Coord> nodes;
    for(int w=0; w<8; ++w)
        nodes[w] = p[elem[w]];

    Coord horizontal;
    horizontal = (nodes[1]-nodes[0] + nodes[2]-nodes[3] + nodes[5]-nodes[4] + nodes[6]-nodes[7])*.25;
    Coord vertical;
    vertical = (nodes[3]-nodes[0] + nodes[2]-nodes[1] + nodes[7]-nodes[4] + nodes[6]-nodes[5])*.25;

    // 	Transformation R_0_2; // Rotation matrix (deformed and displaced Tetrahedron/world)
    computeRotationLarge( _rotations[i], horizontal,vertical);

    // positions of the deformed and displaced Tetrahedron in its frame
    sofa::type::fixed_array<Coord, 8> deformed;
    for(int w=0; w<8; ++w)
        deformed[w] = _rotations[i] * nodes[w];


    // displacement
    Displacement D;
    for(int k=0 ; k<8 ; ++k )
    {
        const int index = k*3;
        for(int j=0 ; j<3 ; ++j )
            D[index+j] = _rotatedInitialElements[i][k][j] - deformed[k][j];
    }

    auto& stiffnesses = *sofa::helper::getWriteOnlyAccessor(d_elementStiffnesses);
    if(d_updateStiffnessMatrix.getValue())
        computeElementStiffness( stiffnesses[i], _materialsStiffnesses[i], deformed, i, _sparseGrid?_sparseGrid->getStiffnessCoef(i):1.0 );

    Displacement F; //forces
    computeForce(F, D, d_elementStiffnesses.getValue()[i] ); // compute force on element

    for(int w=0; w<8; ++w)
        f[elem[w]] += _rotations[i].multTranspose( Deriv( F[w*3],  F[w*3+1],   F[w*3+2]  ) );

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
void HexahedronFEMForceField<DataTypes>::initPolar(sofa::Index i, const Element& elem)
{
    type::Vec<8,Coord> nodes;
    for(int j=0; j<8; ++j)
        nodes[j] = d_initialPoints.getValue()[elem[j]];

    computeRotationPolar( _rotations[i], nodes );

    _initialrotations[i] = _rotations[i];

    for(int j=0; j<8; ++j)
    {
        _rotatedInitialElements[i][j] =  _rotations[i] * nodes[j];
    }

    auto& stiffnesses = *sofa::helper::getWriteOnlyAccessor(d_elementStiffnesses);
    if( stiffnesses.size() <= i )
    {
        stiffnesses.resize( i + 1 );
    }

    computeElementStiffness( stiffnesses[i], _materialsStiffnesses[i], _rotatedInitialElements[i], i, _sparseGrid?_sparseGrid->getStiffnessCoef(i):1.0);
}



template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::computeRotationPolar( Transformation &r, type::Vec<8,Coord> &nodes)
{
    Transformation A;
    Coord Edge =(nodes[1]-nodes[0] + nodes[2]-nodes[3] + nodes[5]-nodes[4] + nodes[6]-nodes[7])*.25;
    A(0,0) = Edge[0];
    A(0,1) = Edge[1];
    A(0,2) = Edge[2];
    Edge = (nodes[3]-nodes[0] + nodes[2]-nodes[1] + nodes[7]-nodes[4] + nodes[6]-nodes[5])*.25;

    A(1,0) = Edge[0];
    A(1,1) = Edge[1];
    A(1,2) = Edge[2];
    Edge = (nodes[4]-nodes[0] + nodes[5]-nodes[1] + nodes[7]-nodes[3] + nodes[6]-nodes[2])*.25;

    A(2,0) = Edge[0];
    A(2,1) = Edge[1];
    A(2,2) = Edge[2];

    Mat33 HT;
    for(int k=0; k<3; ++k)
        for(int j=0; j<3; ++j)
            HT(k,j)=A(k,j);

    helper::Decompose<Real>::polarDecomposition(HT, r);
} 


template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::getNodeRotation(Transformation& R, unsigned int nodeIdx)
{
    core::topology::BaseMeshTopology::HexahedraAroundVertex liste_hexa = this->l_topology->getHexahedraAroundVertex(nodeIdx);

    R(0,0) = R(1,1) = R(2,2) = 1.0 ;
    R(0,1) = R(0,2) = R(1,0) = R(1,2) = R(2,0) = R(2,1) = 0.0 ;

    std::size_t numHexa=liste_hexa.size();

    for (Index ti=0; ti<numHexa; ti++)
    {
        Transformation R0t;
        R0t.transpose(_initialrotations[liste_hexa[ti]]);
        Transformation Rcur = getElementRotation(liste_hexa[ti]);
        R += Rcur * R0t;
    }

    // on "moyenne"
    R(0,0) = R(0,0)/numHexa ; R(0,1) = R(0,1)/numHexa ; R(0,2) = R(0,2)/numHexa ;
    R(1,0) = R(1,0)/numHexa ; R(1,1) = R(1,1)/numHexa ; R(1,2) = R(1,2)/numHexa ;
    R(2,0) = R(2,0)/numHexa ; R(2,1) = R(2,1)/numHexa ; R(2,2) = R(2,2)/numHexa ;

    type::Mat<3,3,Real> Rmoy;
    helper::Decompose<Real>::polarDecomposition( R, Rmoy );

    R = Rmoy;
}

template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::getRotations(linearalgebra::BaseMatrix * rotations,int offset)
{
    auto nbdof = this->mstate->getSize();

    if (linearalgebra::RotationMatrix<float> * diag = dynamic_cast<linearalgebra::RotationMatrix<float> *>(rotations))
    {
        Transformation R;
        for (sofa::Index e=0; e<nbdof; ++e)
        {
            getNodeRotation(R,e);
            for(sofa::Index j=0; j<3; j++)
            {
                for(sofa::Index i=0; i<3; i++)
                {
                    const sofa::Index ind = e * 9 + j * 3 + i;
                    diag->getVector()[ind] = float(R(j,i));
                }
            }
        }
    }
    else if (linearalgebra::RotationMatrix<double> * diag = dynamic_cast<linearalgebra::RotationMatrix<double> *>(rotations))
    {
        Transformation R;
        for (sofa::Index e=0; e<nbdof; ++e)
        {
            getNodeRotation(R,e);
            for(sofa::Index j=0; j<3; j++)
            {
                for(sofa::Index i=0; i<3; i++)
                {
                    const sofa::Index ind = e * 9 + j * 3 + i;
                    diag->getVector()[ind] = R(j,i);
                }
            }
        }
    }
    else
    {
        for (sofa::Index i=0; i<nbdof; ++i)
        {
            Transformation t;
            getNodeRotation(t,i);
            const int e = offset+i*3;
            rotations->set(e+0,e+0,t(0,0)); rotations->set(e+0,e+1,t(0,1)); rotations->set(e+0,e+2,t(0,2));
            rotations->set(e+1,e+0,t(1,0)); rotations->set(e+1,e+1,t(1,1)); rotations->set(e+1,e+2,t(1,2));
            rotations->set(e+2,e+0,t(2,0)); rotations->set(e+2,e+1,t(2,1)); rotations->set(e+2,e+2,t(2,2));
        }
    }
}


template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::accumulateForcePolar( WDataRefVecDeriv &f, RDataRefVecCoord &p, sofa::Index i, const Element&elem )
{
    type::Vec<8,Coord> nodes;
    for(int j=0; j<8; ++j)
        nodes[j] = p[elem[j]];

    // 	Transformation R_0_2; // Rotation matrix (deformed and displaced Tetrahedron/world)
    computeRotationPolar( _rotations[i], nodes );

    // positions of the deformed and displaced Tetrahedre in its frame
    sofa::type::fixed_array<Coord, 8> deformed;
    for(int j=0; j<8; ++j)
        deformed[j] = _rotations[i] * nodes[j];



    // displacement
    Displacement D;
    for(int k=0 ; k<8 ; ++k )
    {
        const int index = k*3;
        for(int j=0 ; j<3 ; ++j )
            D[index+j] = _rotatedInitialElements[i][k][j] - deformed[k][j];
    }

    //forces
    Displacement F;

    auto& stiffnesses = *sofa::helper::getWriteOnlyAccessor(d_elementStiffnesses);
    if(d_updateStiffnessMatrix.getValue())
        computeElementStiffness( stiffnesses[i], _materialsStiffnesses[i], deformed, i, _sparseGrid?_sparseGrid->getStiffnessCoef(i):1.0);


    // compute force on element
    computeForce(F, D, d_elementStiffnesses.getValue()[i] );


    for(int j=0; j<8; ++j)
            f[elem[j]] += _rotations[i].multTranspose( Deriv( F[j*3],  F[j*3+1],   F[j*3+2]  ) );

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
inline SReal HexahedronFEMForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams* mparams, const DataVecCoord& x) const
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(x);

    return m_potentialEnergy;
}

template<class DataTypes>
inline void HexahedronFEMForceField<DataTypes>::handleTopologyChange()
{
    needUpdateTopology = true;
}

template<class DataTypes>
inline void HexahedronFEMForceField<DataTypes>::setMethod(int val)
{
    method = val;
    switch(val)
    {
    case POLAR: d_method.setValue("polar"); break;
    case SMALL: d_method.setValue("small"); break;
    default   : d_method.setValue("large");
    }
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////


template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::addKToMatrix(sofa::linearalgebra::BaseMatrix * matrix, SReal kFact, unsigned int &offset)
{
    // Build Matrix Block for this ForceField

    sofa::Index e { 0 }; //index of the element in the topology

    const auto& stiffnesses = d_elementStiffnesses.getValue();
    const auto* indexedElements = this->getIndexedElements();

    for (const auto& element : *indexedElements)
    {
        const ElementStiffness &Ke = stiffnesses[e];
        const Transformation Rot = getElementRotation(e);
        e++;

        // find index of node 1
        for (Element::size_type n1 = 0; n1 < Element::size(); n1++)
        {
            const auto node1 = element[n1];
            // find index of node 2
            for (Element::size_type n2 = 0; n2 < Element::size(); n2++)
            {
                const auto node2 = element[n2];

                const Mat33 tmp = Rot.multTranspose( Mat33(
                        Coord(Ke(3*n1+0,3*n2+0),Ke(3*n1+0,3*n2+1),Ke(3*n1+0,3*n2+2)),
                        Coord(Ke(3*n1+1,3*n2+0),Ke(3*n1+1,3*n2+1),Ke(3*n1+1,3*n2+2)),
                        Coord(Ke(3*n1+2,3*n2+0),Ke(3*n1+2,3*n2+1),Ke(3*n1+2,3*n2+2))) ) * Rot;

                matrix->add( offset + 3 * node1, offset + 3 * node2, tmp * (-kFact));
            }
        }
    }
}

template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix)
{
    sofa::Index e { 0 }; //index of the element in the topology

    const auto& stiffnesses = d_elementStiffnesses.getValue();
    const auto* indexedElements = this->getIndexedElements();

    auto dfdx = matrix->getForceDerivativeIn(this->mstate)
                       .withRespectToPositionsIn(this->mstate);

    for (const auto& element : *indexedElements)
    {
        const ElementStiffness &Ke = stiffnesses[e];
        const Transformation& Rot = getElementRotation(e);
        e++;

        for (Element::size_type n1 = 0; n1 < Element::size(); n1++)
        {
            const auto node1 = element[n1];
            for (Element::size_type n2 = 0; n2 < Element::size(); n2++)
            {
                const auto node2 = element[n2];

                const Mat33 tmp = Rot.multTranspose( Mat33(
                        Coord(Ke(3*n1+0,3*n2+0),Ke(3*n1+0,3*n2+1),Ke(3*n1+0,3*n2+2)),
                        Coord(Ke(3*n1+1,3*n2+0),Ke(3*n1+1,3*n2+1),Ke(3*n1+1,3*n2+2)),
                        Coord(Ke(3*n1+2,3*n2+0),Ke(3*n1+2,3*n2+1),Ke(3*n1+2,3*n2+2))) ) * Rot;

                dfdx(3 * node1, 3 * node2) += - tmp;
            }
        }
    }
}

template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::computeBBox(const core::ExecParams* params, bool onlyVisible)
{
    SOFA_UNUSED(params);

    if( !onlyVisible ) return;

    helper::ReadAccessor<DataVecCoord> x = this->mstate->read(core::vec_id::write_access::position);

    type::BoundingBox bbox;
    for (const auto& p : x )
    {
        bbox.include(p);
    }

    this->f_bbox.setValue(bbox);
}


template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!d_drawing.getValue()) return;
    if (this->d_componentState.getValue() != sofa::core::objectmodel::ComponentState::Valid) return;
    if (!this->mstate) return;
    if (!this->l_topology) return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    const VecCoord& x = this->mstate->read(core::vec_id::read_access::position)->getValue();

    vparams->drawTool()->setLightingEnabled(false);

    if(_sparseGrid )
    {
        vparams->drawTool()->enableBlending();
    }

    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0,true);

    const Real percentage = d_drawPercentageOffset.getValue();
    const Real oneMinusPercentage = static_cast<Real>(1) - percentage;

    const auto* indexedElements = this->getIndexedElements();

    sofa::type::fixed_array<std::vector<sofa::type::Vec3>, 6 > quads; //one list of quads per hexahedron face
    sofa::type::fixed_array<std::vector<RGBAColor>, 6> colors; //one list of quads per hexahedron face

    for (auto& q : quads)
    {
        q.reserve(indexedElements->size() * 4);
    }
    for (auto& c : colors)
    {
        c.reserve(indexedElements->size() * 4);
    }

    sofa::Index i {};
    for (const auto& element : *indexedElements)
    {
        const Coord& a = x[element[0]];
        const Coord& b = x[element[1]];
        const Coord& c = x[element[2]];
        const Coord& d = x[element[3]];
        const Coord& e = x[element[4]];
        const Coord& f = x[element[5]];
        const Coord& g = x[element[6]];
        const Coord& h = x[element[7]];

        const Coord center = (a + b + c + d + e + f + g + h ) * static_cast<Real>(0.125);
        const Coord centerPercent = center * percentage;

        const Coord pa = a * oneMinusPercentage + centerPercent;
        const Coord pb = b * oneMinusPercentage + centerPercent;
        const Coord pc = c * oneMinusPercentage + centerPercent;
        const Coord pd = d * oneMinusPercentage + centerPercent;
        const Coord pe = e * oneMinusPercentage + centerPercent;
        const Coord pf = f * oneMinusPercentage + centerPercent;
        const Coord pg = g * oneMinusPercentage + centerPercent;
        const Coord ph = h * oneMinusPercentage + centerPercent;

        quads[0].emplace_back(pa);
        quads[0].emplace_back(pb);
        quads[0].emplace_back(pc);
        quads[0].emplace_back(pd);

        quads[1].emplace_back(pe);
        quads[1].emplace_back(pf);
        quads[1].emplace_back(pg);
        quads[1].emplace_back(ph);

        quads[2].emplace_back(pc);
        quads[2].emplace_back(pd);
        quads[2].emplace_back(ph);
        quads[2].emplace_back(pg);

        quads[3].emplace_back(pa);
        quads[3].emplace_back(pb);
        quads[3].emplace_back(pf);
        quads[3].emplace_back(pe);

        quads[4].emplace_back(pa);
        quads[4].emplace_back(pd);
        quads[4].emplace_back(ph);
        quads[4].emplace_back(pe);

        quads[5].emplace_back(pb);
        quads[5].emplace_back(pc);
        quads[5].emplace_back(pg);
        quads[5].emplace_back(pf);

        const float stiffnessCoef = _sparseGrid ? _sparseGrid->getStiffnessCoef(i) : 1.0f;
        sofa::type::fixed_array<sofa::type::RGBAColor, 6> quadColors {
            sofa::type::RGBAColor(0.7f,0.7f,0.1f,stiffnessCoef),
            sofa::type::RGBAColor(0.7f,0.0f,0.0f,stiffnessCoef),
            sofa::type::RGBAColor(0.0f,0.7f,0.0f,stiffnessCoef),
            sofa::type::RGBAColor(0.0f,0.0f,0.7f,stiffnessCoef),
            sofa::type::RGBAColor(0.1f,0.7f,0.7f,stiffnessCoef),
            sofa::type::RGBAColor(0.7f,0.1f,0.7f,stiffnessCoef)
        };

        for (unsigned int j = 0; j < 6; ++j)
        {
            auto& faceColors = colors[j];
            const auto& color = quadColors[j];
            for (unsigned int k = 0; k < 4; ++k)
            {
                faceColors.emplace_back(color);
            }
        }

        ++i;
    }

    for (unsigned int j = 0; j < 6; ++j)
    {
        vparams->drawTool()->drawQuads(quads[j], colors[j]);
    }
}


} //namespace sofa::component::solidmechanics::fem::elastic
