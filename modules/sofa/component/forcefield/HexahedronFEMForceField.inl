/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_FORCEFIELD_HEXAHEDRONFEMFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_HEXAHEDRONFEMFORCEFIELD_INL

#include <sofa/core/componentmodel/behavior/ForceField.inl>
#include <sofa/component/forcefield/HexahedronFEMForceField.h>
#include <sofa/helper/PolarDecompose.h>
#include <sofa/helper/gl/template.h>
#include <assert.h>
#include <iostream>
#include <set>

using std::cerr;
using std::endl;
using std::set;



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

using namespace sofa::defaulttype;


template<class DataTypes> const int HexahedronFEMForceField<DataTypes>::_indices[8] = {0,1,3,2,4,5,7,6};
// template<class DataTypes> const int HexahedronFEMForceField<DataTypes>::_indices[8] = {4,5,7,6,0,1,3,2};


template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::parse(core::objectmodel::BaseObjectDescription* arg)
{
    this->core::componentmodel::behavior::ForceField<DataTypes>::parse(arg);
    if (f_method == "large")
        this->setMethod(LARGE);
    else if (f_method == "polar")
        this->setMethod(POLAR);
    else if (f_method == "fast")
        this->setMethod(FAST);

}

template <class DataTypes>
void HexahedronFEMForceField<DataTypes>::init()
{
    if(_alreadyInit)return;
    else _alreadyInit=true;

    this->core::componentmodel::behavior::ForceField<DataTypes>::init();
    if( this->getContext()->getTopology()==NULL )
    {
        std::cerr << "ERROR(HexahedronFEMForceField): object must have a Topology.\n";
        return;
    }

    _mesh = dynamic_cast<sofa::component::topology::MeshTopology*>(this->getContext()->getTopology());
    if ( _mesh==NULL)
    {
        std::cerr << "ERROR(HexahedronFEMForceField): object must have a MeshTopology.\n";
        return;
    }
    else if( _mesh->getNbCubes()<=0 )
    {
        std::cerr << "ERROR(HexahedronFEMForceField): object must have a hexahedric MeshTopology.\n";
        std::cerr << _mesh->getName()<<std::endl;
        std::cerr << _mesh->getTypeName()<<std::endl;
        cerr<<_mesh->getNbPoints()<<endl;
        return;
    }
// 	if (!_mesh->getCubes().empty())
// 	else
// 	{
    _indexedElements = & (_mesh->getCubes());
// 	}
    _trimgrid = dynamic_cast<topology::FittedRegularGridTopology*>(_mesh);
    _sparseGrid = dynamic_cast<topology::SparseGridTopology*>(_mesh);



    if (_initialPoints.getValue().size() == 0)
    {
        VecCoord& p = *this->mstate->getX();
        _initialPoints.setValue(p);
    }

    _materialsStiffnesses.resize(_indexedElements->size() );
    _rotations.resize( _indexedElements->size() );
    _rotatedInitialElements.resize(_indexedElements->size());

// 	if( _elementStiffnesses.getValue().empty() )
// 		_elementStiffnesses.beginEdit()->resize(_indexedElements->size());
    // 	_stiffnesses.resize( _initialPoints.getValue().size()*3 ); // assembly ?

    reinit();


// 	unsigned int i=0;
// 	typename VecElement::const_iterator it;
// 	for(it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i)
// 	{
// 		Element c = *it;
// 		for(int w=0;w<8;++w)
// 		{
// 			cerr<<"sparse w : "<<c[w]<<"    "<<_initialPoints.getValue()[c[w]]<<endl;
// 		}
// 		cerr<<"------\n";
// 	}



}


template <class DataTypes>
void HexahedronFEMForceField<DataTypes>::reinit()
{
    switch(method)
    {
    case LARGE :
    {
        unsigned int i=0;
        typename VecElement::const_iterator it;
        for(it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i)
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
        for(it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i)
        {
            computeMaterialStiffness(i);
            initPolar(i,*it);
        }
        break;
    }
    case FAST :
    {
        unsigned int i=0;
        typename VecElement::const_iterator it;
        for(it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i)
        {
            computeMaterialStiffness(i);
            initFast(i,*it);
        }
        break;
    }
    }
}



/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////




template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::addForce (VecDeriv& f, const VecCoord& p, const VecDeriv& /*v*/)
{
    f.resize(p.size());

    unsigned int i=0;
    typename VecElement::const_iterator it;



    switch(method)
    {
    case LARGE :
    {
        for(it=_indexedElements->begin(); it!=_indexedElements->end(); ++it,++i)
        {
            if (_trimgrid && !_trimgrid->isCubeActive(i/6)) continue;
            accumulateForceLarge( f, p, i, *it );
        }
        break;
    }
    case POLAR :
    {
        for(it=_indexedElements->begin(); it!=_indexedElements->end(); ++it,++i)
        {
            if (_trimgrid && !_trimgrid->isCubeActive(i/6)) continue;
            accumulateForcePolar( f, p, i, *it );
        }
        break;
    }
    case FAST :
    {
        for(it=_indexedElements->begin(); it!=_indexedElements->end(); ++it,++i)
        {
            if (_trimgrid && !_trimgrid->isCubeActive(i/6)) continue;
            accumulateForceFast( f, p, i, *it );
        }
        break;
    }
    }


}

template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::addDForce (VecDeriv& v, const VecDeriv& x)
{
    if( v.size()!=x.size() ) v.resize(x.size());


    unsigned int i=0;
    typename VecElement::const_iterator it;

    for(it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i)
    {
        if (_trimgrid && !_trimgrid->isCubeActive(i/6)) continue;

        Transformation R_0_2;
        R_0_2.transpose(_rotations[i]);

        Displacement X;

        for(int w=0; w<8; ++w)
        {
            Coord x_2;
            x_2 = R_0_2 * x[(*it)[_indices[w]]];
            X[w*3] = x_2[0];
            X[w*3+1] = x_2[1];
            X[w*3+2] = x_2[2];
        }

        Displacement F;
        computeForce( F, X, _elementStiffnesses.getValue()[i] );

        for(int w=0; w<8; ++w)
            v[(*it)[_indices[w]]] -= _rotations[i] * Deriv( F[w*3],  F[w*3+1],  F[w*3+2]  );
    }

}

template <class DataTypes>
double HexahedronFEMForceField<DataTypes>::getPotentialEnergy(const VecCoord&)
{
    std::cerr<<"HexahedronFEMForceField::getPotentialEnergy-not-implemented !!!"<<std::endl;
    return 0;
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
void HexahedronFEMForceField<DataTypes>::computeElementStiffness( ElementStiffness &K, const MaterialStiffness &M, const helper::fixed_array<Coord,8> &nodes, const int elementIndice)
{
    const bool verbose = this->f_printLog.getValue() && (elementIndice==0);
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
    double detJ = 1.0;
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
        if (verbose)
        {
            std::cout << "J = "<<J<<std::endl;
            std::cout << "invJ = "<<J_1<<std::endl;
            std::cout << "detJ = "<<detJ<<std::endl;
        }
    }
    else
        std::cout << "Hexa "<<elementIndice<<" is NOT a parallelepiped.\n";

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
                        J[c][0] = (nodes[1][c]-nodes[0][c])*(1-x2)*(1-x3)/8+(nodes[2][c]-nodes[3][c])*(1+x2)*(1-x3)/8+(nodes[5][c]-nodes[4][c])*(1-x2)*(1+x3)/8+(nodes[6][c]-nodes[7][c])*(1+x2)*(1+x3)/8;
                        J[c][1] = (nodes[3][c]-nodes[0][c])*(1-x1)*(1-x3)/8+(nodes[2][c]-nodes[1][c])*(1+x1)*(1-x3)/8+(nodes[7][c]-nodes[4][c])*(1-x1)*(1+x3)/8+(nodes[6][c]-nodes[5][c])*(1+x1)*(1+x3)/8;
                        J[c][2] = (nodes[4][c]-nodes[0][c])*(1-x1)*(1-x2)/8+(nodes[5][c]-nodes[1][c])*(1+x1)*(1-x2)/8+(nodes[6][c]-nodes[2][c])*(1+x1)*(1+x2)/8+(nodes[7][c]-nodes[3][c])*(1-x1)*(1+x2)/8;
                    }
                    detJ = defaulttype::determinant(J);
                    J_1.invert(J);
                    J_1t.transpose(J_1);
                    if (verbose)
                    {
                        std::cout << "J = "<<J<<std::endl;
                        std::cout << "invJ = "<<J_1<<std::endl;
                        std::cout << "detJ = "<<detJ<<std::endl;
                    }
                }
                double qx[8];
                double qy[8];
                double qz[8];
                for(int i=0; i<8; ++i)
                {
                    // Ni = 1/8 (1+_coef[i][0]x1)(1+_coef[i][1]x2)(1+_coef[i][2]x3)
                    // qxi = dNi/dx = dNi/dx1 dx1/dx + dNi/dx2 dx2/dx + dNi/dx3 dx3/dx
                    double dNi_dx1 = (_coef[i][0])*(1+_coef[i][1]*x2)*(1+_coef[i][2]*x3)/8;
                    double dNi_dx2 = (1+_coef[i][0]*x1)*(_coef[i][1])*(1+_coef[i][2]*x3)/8;
                    double dNi_dx3 = (1+_coef[i][0]*x1)*(1+_coef[i][1]*x2)*(_coef[i][2])/8;
                    if (verbose) std::cout << "dN"<<i<<"/dxi = "<<dNi_dx1<<" "<<dNi_dx2<<" "<<dNi_dx3<<"\n";
#ifdef DN_USE_J
                    qx[i] = dNi_dx1*J_1[0][0] + dNi_dx2*J_1[1][0] + dNi_dx3*J_1[2][0];
                    qy[i] = dNi_dx1*J_1[0][1] + dNi_dx2*J_1[1][1] + dNi_dx3*J_1[2][1];
                    qz[i] = dNi_dx1*J_1[0][2] + dNi_dx2*J_1[1][2] + dNi_dx3*J_1[2][2];
                    if (verbose) std::cout << "q"<<i<<" = "<<qx[i]<<" "<<qy[i]<<" "<<qz[i]<<"\n";
#else
                    qx[i] = dNi_dx1;
                    qy[i] = dNi_dx2;
                    qz[i] = dNi_dx3;
#endif
                }
                for(int i=0; i<8; ++i)
                {
                    Mat<6,3,Real> MBi;
                    //MBi[0][0] = M[0][0] * qx[i]; MBi[0][1] = M[0][1] * qy[i]; MBi[0][2] = M[0][2] * qz[i];
                    //MBi[1][0] = M[1][0] * qx[i]; MBi[1][1] = M[1][1] * qy[i]; MBi[1][2] = M[1][2] * qz[i];
                    //MBi[2][0] = M[2][0] * qx[i]; MBi[2][1] = M[2][1] * qy[i]; MBi[2][2] = M[2][2] * qz[i];
                    //MBi[3][0] = M[3][3] * qy[i]; MBi[3][1] = M[3][3] * qx[i]; MBi[3][2] = 0;
                    //MBi[4][0] = 0;               MBi[4][1] = M[4][4] * qz[i]; MBi[4][2] = M[4][4] * qy[i];
                    //MBi[5][0] = M[5][5] * qz[i]; MBi[5][1] = 0;               MBi[5][2] = M[5][5] * qx[i];
                    MBi[0][0] = U * qx[i]; MBi[0][1] = V * qy[i]; MBi[0][2] = V * qz[i];
                    MBi[1][0] = V * qx[i]; MBi[1][1] = U * qy[i]; MBi[1][2] = V * qz[i];
                    MBi[2][0] = V * qx[i]; MBi[2][1] = V * qy[i]; MBi[2][2] = U * qz[i];
                    MBi[3][0] = W * qy[i]; MBi[3][1] = W * qx[i]; MBi[3][2] = 0;
                    MBi[4][0] = 0;         MBi[4][1] = W * qz[i]; MBi[4][2] = W * qy[i];
                    MBi[5][0] = W * qz[i]; MBi[5][1] = 0;         MBi[5][2] = W * qx[i];
                    if (verbose) std::cout << "MB"<<i<<" = "<<MBi<<"\n";
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
                        if (verbose) std::cout << "K"<<i<<j<<" += "<<k<<" * "<<detJ<<"\n";
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
        Mat33 k = vol*integrateStiffness(  _coef[i][0], _coef[i][1],_coef[i][2],  _coef[i][0], _coef[i][1],_coef[i][2], M[0][0], M[0][1],M[3][3], J_1  );


        for(int m=0; m<3; ++m)
            for(int l=0; l<3; ++l)
            {
                K[i*3+m][i*3+l] += k[m][l];
            }



        for(int j=i+1; j<8; ++j)
        {
            Mat33 k = vol*integrateStiffness(  _coef[i][0], _coef[i][1],_coef[i][2],  _coef[j][0], _coef[j][1],_coef[j][2], M[0][0], M[0][1],M[3][3], J_1  );


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
    if (elementIndice==0)
    {
        std::cout << "nodes = "<<nodes[0]<<"  "<<nodes[1]<<"  "<<nodes[2]<<"  "<<nodes[3]<<"  "<<nodes[4]<<"  "<<nodes[5]<<"  "<<nodes[6]<<"  "<<nodes[7]<<std::endl;
        std::cout << "M = "<<M<<std::endl;
        std::cout << "K = "<<std::endl;
        for (int i=0; i<24; i++)
            std::cout << K[i] << std::endl;
        std::cout << "K1 = "<<std::endl;
        for (int i=0; i<24; i++)
            std::cout << K1[i] << std::endl;
    }
#ifdef GENERIC_STIFFNESS_MATRIX
    K=K1;
#endif

    // if sparseGrid -> the filling ratio is taken into account
    if( _sparseGrid && _sparseGrid->getType(elementIndice)==topology::SparseGridTopology::BOUNDARY)
        K *= .5;

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
    F = K*Depl;
    return;

    // taking into account null terms in K
    Real t23 = K[0][13]*Depl[13]+K[0][14]*Depl[14]+K[0][15]*Depl[15]+K[0]
            [16]*Depl[16]+K[0][17]*Depl[17]+K[0][18]*Depl[18]+K[0][19]*Depl[19]
            +K[0][20]*Depl[20]+K[0][21]*Depl[21]+K[0][22]*Depl[22]+K[0][23]*Depl
            [23];
    Real t36 = K[1][0]*Depl[0]+K[1][1]*Depl[1]+K[1][2]*Depl[2]+K[1][3]*
            Depl[3]+K[1][5]*Depl[5]+K[1][6]*Depl[6]+K[1][7]*Depl[7]+K[1][8]*
            Depl[8]+K[1][9]*Depl[9]+K[1][10]*Depl[10]+K[1][11]*Depl[11];
    Real t48 = K[1][12]*Depl[12]+K[1][14]*Depl[14]+K[1][15]*Depl[15]+K[1]
            [16]*Depl[16]+K[1][17]*Depl[17]+K[1][18]*Depl[18]+K[1][19]*Depl[19]
            +K[1][20]*Depl[20]+K[1][21]*Depl[21]+K[1][22]*Depl[22]+K[1][23]*Depl
            [23];
    Real t61 = K[2][0]*Depl[0]+K[2][1]*Depl[1]+K[2][2]*Depl[2]+K[2][3]*
            Depl[3]+K[2][4]*Depl[4]+K[2][6]*Depl[6]+K[2][7]*Depl[7]+K[2][8]*
            Depl[8]+K[2][9]*Depl[9]+K[2][10]*Depl[10]+K[2][12]*Depl[12];
    Real t73 = K[2][13]*Depl[13]+K[2][14]*Depl[14]+K[2][15]*Depl[15]+K[2]
            [16]*Depl[16]+K[2][17]*Depl[17]+K[2][18]*Depl[18]+K[2][19]*Depl[19]
            +K[2][20]*Depl[20]+K[2][21]*Depl[21]+K[2][22]*Depl[22]+K[2][23]*Depl
            [23];
    Real t97 = K[3][12]*Depl[12]+K[3][13]*Depl[13]+K[3][14]*Depl[14]+K[3]
            [16]*Depl[16]+K[3][17]*Depl[17]+K[3][18]*Depl[18]+K[3][19]*Depl[19]
            +K[3][20]*Depl[20]+K[3][21]*Depl[21]+K[3][22]*Depl[22]+K[3][23]*Depl
            [23];
    Real t110 = K[4][0]*Depl[0]+K[4][2]*Depl[2]+K[4][3]*Depl[3]+K[4][4]*
            Depl[4]+K[4][5]*Depl[5]+K[4][6]*Depl[6]+K[4][7]*Depl[7]+K[4][8]*
            Depl[8]+K[4][9]*Depl[9]+K[4][10]*Depl[10]+K[4][11]*Depl[11];
    Real t122 = K[4][12]*Depl[12]+K[4][13]*Depl[13]+K[4][14]*Depl[14]+K
            [4][15]*Depl[15]+K[4][17]*Depl[17]+K[4][18]*Depl[18]+K[4][19]*Depl[19]
            +K[4][20]*Depl[20]+K[4][21]*Depl[21]+K[4][22]*Depl[22]+K[4][23]*
            Depl[23];
    Real t135 = K[5][0]*Depl[0]+K[5][1]*Depl[1]+K[5][3]*Depl[3]+K[5][4]*
            Depl[4]+K[5][5]*Depl[5]+K[5][6]*Depl[6]+K[5][7]*Depl[7]+K[5][9]*
            Depl[9]+K[5][10]*Depl[10]+K[5][11]*Depl[11]+K[5][12]*Depl[12];
    Real t147 = K[5][13]*Depl[13]+K[5][14]*Depl[14]+K[5][15]*Depl[15]+K
            [5][16]*Depl[16]+K[5][17]*Depl[17]+K[5][18]*Depl[18]+K[5][19]*Depl[19]
            +K[5][20]*Depl[20]+K[5][21]*Depl[21]+K[5][22]*Depl[22]+K[5][23]*
            Depl[23];
    Real t171 = K[6][12]*Depl[12]+K[6][13]*Depl[13]+K[6][14]*Depl[14]+K
            [6][15]*Depl[15]+K[6][16]*Depl[16]+K[6][17]*Depl[17]+K[6][19]*Depl[19]
            +K[6][20]*Depl[20]+K[6][21]*Depl[21]+K[6][22]*Depl[22]+K[6][23]*
            Depl[23];
    Real t184 = K[7][0]*Depl[0]+K[7][1]*Depl[1]+K[7][2]*Depl[2]+K[7][3]*
            Depl[3]+K[7][4]*Depl[4]+K[7][5]*Depl[5]+K[7][6]*Depl[6]+K[7][7]*
            Depl[7]+K[7][8]*Depl[8]+K[7][9]*Depl[9]+K[7][11]*Depl[11];
    Real t196 = K[7][12]*Depl[12]+K[7][13]*Depl[13]+K[7][14]*Depl[14]+K
            [7][15]*Depl[15]+K[7][16]*Depl[16]+K[7][17]*Depl[17]+K[7][18]*Depl[18]
            +K[7][20]*Depl[20]+K[7][21]*Depl[21]+K[7][22]*Depl[22]+K[7][23]*
            Depl[23];
    Real t209 = K[8][0]*Depl[0]+K[8][1]*Depl[1]+K[8][2]*Depl[2]+K[8][3]*
            Depl[3]+K[8][4]*Depl[4]+K[8][6]*Depl[6]+K[8][7]*Depl[7]+K[8][8]*
            Depl[8]+K[8][9]*Depl[9]+K[8][10]*Depl[10]+K[8][12]*Depl[12];
    Real t221 = K[8][13]*Depl[13]+K[8][14]*Depl[14]+K[8][15]*Depl[15]+K
            [8][16]*Depl[16]+K[8][17]*Depl[17]+K[8][18]*Depl[18]+K[8][19]*Depl[19]
            +K[8][20]*Depl[20]+K[8][21]*Depl[21]+K[8][22]*Depl[22]+K[8][23]*
            Depl[23];
    Real t245 = K[9][12]*Depl[12]+K[9][13]*Depl[13]+K[9][14]*Depl[14]+K
            [9][15]*Depl[15]+K[9][16]*Depl[16]+K[9][17]*Depl[17]+K[9][18]*Depl[18]
            +K[9][19]*Depl[19]+K[9][20]*Depl[20]+K[9][22]*Depl[22]+K[9][23]*
            Depl[23];
    Real t258 = K[10][0]*Depl[0]+K[10][1]*Depl[1]+K[10][2]*Depl[2]+K[10]
            [3]*Depl[3]+K[10][4]*Depl[4]+K[10][5]*Depl[5]+K[10][6]*Depl[6]+K
            [10][8]*Depl[8]+K[10][9]*Depl[9]+K[10][10]*Depl[10]+K[10][11]*Depl[11];
    Real t270 = K[10][12]*Depl[12]+K[10][13]*Depl[13]+K[10][14]*Depl[14]+
            K[10][15]*Depl[15]+K[10][16]*Depl[16]+K[10][17]*Depl[17]+K[10][18]*
            Depl[18]+K[10][19]*Depl[19]+K[10][20]*Depl[20]+K[10][21]*Depl[21]+K
            [10][23]*Depl[23];
    Real t283 = K[11][0]*Depl[0]+K[11][1]*Depl[1]+K[11][3]*Depl[3]+K[11]
            [4]*Depl[4]+K[11][5]*Depl[5]+K[11][6]*Depl[6]+K[11][7]*Depl[7]+K
            [11][9]*Depl[9]+K[11][10]*Depl[10]+K[11][11]*Depl[11]+K[11][12]*Depl
            [12];
    Real t295 = K[11][13]*Depl[13]+K[11][14]*Depl[14]+K[11][15]*Depl[15]+
            K[11][16]*Depl[16]+K[11][17]*Depl[17]+K[11][18]*Depl[18]+K[11][19]*
            Depl[19]+K[11][20]*Depl[20]+K[11][21]*Depl[21]+K[11][22]*Depl[22]+K
            [11][23]*Depl[23];
    Real t319 = K[12][11]*Depl[11]+K[12][12]*Depl[12]+K[12][13]*Depl[13]+
            K[12][14]*Depl[14]+K[12][16]*Depl[16]+K[12][17]*Depl[17]+K[12][18]*
            Depl[18]+K[12][19]*Depl[19]+K[12][20]*Depl[20]+K[12][22]*Depl[22]+K
            [12][23]*Depl[23];
    Real t332 = K[13][0]*Depl[0]+K[13][2]*Depl[2]+K[13][3]*Depl[3]+K[13]
            [4]*Depl[4]+K[13][5]*Depl[5]+K[13][6]*Depl[6]+K[13][7]*Depl[7]+K
            [13][8]*Depl[8]+K[13][9]*Depl[9]+K[13][10]*Depl[10]+K[13][11]*Depl[11];
    Real t344 = K[13][12]*Depl[12]+K[13][13]*Depl[13]+K[13][14]*Depl[14]+
            K[13][15]*Depl[15]+K[13][17]*Depl[17]+K[13][18]*Depl[18]+K[13][19]*
            Depl[19]+K[13][20]*Depl[20]+K[13][21]*Depl[21]+K[13][22]*Depl[22]+K
            [13][23]*Depl[23];
    Real t357 = K[14][0]*Depl[0]+K[14][1]*Depl[1]+K[14][2]*Depl[2]+K[14]
            [3]*Depl[3]+K[14][4]*Depl[4]+K[14][5]*Depl[5]+K[14][6]*Depl[6]+K
            [14][7]*Depl[7]+K[14][8]*Depl[8]+K[14][9]*Depl[9]+K[14][10]*Depl[10];
    Real t369 = K[14][11]*Depl[11]+K[14][12]*Depl[12]+K[14][13]*Depl[13]+
            K[14][14]*Depl[14]+K[14][15]*Depl[15]+K[14][16]*Depl[16]+K[14][18]*
            Depl[18]+K[14][19]*Depl[19]+K[14][20]*Depl[20]+K[14][21]*Depl[21]+K
            [14][22]*Depl[22];
    Real t393 = K[15][11]*Depl[11]+K[15][13]*Depl[13]+K[15][14]*Depl[14]+
            K[15][15]*Depl[15]+K[15][16]*Depl[16]+K[15][17]*Depl[17]+K[15][19]*
            Depl[19]+K[15][20]*Depl[20]+K[15][21]*Depl[21]+K[15][22]*Depl[22]+K
            [15][23]*Depl[23];
    Real t406 = K[16][0]*Depl[0]+K[16][1]*Depl[1]+K[16][2]*Depl[2]+K[16]
            [3]*Depl[3]+K[16][5]*Depl[5]+K[16][6]*Depl[6]+K[16][7]*Depl[7]+K
            [16][8]*Depl[8]+K[16][9]*Depl[9]+K[16][10]*Depl[10]+K[16][11]*Depl[11]
            ;
    Real t418 = K[16][12]*Depl[12]+K[16][14]*Depl[14]+K[16][15]*Depl[15]+
            K[16][16]*Depl[16]+K[16][17]*Depl[17]+K[16][18]*Depl[18]+K[16][19]*
            Depl[19]+K[16][20]*Depl[20]+K[16][21]*Depl[21]+K[16][22]*Depl[22]+K
            [16][23]*Depl[23];
    Real t431 = K[17][0]*Depl[0]+K[17][1]*Depl[1]+K[17][2]*Depl[2]+K[17]
            [3]*Depl[3]+K[17][4]*Depl[4]+K[17][5]*Depl[5]+K[17][6]*Depl[6]+K
            [17][7]*Depl[7]+K[17][8]*Depl[8]+K[17][9]*Depl[9]+K[17][10]*Depl[10]
            ;
    Real t443 = K[17][11]*Depl[11]+K[17][12]*Depl[12]+K[17][13]*Depl[13]+
            K[17][15]*Depl[15]+K[17][16]*Depl[16]+K[17][17]*Depl[17]+K[17][18]*
            Depl[18]+K[17][19]*Depl[19]+K[17][21]*Depl[21]+K[17][22]*Depl[22]+K
            [17][23]*Depl[23];
    Real t467 = K[18][11]*Depl[11]+K[18][12]*Depl[12]+K[18][13]*Depl[13]+
            K[18][14]*Depl[14]+K[18][16]*Depl[16]+K[18][17]*Depl[17]+K[18][18]*
            Depl[18]+K[18][19]*Depl[19]+K[18][20]*Depl[20]+K[18][22]*Depl[22]+K
            [18][23]*Depl[23];
    Real t480 = K[19][0]*Depl[0]+K[19][1]*Depl[1]+K[19][2]*Depl[2]+K[19]
            [3]*Depl[3]+K[19][4]*Depl[4]+K[19][5]*Depl[5]+K[19][6]*Depl[6]+K
            [19][8]*Depl[8]+K[19][9]*Depl[9]+K[19][10]*Depl[10]+K[19][11]*Depl[11]
            ;
    Real t492 = K[19][12]*Depl[12]+K[19][13]*Depl[13]+K[19][14]*Depl[14]+
            K[19][15]*Depl[15]+K[19][16]*Depl[16]+K[19][17]*Depl[17]+K[19][18]*
            Depl[18]+K[19][19]*Depl[19]+K[19][20]*Depl[20]+K[19][21]*Depl[21]+K
            [19][23]*Depl[23];
    Real t505 = K[20][0]*Depl[0]+K[20][1]*Depl[1]+K[20][2]*Depl[2]+K[20]
            [3]*Depl[3]+K[20][4]*Depl[4]+K[20][5]*Depl[5]+K[20][6]*Depl[6]+K
            [20][7]*Depl[7]+K[20][8]*Depl[8]+K[20][9]*Depl[9]+K[20][10]*Depl[10]
            ;
    Real t517 = K[20][11]*Depl[11]+K[20][12]*Depl[12]+K[20][13]*Depl[13]+
            K[20][14]*Depl[14]+K[20][15]*Depl[15]+K[20][16]*Depl[16]+K[20][18]*
            Depl[18]+K[20][19]*Depl[19]+K[20][20]*Depl[20]+K[20][21]*Depl[21]+K
            [20][22]*Depl[22];
    Real t541 = K[21][11]*Depl[11]+K[21][13]*Depl[13]+K[21][14]*Depl[14]+
            K[21][15]*Depl[15]+K[21][16]*Depl[16]+K[21][17]*Depl[17]+K[21][19]*
            Depl[19]+K[21][20]*Depl[20]+K[21][21]*Depl[21]+K[21][22]*Depl[22]+K
            [21][23]*Depl[23];
    Real t554 = K[22][0]*Depl[0]+K[22][1]*Depl[1]+K[22][2]*Depl[2]+K[22]
            [3]*Depl[3]+K[22][4]*Depl[4]+K[22][5]*Depl[5]+K[22][6]*Depl[6]+K
            [22][7]*Depl[7]+K[22][8]*Depl[8]+K[22][9]*Depl[9]+K[22][11]*Depl[11]
            ;
    Real t566 = K[22][12]*Depl[12]+K[22][13]*Depl[13]+K[22][14]*Depl[14]+
            K[22][15]*Depl[15]+K[22][16]*Depl[16]+K[22][17]*Depl[17]+K[22][18]*
            Depl[18]+K[22][20]*Depl[20]+K[22][21]*Depl[21]+K[22][22]*Depl[22]+K
            [22][23]*Depl[23];
    Real t579 = K[23][0]*Depl[0]+K[23][1]*Depl[1]+K[23][2]*Depl[2]+K[23]
            [3]*Depl[3]+K[23][4]*Depl[4]+K[23][5]*Depl[5]+K[23][6]*Depl[6]+K
            [23][7]*Depl[7]+K[23][8]*Depl[8]+K[23][9]*Depl[9]+K[23][10]*Depl[10]
            ;
    Real t591 = K[23][11]*Depl[11]+K[23][12]*Depl[12]+K[23][13]*Depl[13]+
            K[23][15]*Depl[15]+K[23][16]*Depl[16]+K[23][17]*Depl[17]+K[23][18]*
            Depl[18]+K[23][19]*Depl[19]+K[23][21]*Depl[21]+K[23][22]*Depl[22]+K
            [23][23]*Depl[23];
    F[0] = K[0][0]*Depl[0]+K[0][1]*Depl[1]+K[0][2]*Depl[2]+
            K[0][4]*Depl[4]+K[0][5]*Depl[5]+K[0][6]*Depl[6]+K[0][7]*Depl[7]+K
            [0][8]*Depl[8]+K[0][10]*Depl[10]+K[0][11]*Depl[11]+t23;
    F[1] = t36+t48;
    F[2] = t61+t73;
    F[3] = K[3][1]*Depl[1]+K[3][2]*Depl[2]+K[3][3]*Depl[3]+
            K[3][4]*Depl[4]+K[3][5]*Depl[5]+K[3][7]*Depl[7]+K[3][8]*Depl[8]+K
            [3][9]*Depl[9]+K[3][10]*Depl[10]+K[3][11]*Depl[11]+t97;
    F[4] = t110+t122;
    F[5] = t135+t147;
    F[6] = K[6][0]*Depl[0]+K[6][1]*Depl[1]+K[6][2]*Depl[2]+
            K[6][4]*Depl[4]+K[6][5]*Depl[5]+K[6][6]*Depl[6]+K[6][7]*Depl[7]+K
            [6][8]*Depl[8]+K[6][10]*Depl[10]+K[6][11]*Depl[11]+t171;
    F[7] = t184+t196;
    F[8] = t209+t221;
    F[9] = K[9][1]*Depl[1]+K[9][2]*Depl[2]+K[9][3]*Depl[3]+
            K[9][4]*Depl[4]+K[9][5]*Depl[5]+K[9][7]*Depl[7]+K[9][8]*Depl[8]+K
            [9][9]*Depl[9]+K[9][10]*Depl[10]+K[9][11]*Depl[11]+t245;
    F[10] = t258+t270;
    F[11] = t283+t295;
    F[12] = K[12][1]*Depl[1]+K[12][2]*Depl[2]+K[12][3]*Depl[3]
            +K[12][4]*Depl[4]+K[12][5]*Depl[5]+K[12][6]*Depl[6]+K[12][7]*Depl
            [7]+K[12][8]*Depl[8]+K[12][9]*Depl[9]+K[12][10]*Depl[10]+t319;
    F[13] = t332+t344;
    F[14] = t357+t369;
    F[15] = K[15][0]*Depl[0]+K[15][1]*Depl[1]+K[15][2]*Depl[2]
            +K[15][4]*Depl[4]+K[15][5]*Depl[5]+K[15][6]*Depl[6]+K[15][7]*Depl
            [7]+K[15][8]*Depl[8]+K[15][9]*Depl[9]+K[15][10]*Depl[10]+t393;
    F[16] = t406+t418;
    F[17] = t431+t443;
    F[18] = K[18][0]*Depl[0]+K[18][1]*Depl[1]+K[18][2]*Depl[2]
            +K[18][3]*Depl[3]+K[18][4]*Depl[4]+K[18][5]*Depl[5]+K[18][7]*Depl
            [7]+K[18][8]*Depl[8]+K[18][9]*Depl[9]+K[18][10]*Depl[10]+t467;
    F[19] = t480+t492;
    F[20] = t505+t517;
    F[21] = K[21][0]*Depl[0]+K[21][1]*Depl[1]+K[21][2]*Depl[2]
            +K[21][3]*Depl[3]+K[21][4]*Depl[4]+K[21][5]*Depl[5]+K[21][6]*Depl
            [6]+K[21][7]*Depl[7]+K[21][8]*Depl[8]+K[21][10]*Depl[10]+t541;
    F[22] = t554+t566;
    F[23] = t579+t591;


}

template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::computeForceOptimized( Displacement &F, const Displacement &Depl, const ElementStiffness &K )
{
    // taking into account null terms in K and in Depl
    Real t16 = K[0][6]*Depl[6]+K[0][7]*Depl[7]+K[0][8]*Depl[8]+K[0][10]*Depl[10]+K[0][13]*
            Depl[13]+K[0][14]*Depl[14]+K[0][15]*Depl[15]+K[0][16]*Depl[16]+K[0][17]*Depl[17]+K[0][18]
            *Depl[18]+K[0][19]*Depl[19]+K[0][20]*Depl[20]+K[0][21]*Depl[21]+K[0][22]*Depl[22]+K[0]
            [23]*Depl[23];
    Real t34 = K[1][3]*Depl[3]+K[1][6]*Depl[6]+K[1][7]*Depl[7]+K[1][8]*Depl[8]+K[1][9]*Depl
            [9]+K[1][10]*Depl[10]+K[1][12]*Depl[12]+K[1][14]*Depl[14]+K[1][15]*Depl[15]+K[1][16]*Depl
            [16]+K[1][17]*Depl[17]+K[1][18]*Depl[18]+K[1][19]*Depl[19]+K[1][20]*Depl[20]+K[1][21]*
            Depl[21]+K[1][22]*Depl[22]+K[1][23]*Depl[23];
    Real t53 = K[2][3]*Depl[3]+K[2][6]*Depl[6]+K[2][7]*Depl[7]+K[2][8]*Depl[8]+K[2][9]*Depl
            [9]+K[2][10]*Depl[10]+K[2][12]*Depl[12]+K[2][13]*Depl[13]+K[2][14]*Depl[14]+K[2][15]*Depl
            [15]+K[2][16]*Depl[16]+K[2][17]*Depl[17]+K[2][18]*Depl[18]+K[2][19]*Depl[19]+K[2][20]*
            Depl[20]+K[2][21]*Depl[21]+K[2][22]*Depl[22]+K[2][23]*Depl[23];
    Real t70 = K[3][3]*Depl[3]+K[3][7]*Depl[7]+K[3][8]*Depl[8]+K[3][9]*Depl[9]+K[3][10]*Depl
            [10]+K[3][12]*Depl[12]+K[3][13]*Depl[13]+K[3][14]*Depl[14]+K[3][16]*Depl[16]+K[3][17]*
            Depl[17]+K[3][18]*Depl[18]+K[3][19]*Depl[19]+K[3][20]*Depl[20]+K[3][21]*Depl[21]+K[3][22]
            *Depl[22]+K[3][23]*Depl[23];
    Real t88 = K[4][3]*Depl[3]+K[4][6]*Depl[6]+K[4][7]*Depl[7]+K[4][8]*Depl[8]+K[4][9]*Depl
            [9]+K[4][10]*Depl[10]+K[4][12]*Depl[12]+K[4][13]*Depl[13]+K[4][14]*Depl[14]+K[4][15]*Depl
            [15]+K[4][17]*Depl[17]+K[4][18]*Depl[18]+K[4][19]*Depl[19]+K[4][20]*Depl[20]+K[4][21]*
            Depl[21]+K[4][22]*Depl[22]+K[4][23]*Depl[23];
    Real t106 = K[5][3]*Depl[3]+K[5][6]*Depl[6]+K[5][7]*Depl[7]+K[5][9]*Depl[9]+K[5][10]*
            Depl[10]+K[5][12]*Depl[12]+K[5][13]*Depl[13]+K[5][14]*Depl[14]+K[5][15]*Depl[15]+K[5][16]
            *Depl[16]+K[5][17]*Depl[17]+K[5][18]*Depl[18]+K[5][19]*Depl[19]+K[5][20]*Depl[20]+K[5]
            [21]*Depl[21]+K[5][22]*Depl[22]+K[5][23]*Depl[23];
    Real t122 = K[6][6]*Depl[6]+K[6][7]*Depl[7]+K[6][8]*Depl[8]+K[6][10]*Depl[10]+K[6][12]
            *Depl[12]+K[6][13]*Depl[13]+K[6][14]*Depl[14]+K[6][15]*Depl[15]+K[6][16]*Depl[16]+K[6]
            [17]*Depl[17]+K[6][19]*Depl[19]+K[6][20]*Depl[20]+K[6][21]*Depl[21]+K[6][22]*Depl[22]+K
            [6][23]*Depl[23];
    Real t139 = K[7][3]*Depl[3]+K[7][6]*Depl[6]+K[7][7]*Depl[7]+K[7][8]*Depl[8]+K[7][9]*Depl
            [9]+K[7][12]*Depl[12]+K[7][13]*Depl[13]+K[7][14]*Depl[14]+K[7][15]*Depl[15]+K[7][16]*Depl
            [16]+K[7][17]*Depl[17]+K[7][18]*Depl[18]+K[7][20]*Depl[20]+K[7][21]*Depl[21]+K[7][22]*
            Depl[22]+K[7][23]*Depl[23];
    Real t158 = K[8][3]*Depl[3]+K[8][6]*Depl[6]+K[8][7]*Depl[7]+K[8][8]*Depl[8]+K[8][9]*Depl
            [9]+K[8][10]*Depl[10]+K[8][12]*Depl[12]+K[8][13]*Depl[13]+K[8][14]*Depl[14]+K[8][15]*Depl
            [15]+K[8][16]*Depl[16]+K[8][17]*Depl[17]+K[8][18]*Depl[18]+K[8][19]*Depl[19]+K[8][20]*
            Depl[20]+K[8][21]*Depl[21]+K[8][22]*Depl[22]+K[8][23]*Depl[23];
    Real t175 = K[9][3]*Depl[3]+K[9][7]*Depl[7]+K[9][8]*Depl[8]+K[9][9]*Depl[9]+K[9][10]*
            Depl[10]+K[9][12]*Depl[12]+K[9][13]*Depl[13]+K[9][14]*Depl[14]+K[9][15]*Depl[15]+K[9][16]
            *Depl[16]+K[9][17]*Depl[17]+K[9][18]*Depl[18]+K[9][19]*Depl[19]+K[9][20]*Depl[20]+K[9]
            [22]*Depl[22]+K[9][23]*Depl[23];
    Real t192 = K[10][3]*Depl[3]+K[10][6]*Depl[6]+K[10][8]*Depl[8]+K[10][9]*Depl[9]+K[10]
            [10]*Depl[10]+K[10][12]*Depl[12]+K[10][13]*Depl[13]+K[10][14]*Depl[14]+K[10][15]*Depl[15]
            +K[10][16]*Depl[16]+K[10][17]*Depl[17]+K[10][18]*Depl[18]+K[10][19]*Depl[19]+K[10][20]*
            Depl[20]+K[10][21]*Depl[21]+K[10][23]*Depl[23];
    Real t210 = K[11][3]*Depl[3]+K[11][6]*Depl[6]+K[11][7]*Depl[7]+K[11][9]*Depl[9]+K[11]
            [10]*Depl[10]+K[11][12]*Depl[12]+K[11][13]*Depl[13]+K[11][14]*Depl[14]+K[11][15]*Depl[15]
            +K[11][16]*Depl[16]+K[11][17]*Depl[17]+K[11][18]*Depl[18]+K[11][19]*Depl[19]+K[11][20]*
            Depl[20]+K[11][21]*Depl[21]+K[11][22]*Depl[22]+K[11][23]*Depl[23];
    Real t227 = K[12][3]*Depl[3]+K[12][6]*Depl[6]+K[12][7]*Depl[7]+K[12][8]*Depl[8]+K[12]
            [9]*Depl[9]+K[12][10]*Depl[10]+K[12][12]*Depl[12]+K[12][13]*Depl[13]+K[12][14]*Depl[14]+K
            [12][16]*Depl[16]+K[12][17]*Depl[17]+K[12][18]*Depl[18]+K[12][19]*Depl[19]+K[12][20]*Depl
            [20]+K[12][22]*Depl[22]+K[12][23]*Depl[23];
    Real t245 = K[13][3]*Depl[3]+K[13][6]*Depl[6]+K[13][7]*Depl[7]+K[13][8]*Depl[8]+K[13]
            [9]*Depl[9]+K[13][10]*Depl[10]+K[13][12]*Depl[12]+K[13][13]*Depl[13]+K[13][14]*Depl[14]+K
            [13][15]*Depl[15]+K[13][17]*Depl[17]+K[13][18]*Depl[18]+K[13][19]*Depl[19]+K[13][20]*Depl
            [20]+K[13][21]*Depl[21]+K[13][22]*Depl[22]+K[13][23]*Depl[23];
    Real t262 = K[14][3]*Depl[3]+K[14][6]*Depl[6]+K[14][7]*Depl[7]+K[14][8]*Depl[8]+K[14]
            [9]*Depl[9]+K[14][10]*Depl[10]+K[14][12]*Depl[12]+K[14][13]*Depl[13]+K[14][14]*Depl[14]+K
            [14][15]*Depl[15]+K[14][16]*Depl[16]+K[14][18]*Depl[18]+K[14][19]*Depl[19]+K[14][20]*Depl
            [20]+K[14][21]*Depl[21]+K[14][22]*Depl[22];
    Real t278 = K[15][6]*Depl[6]+K[15][7]*Depl[7]+K[15][8]*Depl[8]+K[15][9]*Depl[9]+K[15]
            [10]*Depl[10]+K[15][13]*Depl[13]+K[15][14]*Depl[14]+K[15][15]*Depl[15]+K[15][16]*Depl[16]
            +K[15][17]*Depl[17]+K[15][19]*Depl[19]+K[15][20]*Depl[20]+K[15][21]*Depl[21]+K[15][22]*
            Depl[22]+K[15][23]*Depl[23];
    Real t296 = K[16][3]*Depl[3]+K[16][6]*Depl[6]+K[16][7]*Depl[7]+K[16][8]*Depl[8]+K[16]
            [9]*Depl[9]+K[16][10]*Depl[10]+K[16][12]*Depl[12]+K[16][14]*Depl[14]+K[16][15]*Depl[15]+K
            [16][16]*Depl[16]+K[16][17]*Depl[17]+K[16][18]*Depl[18]+K[16][19]*Depl[19]+K[16][20]*Depl
            [20]+K[16][21]*Depl[21]+K[16][22]*Depl[22]+K[16][23]*Depl[23];
    Real t313 = K[17][3]*Depl[3]+K[17][6]*Depl[6]+K[17][7]*Depl[7]+K[17][8]*Depl[8]+K[17]
            [9]*Depl[9]+K[17][10]*Depl[10]+K[17][12]*Depl[12]+K[17][13]*Depl[13]+K[17][15]*Depl[15]+K
            [17][16]*Depl[16]+K[17][17]*Depl[17]+K[17][18]*Depl[18]+K[17][19]*Depl[19]+K[17][21]*Depl
            [21]+K[17][22]*Depl[22]+K[17][23]*Depl[23];
    Real t329 = K[18][3]*Depl[3]+K[18][7]*Depl[7]+K[18][8]*Depl[8]+K[18][9]*Depl[9]+K[18]
            [10]*Depl[10]+K[18][12]*Depl[12]+K[18][13]*Depl[13]+K[18][14]*Depl[14]+K[18][16]*Depl[16]
            +K[18][17]*Depl[17]+K[18][18]*Depl[18]+K[18][19]*Depl[19]+K[18][20]*Depl[20]+K[18][22]*
            Depl[22]+K[18][23]*Depl[23];
    Real t346 = K[19][3]*Depl[3]+K[19][6]*Depl[6]+K[19][8]*Depl[8]+K[19][9]*Depl[9]+K[19]
            [10]*Depl[10]+K[19][12]*Depl[12]+K[19][13]*Depl[13]+K[19][14]*Depl[14]+K[19][15]*Depl[15]
            +K[19][16]*Depl[16]+K[19][17]*Depl[17]+K[19][18]*Depl[18]+K[19][19]*Depl[19]+K[19][20]*
            Depl[20]+K[19][21]*Depl[21]+K[19][23]*Depl[23];
    Real t363 = K[20][3]*Depl[3]+K[20][6]*Depl[6]+K[20][7]*Depl[7]+K[20][8]*Depl[8]+K[20]
            [9]*Depl[9]+K[20][10]*Depl[10]+K[20][12]*Depl[12]+K[20][13]*Depl[13]+K[20][14]*Depl[14]+K
            [20][15]*Depl[15]+K[20][16]*Depl[16]+K[20][18]*Depl[18]+K[20][19]*Depl[19]+K[20][20]*Depl
            [20]+K[20][21]*Depl[21]+K[20][22]*Depl[22];
    Real t379 = K[21][3]*Depl[3]+K[21][6]*Depl[6]+K[21][7]*Depl[7]+K[21][8]*Depl[8]+K[21]
            [10]*Depl[10]+K[21][13]*Depl[13]+K[21][14]*Depl[14]+K[21][15]*Depl[15]+K[21][16]*Depl[16]
            +K[21][17]*Depl[17]+K[21][19]*Depl[19]+K[21][20]*Depl[20]+K[21][21]*Depl[21]+K[21][22]*
            Depl[22]+K[21][23]*Depl[23];
    Real t396 = K[22][3]*Depl[3]+K[22][6]*Depl[6]+K[22][7]*Depl[7]+K[22][8]*Depl[8]+K[22]
            [9]*Depl[9]+K[22][12]*Depl[12]+K[22][13]*Depl[13]+K[22][14]*Depl[14]+K[22][15]*Depl[15]+K
            [22][16]*Depl[16]+K[22][17]*Depl[17]+K[22][18]*Depl[18]+K[22][20]*Depl[20]+K[22][21]*Depl
            [21]+K[22][22]*Depl[22]+K[22][23]*Depl[23];
    Real t413 = K[23][3]*Depl[3]+K[23][6]*Depl[6]+K[23][7]*Depl[7]+K[23][8]*Depl[8]+K[23]
            [9]*Depl[9]+K[23][10]*Depl[10]+K[23][12]*Depl[12]+K[23][13]*Depl[13]+K[23][15]*Depl[15]+K
            [23][16]*Depl[16]+K[23][17]*Depl[17]+K[23][18]*Depl[18]+K[23][19]*Depl[19]+K[23][21]*Depl
            [21]+K[23][22]*Depl[22]+K[23][23]*Depl[23];
    F[0] = t16;
    F[1] = t34;
    F[2] = t53;
    F[3] = t70;
    F[4] = t88;
    F[5] = t106;
    F[6] = t122;
    F[7] = t139;
    F[8] = t158;
    F[9] = t175;
    F[10] = t192;
    F[11] = t210;
    F[12] = t227;
    F[13] = t245;
    F[14] = t262;
    F[15] = t278;
    F[16] = t296;
    F[17] = t313;
    F[18] = t329;
    F[19] = t346;
    F[20] = t363;
    F[21] = t379;
    F[22] = t396;
    F[23] = t413;
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



    Vec<8,Coord> nodes;
    for(int w=0; w<8; ++w)
        nodes[w] = _initialPoints.getValue()[elem[_indices[w]]];


    Coord horizontal;
    horizontal = (nodes[1]-nodes[0] + nodes[2]-nodes[3] + nodes[5]-nodes[4] + nodes[6]-nodes[7])*.25;
    Coord vertical;
    vertical = (nodes[3]-nodes[0] + nodes[2]-nodes[1] + nodes[7]-nodes[4] + nodes[6]-nodes[5])*.25;
    Transformation R_0_1;
    computeRotationLarge( R_0_1, horizontal,vertical);

    for(int w=0; w<8; ++w)
        _rotatedInitialElements[i][w] = R_0_1*_initialPoints.getValue()[elem[_indices[w]]];


    if( _elementStiffnesses.getValue().size() <= (unsigned)i )
    {
        _elementStiffnesses.beginEdit()->resize( _elementStiffnesses.getValue().size()+1 );
        computeElementStiffness( (*_elementStiffnesses.beginEdit())[i], _materialsStiffnesses[i], _rotatedInitialElements[i], i );
    }


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
void HexahedronFEMForceField<DataTypes>::accumulateForceLarge( Vector& f, const Vector & p, int i, const Element&elem )
{
    Vec<8,Coord> nodes;
    for(int w=0; w<8; ++w)
        nodes[w] = p[elem[_indices[w]]];

    Coord horizontal;
    horizontal = (nodes[1]-nodes[0] + nodes[2]-nodes[3] + nodes[5]-nodes[4] + nodes[6]-nodes[7])*.25;
    Coord vertical;
    vertical = (nodes[3]-nodes[0] + nodes[2]-nodes[1] + nodes[7]-nodes[4] + nodes[6]-nodes[5])*.25;

    Transformation R_0_2; // Rotation matrix (deformed and displaced Tetrahedron/world)
    computeRotationLarge( R_0_2, horizontal,vertical);

    _rotations[i].transpose(R_0_2);

    // positions of the deformed and displaced Tetrahedre in its frame
    Vec<8,Coord> deformed;
    for(int w=0; w<8; ++w)
        deformed[w] = R_0_2 * nodes[w];


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
        computeElementStiffness( (*_elementStiffnesses.beginEdit())[i], _materialsStiffnesses[i], deformed, i );


    Displacement F; //forces
    computeForce( F, D, _elementStiffnesses.getValue()[i] ); // compute force on element

    for(int w=0; w<8; ++w)
        f[elem[_indices[w]]] += _rotations[i] * Deriv( F[w*3],  F[w*3+1],   F[w*3+2]  );
}







/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
////////////// polar decomposition method



template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::initPolar(int i, const Element& elem)
{
    Vec<8,Coord> nodes;
    for(int j=0; j<8; ++j)
        nodes[j] = _initialPoints.getValue()[elem[_indices[j]]];

    Transformation R_0_1; // Rotation matrix (deformed and displaced Tetrahedron/world)
    computeRotationPolar( R_0_1, nodes );


    for(int j=0; j<8; ++j)
    {
        _rotatedInitialElements[i][j] = R_0_1 * nodes[j];
    }


    if( _elementStiffnesses.getValue().size() <= (unsigned)i )
    {
        _elementStiffnesses.beginEdit()->resize( _elementStiffnesses.getValue().size()+1 );
        computeElementStiffness( (*_elementStiffnesses.beginEdit())[i], _materialsStiffnesses[i], _rotatedInitialElements[i], i );
    }
}



template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::computeRotationPolar( Transformation &r, Vec<8,Coord> &nodes)
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
    Mat33 S;

    polar_decomp(HT, r, S);
}


template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::accumulateForcePolar( Vector& f, const Vector & p, int i, const Element&elem )
{
    Vec<8,Coord> nodes;
    for(int j=0; j<8; ++j)
        nodes[j] = p[elem[_indices[j]]];


    Transformation R_0_2; // Rotation matrix (deformed and displaced Tetrahedron/world)
    computeRotationPolar( R_0_2, nodes );

    _rotations[i].transpose( R_0_2 );


    // positions of the deformed and displaced Tetrahedre in its frame
    Vec<8,Coord> deformed;
    for(int j=0; j<8; ++j)
        deformed[j] = R_0_2 * nodes[j];



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
        computeElementStiffness( (*_elementStiffnesses.beginEdit())[i], _materialsStiffnesses[i], deformed, i );

    // compute force on element
    computeForce( F, D, _elementStiffnesses.getValue()[i] );


    for(int j=0; j<8; ++j)
        f[elem[_indices[j]]] += _rotations[i] * Deriv( F[j*3],  F[j*3+1],   F[j*3+2]  );
}



/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
////////////// fast displacements method


template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::initFast(int i, const Element &elem)
{
    // Rotation matrix (initial Tetrahedre/world)
    // first vector on first edge
    // second vector in the plane of the two first edges
    // third vector orthogonal to first and second



    Vec<8,Coord> nodes;
    for(int j=0; j<8; ++j)
        nodes[j] = _initialPoints.getValue()[elem[_indices[j]]];

    Transformation R_0_1; // Rotation matrix (deformed and displaced Tetrahedron/world)
    computeRotationFast( R_0_1, nodes );


    for(int j=0; j<8; ++j)
    {
        _rotatedInitialElements[i][j] = R_0_1 * (nodes[j]-nodes[0]);
    }


    if( _elementStiffnesses.getValue().size() <= (unsigned)i )
    {
        _elementStiffnesses.beginEdit()->resize( _elementStiffnesses.getValue().size()+1 );
        computeElementStiffness( (*_elementStiffnesses.beginEdit())[i], _materialsStiffnesses[i], nodes, i );
    }
}

template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::computeRotationFast( Transformation &r, Vec<8,Coord> &nodes)
{
    // first vector on first edge
    // second vector in the plane of the two first edges
    // third vector orthogonal to first and second
    Coord edgex = nodes[1]-nodes[0];
    edgex.normalize();

    Coord edgey = nodes[3]-nodes[0];
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
void HexahedronFEMForceField<DataTypes>::accumulateForceFast( Vector& f, const Vector & p, int i, const Element&elem )
{
    Vec<8,Coord> nodes;
    for(int w=0; w<8; ++w)
        nodes[w] = p[elem[_indices[w]]];


    Transformation R_0_2; // Rotation matrix (deformed and displaced Tetrahedron/world)
    computeRotationFast( R_0_2, nodes);

    _rotations[i].transpose(R_0_2);

    // positions of the deformed and displaced Tetrahedre in its frame
    Vec<8,Coord> deformed;
    for(int w=0; w<8; ++w)
        deformed[w] = R_0_2 * (nodes[w] - nodes[0]);


// 	cerr<<deformed<<endl<<endl;


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
        computeElementStiffness( (*_elementStiffnesses.beginEdit())[i], _materialsStiffnesses[i], deformed, i );


    Displacement F; //forces
    computeForceOptimized( F, D, _elementStiffnesses.getValue()[i] ); // compute force on element

    for(int w=0; w<8; ++w)
        f[elem[_indices[w]]] += _rotations[i] * Deriv( F[w*3],  F[w*3+1],   F[w*3+2]  );
}



/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////


template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::addKToMatrix(sofa::defaulttype::BaseMatrix *mat, double k, unsigned int &offset)
{
    // Build Matrix Block for this ForceField
    int i,j,n1, n2, e;

    typename VecElement::const_iterator it;

    Index node1, node2;

    for(it = _indexedElements->begin(), e=0 ; it != _indexedElements->end() ; ++it,++e)
    {
        const ElementStiffness &Ke = _elementStiffnesses.getValue()[e];
        const Transformation& Rt = _rotations[e];
        Transformation R; R.transpose(Rt);

        // find index of node 1
        for (n1=0; n1<8; n1++)
        {
            node1 = (*it)[_indices[n1]];
            // find index of node 2
            for (n2=0; n2<8; n2++)
            {
                node2 = (*it)[_indices[n2]];
                Mat33 tmp = Rt * Mat33(Coord(Ke[3*n1+0][3*n2+0],Ke[3*n1+0][3*n2+1],Ke[3*n1+0][3*n2+2]),
                        Coord(Ke[3*n1+1][3*n2+0],Ke[3*n1+1][3*n2+1],Ke[3*n1+1][3*n2+2]),
                        Coord(Ke[3*n1+2][3*n2+0],Ke[3*n1+2][3*n2+1],Ke[3*n1+2][3*n2+2])) * R;
                for(i=0; i<3; i++)
                    for (j=0; j<3; j++)
                        mat->add(offset+3*node1+i, offset+3*node2+j, - tmp[i][j]*k);
            }
        }
    }
}




template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::draw()
{
// 	cerr<<"HexahedronFEMForceField<DataTypes>::draw()\n";
    if (!getContext()->getShowForceFields()) return;
    if (!this->mstate) return;


    const VecCoord& x = *this->mstate->getX();

    if (getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glDisable(GL_LIGHTING);

    typename VecElement::const_iterator it;
    int i;
    for(it = _indexedElements->begin(), i = 0 ; it != _indexedElements->end() ; ++it, ++i)
    {
        if (_trimgrid && !_trimgrid->isCubeActive(i/6)) continue;
        Index a = (*it)[0];
        Index b = (*it)[1];
        Index d = (*it)[2];
        Index c = (*it)[3];
        Index e = (*it)[4];
        Index f = (*it)[5];
        Index h = (*it)[6];
        Index g = (*it)[7];

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
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }


// 		if( _sparseGrid && _sparseGrid->getType(i)==topology::SparseGridTopology::BOUNDARY )
// 			continue;

        glColor4f(0.7f, 0.7f, 0.1f, (_sparseGrid && _sparseGrid->getType(i)==topology::SparseGridTopology::BOUNDARY?.5f:1.0f));
        glBegin(GL_POLYGON);
        helper::gl::glVertexT(pa);
        helper::gl::glVertexT(pb);
        helper::gl::glVertexT(pc);
        helper::gl::glVertexT(pd);
        glEnd();
        glColor4f(0.7f, 0, 0, (_sparseGrid && _sparseGrid->getType(i)==topology::SparseGridTopology::BOUNDARY?.5f:1.0f));
        glBegin(GL_POLYGON);
        helper::gl::glVertexT(pe);
        helper::gl::glVertexT(pf);
        helper::gl::glVertexT(pg);
        helper::gl::glVertexT(ph);
        glEnd();
        glColor4f(0, 0.7f, 0, (_sparseGrid && _sparseGrid->getType(i)==topology::SparseGridTopology::BOUNDARY?.5f:1.0f));
        glBegin(GL_POLYGON);
        helper::gl::glVertexT(pc);
        helper::gl::glVertexT(pd);
        helper::gl::glVertexT(ph);
        helper::gl::glVertexT(pg);
        glEnd();
        glColor4f(0, 0, 0.7f, (_sparseGrid && _sparseGrid->getType(i)==topology::SparseGridTopology::BOUNDARY?.5f:1.0f));
        glBegin(GL_POLYGON);
        helper::gl::glVertexT(pa);
        helper::gl::glVertexT(pb);
        helper::gl::glVertexT(pf);
        helper::gl::glVertexT(pe);
        glEnd();
        glColor4f(0.1f, 0.7f, 0.7f, (_sparseGrid && _sparseGrid->getType(i)==topology::SparseGridTopology::BOUNDARY?.5f:1.0f));
        glBegin(GL_POLYGON);
        helper::gl::glVertexT(pa);
        helper::gl::glVertexT(pd);
        helper::gl::glVertexT(ph);
        helper::gl::glVertexT(pe);
        glEnd();
        glColor4f(0.7f, 0.1f, 0.7f, (_sparseGrid && _sparseGrid->getType(i)==topology::SparseGridTopology::BOUNDARY?.5f:1.0f));
        glBegin(GL_POLYGON);
        helper::gl::glVertexT(pb);
        helper::gl::glVertexT(pc);
        helper::gl::glVertexT(pg);
        helper::gl::glVertexT(pf);
        glEnd();
    }

    if (getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    if(_sparseGrid )
        glDisable(GL_BLEND);
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
