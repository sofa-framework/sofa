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
#ifndef SOFA_COMPONENT_FORCEFIELD_TETRAHEDRONFEMFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_TETRAHEDRONFEMFORCEFIELD_INL

#include <sofa/core/componentmodel/behavior/ForceField.inl>
#include <sofa/component/forcefield/HexahedronFEMForceField.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/topology/GridTopology.h>
#include <sofa/helper/PolarDecompose.h>
#include <sofa/helper/gl/template.h>
#include <assert.h>
#include <iostream>
#include <set>
#include <GL/gl.h>
using std::cerr;
using std::endl;
using std::set;



// WARNING: indices ordering is different than in topology node
//
//        ^ Y
//        |
// 	      7---------6
//       /	       /|
//      /	      / |
//     3---------2  |
//     |		 |  |
//     |  4------|--5-->X
//     | / 	     | /
//     |/	     |/
//     0---------1
//    /
//   Z




namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;


template<class DataTypes> const int HexahedronFEMForceField<DataTypes>::_indices[8] = {0,1,3,2,4,5,7,6};


template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::parse(core::objectmodel::BaseObjectDescription* arg)
{
    this->core::componentmodel::behavior::ForceField<DataTypes>::parse(arg);
    std::string method = arg->getAttribute("method","");
    if (method == "large")
        this->setMethod(LARGE);
    else if (method == "polar")
        this->setMethod(POLAR);
}

template <class DataTypes>
void HexahedronFEMForceField<DataTypes>::init()
{
    this->core::componentmodel::behavior::ForceField<DataTypes>::init();
    _mesh = dynamic_cast<sofa::component::topology::MeshTopology*>(this->getContext()->getTopology());
    if ( _mesh==NULL ||  _mesh->getNbCubes()<=0 )
    {
        std::cerr << "ERROR(HexahedronFEMForceField): object must have a hexahedric MeshTopology.\n";
        return;
    }
    if (!_mesh->getCubes().empty())
    {
        _indexedElements = & (_mesh->getCubes());
    }

    if (_initialPoints.getValue().size() == 0)
    {
        VecCoord& p = *this->mstate->getX();
        _initialPoints.setValue(p);
    }

    _materialsStiffnesses.resize(_indexedElements->size() );
    _rotations.resize( _indexedElements->size() );
    _rotatedInitialElements.resize(_indexedElements->size());
    _elementStiffnesses.resize(_indexedElements->size());
    // 	_stiffnesses.resize( _initialPoints.getValue().size()*3 ); // assembly ?

    reinit();
}


template <class DataTypes>
void HexahedronFEMForceField<DataTypes>::reinit()
{
    switch(f_method.getValue())
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

    if(f_method.getValue()==LARGE)
        for(it=_indexedElements->begin(); it!=_indexedElements->end(); ++it,++i)
        {
            if (_trimgrid && !_trimgrid->isCubeActive(i/6)) continue;
            accumulateForceLarge( f, p, i, *it );
        }
    else
        for(it=_indexedElements->begin(); it!=_indexedElements->end(); ++it,++i)
        {
            if (_trimgrid && !_trimgrid->isCubeActive(i/6)) continue;
            accumulateForcePolar( f, p, i, *it );
        }
}

template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::addDForce (VecDeriv& v, const VecDeriv& x)
{
    v.resize(x.size());

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
        computeForce( F, X, _elementStiffnesses[i] );

        for(int w=0; w<8; ++w)
            v[(*it)[_indices[w]]] -= _rotations[i] * Deriv( F[w*3],  F[w*3+1],  F[w*3+2]  );
    }
}

template <class DataTypes>
double HexahedronFEMForceField<DataTypes>::getPotentialEnergy(const VecCoord&)
{
    cerr<<"HexahedronFEMForceField::getPotentialEnergy-not-implemented !!!"<<endl;
    return 0;
}





/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////






template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::computeElementStiffness( ElementStiffness &K, const MaterialStiffness &M, const Vec<8,Coord> &nodes)
{
    Mat33 J_1; // only accurate for orthogonal regular hexa
// 	m_assign( J_1, 0.0 );
    Coord l = nodes[6] - nodes[0];
    J_1[0][0]=2.0 / l[0];
    J_1[1][1]=2.0 / l[1];
    J_1[2][2]=2.0 / l[2];


    Real vol = (nodes[1]-nodes[0]).norm()*(nodes[3]-nodes[0]).norm()*(nodes[4]-nodes[0]).norm();
// 	Real vol_8 = 1.0 / 8.0;


    for(int i=0; i<8; ++i)
    {
        Mat33 k = vol*integrateStiffness( -1,1,-1,1,-1,1, _coef[i][0], _coef[i][1],_coef[i][2],  _coef[i][0], _coef[i][1],_coef[i][2], M[0][0], M[0][1],M[2][2], J_1  );


        for(int m=0; m<3; ++m)
            for(int l=0; l<3; ++l)
            {
                K[i*3+m][i*3+l] += k[m][l];
            }



        for(int j=i+1; j<8; ++j)
        {
            Mat33 k = vol*integrateStiffness( -1,1,-1,1,-1,1, _coef[i][0], _coef[i][1],_coef[i][2],  _coef[j][0], _coef[j][1],_coef[j][2], M[0][0], M[0][1],M[2][2], J_1  );


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

    // faut il multiplier par le volume ????
// 	m_teq(K,vol);
}




template<class DataTypes>
typename HexahedronFEMForceField<DataTypes>::Mat33 HexahedronFEMForceField<DataTypes>::integrateStiffness( const Real xmin, const Real xmax, const Real ymin, const Real ymax, const Real zmin, const Real zmax, int signx0, int signy0, int signz0, int signx1, int signy1, int signz1, const Real u, const Real v, const Real w, const Mat33& J_1  )
{
    Real t1 = signx0*signy0;
    Real t2 = signz0*w;
    Real t3 = t2*signx1;
    Real t5 = xmax-xmin;
    Real t6 = t5*signy1/2.0;
    Real t7 = ymax-ymin;
    Real t9 = zmax-zmin;
    Real t11 = t6*t7*signz1*t9/4.0;
    Real t13 = t1*t3*t11/36.0;
    Real t14 = signy0*signz0;

    Real t17 = 1.0+signx1*(xmax+xmin)/2.0;
    Real t18 = w*t17;
    Real t20 = signy1*t7/2.0;
    Real t21 = signz1*t9/2.0;
    Real t22 = t20*t21;
    Real t23 = t14*t18*t22;
    Real t25 = signx0*signz0;
    Real t26 = u*signx1;
    Real t30 = 1.0+signy1*(ymax+ymin)/2.0;
    Real t31 = t5*t30/2.0;
    Real t32 = t31*t21;
    Real t34 = t25*t26*t32/24.0;
    Real t35 = signz0*u;

    Real t39 = t1*t35*signx1*t11/72.0;
    Real t43 = 1.0+signz1*(zmax+zmin)/2.0;
    Real t45 = t6*t7*t43/2.0;
    Real t47 = t1*t26*t45/24.0;
    Real t48 = w*signx1;
    Real t49 = t1*t48;
    Real t50 = t49*t45;
    Real t51 = t50/24.0;
    Real t52 = t2*t17;
    Real t54 = t30*signz1*t9/2.0;
    Real t56 = t52*t54/8.0;

    Real t57 = signy0*w;
    Real t58 = t57*t17;
    Real t59 = t20*t43;
    Real t61 = t58*t59/8.0;
    Real t62 = t25*t48;
    Real t63 = t62*t32;
    Real t64 = t63/24.0;
    Real t67 = t31*t43;
    Real t71 = v*t17;
    Real t75 = t14*t48;
    Real t79 = signx0*v*t17;
    Real t96 = v*signx1;
    Real t97 = t14*t96;
    Real t103 = signy0*v;

    Real t107 = signx0*w;
    Real t108 = t107*t17;
    Real t116 = t14*u*t17*t22/24.0;
    Real t117 = t23/24.0;
    Real t124 = t107*signx1*t67/8.0;
    Real t142 = signz0*v;

    Mat33 K;
    K[0][0] = t13+t23/12.0+t34+t39+t47+t51+t56+t61+t64+signx0*u*signx1*t67/8.0;
    K[0][1] = t25*t71*t22/24.0+t75*t32/24.0+t79*t59/8.0+t57*signx1*t67/8.0;
    K[0][2] = t1*t71*t22/24.0+t75*t45/24.0+t79*t54/8.0+t3*t67/8.0;
    K[1][0] = t97*t32/24.0+t25*t18*t22/24.0+t103*signx1*t67/8.0+t108*t59/8.0;
    K[1][1] = t13+t39+t63/12.0+t116+t51+t56+t117+t47+signy0*u*t17*t59/8.0+t124;
    K[1][2] = t1*t96*t32/24.0+t62*t45/24.0+t103*t17*t54/8.0+t52*t59/8.0;
    K[2][0] = t97*t45/24.0+t1*t18*t22/24.0+t142*signx1*t67/8.0+t108*t54/8.0;
    K[2][1] = t25*t96*t45/24.0+t49*t32/24.0+t142*t17*t59/8.0+t58*t54/8.0;
    K[2][2] = t13+t64+t117+t39+t50/12.0+t61+t116+t34+t35*t17*t54/8.0+t124;

    return J_1 * K;
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
}

template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::computeForce( Displacement &F, const Displacement &Depl, const ElementStiffness &K )
{
    F = K*Depl;
}









/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
////////////// large displacements method


template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::initLarge(int i, const Element &elem)
{
    // Rotation matrix (initial Tetrahedre/world)
    // first vector on first edge
    // second vector in the plane of the two first edges
    // third vector orthogonal to first and second



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
        _rotatedInitialElements[i][_indices[w]] = R_0_1*_initialPoints.getValue()[elem[w]];


    computeElementStiffness( _elementStiffnesses[i], _materialsStiffnesses[i], nodes );

// 		if(i==0) cerr<<_elementStiffnesses[i]<<endl;
}

template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::computeRotationLarge( Transformation &r, Coord &edgex, Coord &edgey)
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
        computeElementStiffness( _elementStiffnesses[i], _materialsStiffnesses[i], deformed );


    Displacement F; //forces
    computeForce( F, D, _elementStiffnesses[i] ); // compute force on element

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

    computeElementStiffness( _elementStiffnesses[i], _materialsStiffnesses[i], nodes );
}



template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::computeRotationPolar( Transformation &r, Vec<8,Coord> &nodes)
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
    HT[3][0] = HT[3][1] = HT[3][2] = HT[0][3] = HT[1][3] = HT[2][3] = 0;
    HT[3][3] = 1;
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
    {
        computeElementStiffness( _elementStiffnesses[i], _materialsStiffnesses[i], deformed );
    }

    // compute force on element
    computeForce( F, D, _elementStiffnesses[i] );


    for(int j=0; j<8; ++j)
        f[elem[_indices[j]]] += _rotations[i] * Deriv( F[j*3],  F[j*3+1],   F[j*3+2]  );
}






/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////





template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::draw()
{
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
// 		if (_trimgrid && !_trimgrid->isCubeActive(i/6)) continue;
        Index a = (*it)[0];
        Index b = (*it)[1];
        Index d = (*it)[2];
        Index c = (*it)[3];
        Index e = (*it)[4];
        Index f = (*it)[5];
        Index h = (*it)[6];
        Index g = (*it)[7];
        Coord center = (x[a]+x[b]+x[c]+x[d]+x[e]+x[g]+x[f]+x[h])*0.0625;
        Coord pa = (x[a]+center)*(Real)0.666667;
        Coord pb = (x[b]+center)*(Real)0.666667;
        Coord pc = (x[c]+center)*(Real)0.666667;
        Coord pd = (x[d]+center)*(Real)0.666667;
        Coord pe = (x[e]+center)*(Real)0.666667;
        Coord pf = (x[f]+center)*(Real)0.666667;
        Coord pg = (x[g]+center)*(Real)0.666667;
        Coord ph = (x[h]+center)*(Real)0.666667;



        glColor3f(0.7, 0.7, 0.1);
        glBegin(GL_POLYGON);
        helper::gl::glVertexT(pa);
        helper::gl::glVertexT(pb);
        helper::gl::glVertexT(pc);
        helper::gl::glVertexT(pd);
        glEnd();
        glColor3f(0.7, 0, 0);
        glBegin(GL_POLYGON);
        helper::gl::glVertexT(pe);
        helper::gl::glVertexT(pf);
        helper::gl::glVertexT(pg);
        helper::gl::glVertexT(ph);
        glEnd();
        glColor3f(0, 0.7, 0);
        glBegin(GL_POLYGON);
        helper::gl::glVertexT(pc);
        helper::gl::glVertexT(pd);
        helper::gl::glVertexT(ph);
        helper::gl::glVertexT(pg);
        glEnd();
        glColor3f(0, 0, 0.7);
        glBegin(GL_POLYGON);
        helper::gl::glVertexT(pa);
        helper::gl::glVertexT(pb);
        helper::gl::glVertexT(pf);
        helper::gl::glVertexT(pe);
        glEnd();
        glColor3f(0.1, 0.7, 0.7);
        glBegin(GL_POLYGON);
        helper::gl::glVertexT(pa);
        helper::gl::glVertexT(pd);
        helper::gl::glVertexT(ph);
        helper::gl::glVertexT(pe);
        glEnd();
        glColor3f(0.7, 0.1, 0.7);
        glBegin(GL_POLYGON);
        helper::gl::glVertexT(pb);
        helper::gl::glVertexT(pc);
        helper::gl::glVertexT(pg);
        helper::gl::glVertexT(pf);
        glEnd();
    }

    if (getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::contributeToMatrixDimension(unsigned int * const nbRow, unsigned int * const nbCol)
{
    if (this->mstate)
    {
        VecDeriv& p = *this->mstate->getV();
        if (p.size() != 0)
        {
            (*nbRow) += p.size() * p[0].size();
            (*nbCol) = *nbRow;
        }
    }
}


template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::computeMatrix(sofa::defaulttype::SofaBaseMatrix */*mat*/, double /*m*/, double /*b*/, double /*k*/, unsigned int &/*offset*/)
{

}



template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::computeVector(sofa::defaulttype::SofaBaseVector */*vect*/, unsigned int &/*offset*/)
{

}


template<class DataTypes>
void HexahedronFEMForceField<DataTypes>::matResUpdatePosition(sofa::defaulttype::SofaBaseVector */*vect*/, unsigned int &/*offset*/)
{

}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
