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
#ifndef SOFA_COMPONENT_FORCEFIELD_HEXAHEDRONANDMASSFEMFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_HEXAHEDRONANDMASSFEMFORCEFIELD_INL


#include "HexahedronFEMForceFieldAndMass.h"
#include "HexahedronFEMForceField.inl"
// #include <sofa/core/componentmodel/behavior/ForceField.inl>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;
using std::cerr; using std::endl;


template<class DataTypes>
HexahedronFEMForceFieldAndMass<DataTypes>::HexahedronFEMForceFieldAndMass()
    : Mass()
    , HexahedronFEMForceField()
    , _density(dataField(&_density,(Real)1.0,"density","density == volumetric mass in english (kg.m-3)"))
{}


template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::init( )
{
    HexahedronFEMForceField::init();
    Mass::init();

    computeElementMasses();


}



template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::reinit( )
{
    HexahedronFEMForceField::reinit();
    Mass::reinit();

    computeElementMasses();
}



template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::computeElementMasses(  )
{
    _elementMasses.resize( this->_elementStiffnesses.size() );

    int i=0;
    typename VecElement::const_iterator it;
    for(it = this->_indexedElements->begin() ; it != this->_indexedElements->end() ; ++it, ++i)
    {
        Vec<8,Coord> nodes;
        for(int w=0; w<8; ++w)
            nodes[w] = this->_initialPoints.getValue()[(*it)[this->_indices[w]]];


        computeElementMass( _elementMasses[i], nodes );


    }
}

template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::computeElementMass( ElementMass &Mass, const Vec<8,Coord> &nodes)
{
    Real vol = (nodes[1]-nodes[0]).norm()*(nodes[3]-nodes[0]).norm()*(nodes[4]-nodes[0]).norm();


    Mass.clear();

    for(int i=0; i<8; ++i)
    {
        Real mass = vol * integrateMass(-1.0,1.0,-1.0,1.0,-1.0,1.0, this->_coef[i][0], this->_coef[i][1],this->_coef[i][2],  this->_coef[i][0], this->_coef[i][1],this->_coef[i][2]);

        Mass[i*3][i*3] += mass;
        Mass[i*3+1][i*3+1] += mass;
        Mass[i*3+2][i*3+2] += mass;



        for(int j=i+1; j<8; ++j)
        {
            Real mass = vol * integrateMass(-1.0,1.0,-1.0,1.0,-1.0,1.0,this->_coef[i][0], this->_coef[i][1],this->_coef[i][2],  this->_coef[i][0], this->_coef[i][1],this->_coef[i][2]);

            Mass[i*3][j*3] += mass;
            Mass[i*3+1][j*3+1] += mass;
            Mass[i*3+2][j*3+2] += mass;
        }
    }

    for(int i=0; i<24; ++i)
        for(int j=i+1; j<24; ++j)
        {
            Mass[j][i] = Mass[i][j];
        }
}


template<class DataTypes>
typename HexahedronFEMForceFieldAndMass<DataTypes>::Real HexahedronFEMForceFieldAndMass<DataTypes>::integrateMass( const Real xmin, const Real xmax, const Real ymin, const Real ymax, const Real zmin, const Real zmax, int signx0, int signy0, int signz0, int signx1, int signy1, int signz1  )
{
    Real t3 = xmax-xmin;
    Real t6 = 1.0f+signy1*(ymax+ymin)/2.0f;
    Real t7 = t3*t6/2.0f;
    Real t8 = zmax-zmin;
    Real t9 = signz1*t8/2.0f;
    Real t16 = 1.0f+signx1*(xmax+xmin)/2.0f;
    Real t18 = ymax-ymin;
    Real t19 = signy1*t18/2.0f;
    Real t23 = (Real)(signx0*signy0);
    Real t26 = t3*signy1/2.0f;
    Real t40 = 1.0f+signz1*(zmax+zmin)/2.0f;
    return (signx0*signz0*signx1*t7*t9/72.0f+signy0*signz0*t16*t19*t9/72.0f+t23*signz0*signx1*t26*t18*signz1*t8/864.0f+signz0*t16*t6*signz1*t8/48.0f+t23*signx1*t26*t18*t40/144.0f+t16*t6*t40/8.0f+signx0*signx1*t7*t40/24.0f+signy0*t16*t19*t40/24.0f)*_density.getValue();
}



template<class DataTypes>
std::string HexahedronFEMForceFieldAndMass<DataTypes>::getTemplateName() const
{
    return HexahedronFEMForceField::getTemplateName();
}


template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::addMDx(VecDeriv& f, const VecDeriv& dx, double factor)
{
    unsigned int i=0;
    typename VecElement::const_iterator it;

    for(it=this->_indexedElements->begin(); it!=this->_indexedElements->end(); ++it,++i)
    {
        if (this->_trimgrid && !this->_trimgrid->isCubeActive(i/6)) continue;

        Vec<24, Real> actualDx, actualF;

        for(int k=0 ; k<8 ; ++k )
        {
            int indice = k*3;
            for(int j=0 ; j<3 ; ++j )
                actualDx[indice+j] = dx[(*it)[this->_indices[k]]][j];
        }

        actualF = _elementMasses[i] * actualDx;


        for(int w=0; w<8; ++w)
            f[(*it)[this->_indices[w]]] += Deriv( actualF[w*3],  actualF[w*3+1],   actualF[w*3+2]  ) * factor;

    }
}



template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::accFromF(VecDeriv& /*a*/, const VecDeriv& /*f*/)
{
    cerr<<"HexahedronFEMForceFieldAndMass<DataTypes>::accFromF not yet implemented\n";
}




template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    HexahedronFEMForceField::addForce(f,x,v);
//         Mass::addForce(f,x,v);
}



template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::addDForce(VecDeriv& df, const VecDeriv& dx)
{
    HexahedronFEMForceField::addDForce(df,dx);
}



template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::draw()
{
    HexahedronFEMForceField::draw();

    if (!this->getContext()->getShowBehaviorModels())
        return;
    const VecCoord& x = *this->mstate->getX();
    glDisable (GL_LIGHTING);
    glPointSize(2);
    glColor4f (1,1,1,1);
    glBegin (GL_POINTS);
    for (unsigned int i=0; i<x.size(); i++)
    {
        helper::gl::glVertexT(x[i]);
    }
    glEnd();
}



template<class DataTypes>
bool HexahedronFEMForceFieldAndMass<DataTypes>::addBBox(double* minBBox, double* maxBBox)
{
    const VecCoord& x = *this->mstate->getX();
    for (unsigned int i=0; i<x.size(); i++)
    {
        //const Coord& p = x[i];
        double p[3] = {0.0, 0.0, 0.0};
        DataTypes::get(p[0],p[1],p[2],x[i]);
        for (int c=0; c<3; c++)
        {
            if (p[c] > maxBBox[c]) maxBBox[c] = p[c];
            if (p[c] < minBBox[c]) minBBox[c] = p[c];
        }
    }
    return true;
}

}
}
}


#endif
