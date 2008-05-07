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
#include <sofa/component/topology/SparseGridTopology.h>

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
    : MassT()
    , HexahedronFEMForceFieldT()
    , _density(initData(&_density,(Real)1.0,"density","density == volumetric mass in english (kg.m-3)"))
{}


template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::init( )
{
    if(this->_alreadyInit)return;

// 		  cerr<<"HexahedronFEMForceFieldAndMass<DataTypes>::init( ) "<<this->getName()<<endl;
    HexahedronFEMForceFieldT::init();
    MassT::init();

//         computeElementMasses();

// 		_particleMasses.clear();
    _particleMasses.resize( this->_initialPoints.getValue().size() );

    int i=0;
    for(typename VecElement::const_iterator it = this->_indexedElements->begin() ; it != this->_indexedElements->end() ; ++it, ++i)
    {
        Vec<8,Coord> nodes;
        for(int w=0; w<8; ++w)
            nodes[w] = this->_initialPoints.getValue()[(*it)[this->_indices[w]]];

        // volume of a element
        Real volume = (nodes[1]-nodes[0]).norm()*(nodes[3]-nodes[0]).norm()*(nodes[4]-nodes[0]).norm();

        if( this->_sparseGrid ) // if sparseGrid -> the filling ratio is taken into account
            volume *= (Real) (this->_sparseGrid->getType(i)==topology::SparseGridTopology::BOUNDARY?.5:1.0);

        // mass of a particle...
        Real mass = Real (( volume * _density.getValue() ) / 8.0 );

        // ... is added to each particle of the element
        for(int w=0; w<8; ++w)
            _particleMasses[ (*it)[w] ] += mass;
    }



// 		Real totalmass = 0.0;
// 		for( unsigned i=0;i<_particleMasses.size();++i)
// 		{
// 			totalmass+=_particleMasses[i];
// 		}
// 		cerr<<"TOTAL MASS = "<<totalmass<<endl;
}



template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::reinit( )
{
// 		  cerr<<"HexahedronFEMForceFieldAndMass<DataTypes>::reinit( )"<<endl;
    HexahedronFEMForceFieldT::reinit();
//         Mass::reinit();

    computeElementMasses();
}



template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::computeElementMasses(  )
{
// 		  _elementMasses.resize( this->_elementStiffnesses.getValue().size() );

    int i=0;
    typename VecElement::const_iterator it;
    for(it = this->_indexedElements->begin() ; it != this->_indexedElements->end() ; ++it, ++i)
    {
        Vec<8,Coord> nodes;
        for(int w=0; w<8; ++w)
            nodes[w] = this->_initialPoints.getValue()[(*it)[this->_indices[w]]];

        if( _elementMasses.getValue().size() <= (unsigned)i )
        {
            _elementMasses.beginEdit()->resize( _elementMasses.getValue().size()+1 );
// 			  computeElementMass( (*_elementMasses.beginEdit())[i], nodes,i );
            computeElementMass( (*_elementMasses.beginEdit())[i], this->_rotatedInitialElements[i],i,
                    (this->_sparseGrid && this->_sparseGrid->getType(i)==topology::SparseGridTopology::BOUNDARY)?.5:1.0 );
        }


    }
}

template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::computeElementMass( ElementMass &Mass, const helper::fixed_array<Coord,8> &nodes, const int elementIndice, double stiffnessFactor)
{
    Real vol = (nodes[1]-nodes[0]).norm()*(nodes[3]-nodes[0]).norm()*(nodes[4]-nodes[0]).norm();

    Coord l = nodes[6] - nodes[0];

    Mass.clear();

    for(int i=0; i<8; ++i)
    {
        Real mass = vol * integrateMass(this->_coef[i][0], this->_coef[i][1],this->_coef[i][2], 2.0f/l[0], 2.0f/l[1], 2.0f/l[2]);

        Mass[i*3][i*3] += mass;
        Mass[i*3+1][i*3+1] += mass;
        Mass[i*3+2][i*3+2] += mass;



        for(int j=i+1; j<8; ++j)
        {
            Real mass = vol * integrateMass(this->_coef[i][0], this->_coef[i][1],this->_coef[i][2], 2.0f/l[0], 2.0f/l[1], 2.0f/l[2]);

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

    Mass *= stiffnessFactor;
}


template<class DataTypes>
typename HexahedronFEMForceFieldAndMass<DataTypes>::Real HexahedronFEMForceFieldAndMass<DataTypes>::integrateMass(  int signx, int signy, int signz,Real /*l0*/,Real /*l1*/,Real /*l2*/  )
{
    Real t1 = (Real)(signx*signx);
    Real t2 = (Real)(signy*signy);
    Real t3 = (Real)(signz*signz);
    Real t9 = (Real)(t1*t2);
    return (Real)(t1*t3/72.0+t2*t3/72.0+t9*t3/216.0+t3/24.0+1.0/8.0+t9/72.0+t1/24.0+t2/24.0*_density.getValue());


// 		  Real t1 = l0*l0;
// 		  Real t2 = t1*signx;
// 		  Real t3 = signz*signx;
// 		  Real t7 = t1*signy;
// 		  return t2*t3*signz/72.0+t7*signz*signy*signz/72.0+t2*signy*t3*signy*
// 				  signz/216.0+t1*signz*signz/24.0+t2*signy*signx*signy/72.0+t1/8.0+t2*signx/
// 				  24.0+t7*signy/24.0 *_density.getValue() /(l0*l1*l2);

}



template<class DataTypes>
std::string HexahedronFEMForceFieldAndMass<DataTypes>::getTemplateName() const
{
    return HexahedronFEMForceFieldT::getTemplateName();
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

        actualF = _elementMasses.getValue()[i] * actualDx;


        for(int w=0; w<8; ++w)
            f[(*it)[this->_indices[w]]] += Deriv( actualF[w*3],  actualF[w*3+1],   actualF[w*3+2]  ) * factor;

    }

// 		  for(unsigned i=0;i<_particleMasses.size();++i)
// 		  		f[i] += _particleMasses[i] * dx[i] *factor;
}



template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::accFromF(VecDeriv& /*a*/, const VecDeriv& /*f*/)
{
    cerr<<"HexahedronFEMForceFieldAndMass<DataTypes>::accFromF not yet implemented\n";
    // need to built the big global mass matrix and to inverse it...
}


template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::addGravityToV(double dt)
{
    if(this->mstate)
    {
        VecDeriv& v = *this->mstate->getV();
        for (unsigned int i=0; i<_particleMasses.size(); i++)
        {
            v[i] +=this->getContext()->getLocalGravity()*dt;
        }
    }
}


template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    HexahedronFEMForceFieldT::addForce(f,x,v);

    //if gravity was added separately (in solver's "solve" method), then nothing to do here
    if (this->m_separateGravity.getValue())
        return;

    // gravity
// 		Vec3d g ( this->getContext()->getLocalGravity() );
// 		Deriv theGravity;
// 		DataTypes::set ( theGravity, g[0], g[1], g[2]);

    // velocity-based stuff
    core::objectmodel::BaseContext::SpatialVector vframe = this->getContext()->getVelocityInWorld();
    core::objectmodel::BaseContext::Vec3 aframe = this->getContext()->getVelocityBasedLinearAccelerationInWorld() ;

    // project back to local frame
    vframe = this->getContext()->getPositionInWorld() / vframe;
    aframe = this->getContext()->getPositionInWorld().backProjectVector( aframe );

    // add weight and inertia force
    for (unsigned int i=0; i<_particleMasses.size(); i++)
    {
        f[i] += this->getContext()->getLocalGravity()*_particleMasses[i] + core::componentmodel::behavior::inertiaForce(vframe,aframe,_particleMasses[i],x[i],v[i]);
    }
}



template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::addDForce(VecDeriv& df, const VecDeriv& dx)
{
    HexahedronFEMForceFieldT::addDForce(df,dx);
}


template<class DataTypes>
double HexahedronFEMForceFieldAndMass<DataTypes>::getElementMass(unsigned int /*index*/)
{
    std::cerr<<"HexahedronFEMForceFieldAndMass<DataTypes>::getElementMass not yet implemented\n"; return 0.0;
}


template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::draw()
{
// 		  cerr<<"HexahedronFEMForceFieldAndMass<DataTypes>::draw()  "<<this->_indexedElements->size()<<"\n";
    HexahedronFEMForceFieldT::draw();

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
        Real p[3] = {0.0, 0.0, 0.0};
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
