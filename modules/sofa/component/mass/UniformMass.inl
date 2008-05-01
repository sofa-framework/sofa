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
#ifndef SOFA_COMPONENT_MASS_UNIFORMMASS_INL
#define SOFA_COMPONENT_MASS_UNIFORMMASS_INL

#include <sofa/component/mass/UniformMass.h>
#include <sofa/core/componentmodel/behavior/Mass.inl>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/helper/gl/template.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/component/mass/AddMToMatrixFunctor.h>
#include <iostream>
#include <string.h>

using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace mass
{

using namespace sofa::defaulttype;


template <class DataTypes, class MassType>
UniformMass<DataTypes, MassType>::UniformMass()
    : mass( initData(&mass, MassType(1.0f), "mass", "Mass of each particle") )
    , totalMass( initData(&totalMass, 0.0, "totalmass", "Sum of the particles' masses") )
    , showCenterOfGravity( initData(&showCenterOfGravity, false, "showGravityCenter", "display the center of gravity of the system" ) )
    , showAxisSize( initData(&showAxisSize, 1.0f, "showAxisSizeFactor", "factor length of the axis displayed (only used for rigids)" ) )
{}

template <class DataTypes, class MassType>
UniformMass<DataTypes, MassType>::~UniformMass()
{}

template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::setMass(const MassType& m)
{
    this->mass.setValue(m);
}

template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::setTotalMass(double m)
{
    this->totalMass.setValue(m);
}

template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::init()
{
    this->core::componentmodel::behavior::Mass<DataTypes>::init();
    if (this->totalMass.getValue()>0 && this->mstate!=NULL)
    {
        MassType* m = this->mass.beginEdit();
        *m = ((typename DataTypes::Real)this->totalMass.getValue() / this->mstate->getX()->size());
        this->mass.endEdit();
    }
}

// -- Mass interface
template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::addMDx(VecDeriv& res, const VecDeriv& dx, Real_Sofa factor)
{
    MassType m = mass.getValue();
    if (factor != 1.0)
        m *= (typename DataTypes::Real)factor;
    for (unsigned int i=0; i<dx.size(); i++)
    {
        res[i] += dx[i] * m;
    }
}

template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::accFromF(VecDeriv& a, const VecDeriv& f)
{
    const MassType& m = mass.getValue();
    for (unsigned int i=0; i<f.size(); i++)
    {
        a[i] = f[i] / m;
    }
}


template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::addMDxToVector(defaulttype::BaseVector * /*resVect*/, const VecDeriv* /*dx*/, Real_Sofa /*mFact*/, unsigned int& /*offset*/)
{

}

template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::addGravityToV(double dt)
{
    if (this->mstate)
    {
        VecDeriv& v = *this->mstate->getV();
        const Real_Sofa* g = this->getContext()->getLocalGravity().ptr();
        Deriv theGravity;
        DataTypes::set( theGravity, g[0], g[1], g[2]);
        Deriv hg = theGravity * (Real)dt;
        if (this->f_printLog.getValue())
            std::cerr << "UniformMass::addGravityToV hg = "<<theGravity<<"*"<<dt<<"="<<hg<<std::endl;
        for (unsigned int i=0; i<v.size(); i++)
        {
            v[i] += hg;
        }
    }
}

template <class DataTypes, class MassType>
#ifdef SOFA_SUPPORT_MOVING_FRAMES
void UniformMass<DataTypes, MassType>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
#else
void UniformMass<DataTypes, MassType>::addForce(VecDeriv& f, const VecCoord& /*x*/, const VecDeriv& /*v*/)
#endif
{
    //if gravity was added separately (in solver's "solve" method), then nothing to do here
    if(this->m_separateGravity.getValue())
        return;

    // weight
    const Real_Sofa* g = this->getContext()->getLocalGravity().ptr();
    Deriv theGravity;
    DataTypes::set
    ( theGravity, g[0], g[1], g[2]);
    const MassType& m = mass.getValue();
    Deriv mg = theGravity * m;
    if (this->f_printLog.getValue())
        cerr<<"UniformMass::addForce, mg = "<<mass<<" * "<<theGravity<<" = "<<mg<<endl;

#ifdef SOFA_SUPPORT_MOVING_FRAMES
    // velocity-based stuff
    core::objectmodel::BaseContext::SpatialVector vframe = getContext()->getVelocityInWorld();
    core::objectmodel::BaseContext::Vec3 aframe = getContext()->getVelocityBasedLinearAccelerationInWorld() ;
//     cerr<<"UniformMass<DataTypes, MassType>::computeForce(), vFrame in world coordinates = "<<vframe<<endl;
    //cerr<<"UniformMass<DataTypes, MassType>::computeForce(), aFrame in world coordinates = "<<aframe<<endl;
//     cerr<<"UniformMass<DataTypes, MassType>::computeForce(), getContext()->getLocalToWorld() = "<<getContext()->getPositionInWorld()<<endl;

    // project back to local frame
    vframe = getContext()->getPositionInWorld() / vframe;
    aframe = getContext()->getPositionInWorld().backProjectVector( aframe );
//     cerr<<"UniformMass<DataTypes, MassType>::computeForce(), vFrame in local coordinates= "<<vframe<<endl;
//     cerr<<"UniformMass<DataTypes, MassType>::computeForce(), aFrame in local coordinates= "<<aframe<<endl;
//     cerr<<"UniformMass<DataTypes, MassType>::computeForce(), mg in local coordinates= "<<mg<<endl;
#endif

    // add weight and inertia force
    for (unsigned int i=0; i<f.size(); i++)
    {
#ifdef SOFA_SUPPORT_MOVING_FRAMES
        f[i] += mg + core::componentmodel::behavior::inertiaForce(vframe,aframe,m,x[i],v[i]);
#else
        f[i] += mg;
#endif
        //cerr<<"UniformMass<DataTypes, MassType>::computeForce(), vframe = "<<vframe<<", aframe = "<<aframe<<", x = "<<x[i]<<", v = "<<v[i]<<endl;
        //cerr<<"UniformMass<DataTypes, MassType>::computeForce() = "<<mg + Core::inertiaForce(vframe,aframe,mass,x[i],v[i])<<endl;
    }

}

template <class DataTypes, class MassType>
sofa::defaulttype::Vector3::value_type UniformMass<DataTypes, MassType>::getKineticEnergy( const VecDeriv& v )
{
    double e=0;
    const MassType& m = mass.getValue();
    for (unsigned int i=0; i<v.size(); i++)
    {
        e+= v[i]*m*v[i];
    }
    //cerr<<"UniformMass<DataTypes, MassType>::getKineticEnergy = "<<e/2<<endl;
    return e/2;
}

template <class DataTypes, class MassType>
sofa::defaulttype::Vector3::value_type UniformMass<DataTypes, MassType>::getPotentialEnergy( const VecCoord& x )
{
    double e = 0;
    const MassType& m = mass.getValue();
    // gravity
    Vec3d g ( this->getContext()->getLocalGravity() );
    Deriv theGravity;
    DataTypes::set
    ( theGravity, g[0], g[1], g[2]);
    Deriv mg = theGravity * m;
    //cerr<<"UniformMass<DataTypes, MassType>::getPotentialEnergy, theGravity = "<<theGravity<<endl;
    for (unsigned int i=0; i<x.size(); i++)
    {
        /*        cerr<<"UniformMass<DataTypes, MassType>::getPotentialEnergy, mass = "<<mass<<endl;
                cerr<<"UniformMass<DataTypes, MassType>::getPotentialEnergy, x = "<<x[i]<<endl;
                cerr<<"UniformMass<DataTypes, MassType>::getPotentialEnergy, remove "<<theGravity*mass*x[i]<<endl;*/
        e -= mg*x[i];
    }
    return e;
}

/// Add Mass contribution to global Matrix assembling
template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::addMToMatrix(defaulttype::BaseMatrix * mat, Real_Sofa mFact, unsigned int &offset)
{
    const MassType& m = mass.getValue();
    const int N = defaulttype::DataTypeInfo<Deriv>::size();
    const unsigned int size = this->mstate->getSize();
    AddMToMatrixFunctor<Deriv,MassType> calc;
    for (unsigned int i=0; i<size; i++)
        calc(mat, m, offset + N*i, mFact);
}


template <class DataTypes, class MassType>
sofa::defaulttype::Vector3::value_type UniformMass<DataTypes, MassType>::getElementMass(unsigned int )
{
    return (sofa::defaulttype::Vector3::value_type)(mass.getValue());
}


template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::draw()
{
    if (!getContext()->getShowBehaviorModels())
        return;
    const VecCoord& x = *this->mstate->getX();
    //cerr<<"UniformMass<DataTypes, MassType>::draw() "<<x<<endl;
    Coord gravityCenter;
    glDisable (GL_LIGHTING);
    glPointSize(2);
    glColor4f (1,1,1,1);
    glBegin (GL_POINTS);
    for (unsigned int i=0; i<x.size(); i++)
    {
        helper::gl::glVertexT(x[i]);
        gravityCenter += x[i];
    }
    glEnd();

    if(showCenterOfGravity.getValue())
    {
        glBegin (GL_LINES);
        glColor4f (1,1,0,1);
        gravityCenter /= x.size();
        for(unsigned int i=0 ; i<Coord::static_size ; i++)
        {
            Coord v;
            v[i] = showAxisSize.getValue();
            helper::gl::glVertexT(gravityCenter-v);
            helper::gl::glVertexT(gravityCenter+v);
        }
        glEnd();
    }
}

template <class DataTypes, class MassType>
bool UniformMass<DataTypes, MassType>::addBBox(Real_Sofa* minBBox, Real_Sofa* maxBBox)
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

template<class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::parse(core::objectmodel::BaseObjectDescription* arg)
{
    Inherited::parse(arg);
    /*
    if (arg->getAttribute("mass"))
      {
        this->setMass((MassType)atof(arg->getAttribute("mass")));
      }
      if (arg->getAttribute("totalmass"))
      {
        this->setTotalMass(atof(arg->getAttribute("totalmass")));
      }
    */
}


//Specialization for rigids
#ifndef SOFA_FLOAT

template<>
void UniformMass<Rigid3dTypes, Rigid3dMass>::parse(core::objectmodel::BaseObjectDescription* arg);
template <>
void UniformMass<Rigid3dTypes, Rigid3dMass>::draw();
template <>
void UniformMass<Rigid2dTypes, Rigid2dMass>::draw();
template <>
sofa::defaulttype::Vector3::value_type UniformMass<Rigid3dTypes,Rigid3dMass>::getPotentialEnergy( const VecCoord& x );
template <>
sofa::defaulttype::Vector3::value_type UniformMass<Rigid2dTypes,Rigid2dMass>::getPotentialEnergy( const VecCoord& x );
template <>
void UniformMass<Vec6dTypes,double>::draw();
#endif
#ifndef SOFA_DOUBLE
template<>
void UniformMass<Rigid3fTypes, Rigid3fMass>::parse(core::objectmodel::BaseObjectDescription* arg);
template <>
void UniformMass<Rigid3fTypes, Rigid3fMass>::draw();
template <>
void UniformMass<Rigid2fTypes, Rigid2fMass>::draw();
template <>
sofa::defaulttype::Vector3::value_type UniformMass<Rigid3fTypes,Rigid3fMass>::getPotentialEnergy( const VecCoord& x );
template <>
sofa::defaulttype::Vector3::value_type UniformMass<Rigid2fTypes,Rigid2fMass>::getPotentialEnergy( const VecCoord& x );
template <>
void UniformMass<Vec6fTypes,float>::draw();
#endif



} // namespace mass

} // namespace component

} // namespace sofa

#endif



