#ifndef SOFA_COMPONENTS_UNIFORMMASS_INL
#define SOFA_COMPONENTS_UNIFORMMASS_INL

#include "UniformMass.h"
#include "Sofa/Core/Mass.inl"
#include <Sofa/Core/Context.h>
#include "GL/template.h"
#include "Common/RigidTypes.h"
//#include "Common/SolidTypes.h"
#include <iostream>
using std::cerr;
using std::endl;

namespace Sofa
{


namespace Components
{

using namespace Common;



//Specialization for rigids
template <>
void UniformMass<RigidTypes, RigidMass>::draw();



template <class DataTypes, class MassType>
UniformMass<DataTypes, MassType>::UniformMass()
{
}


template <class DataTypes, class MassType>
UniformMass<DataTypes, MassType>::UniformMass(Core::MechanicalModel<DataTypes>* mmodel)
    : Core::Mass<DataTypes>(mmodel)
{
}

template <class DataTypes, class MassType>
UniformMass<DataTypes, MassType>::~UniformMass()
{}

template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::setMass(const MassType& m)
{
    this->mass = m;
}

// -- Mass interface
template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::addMDx(VecDeriv& res, const VecDeriv& dx)
{
    for (unsigned int i=0; i<dx.size(); i++)
    {
        res[i] += dx[i] * mass;
    }
}

template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::accFromF(VecDeriv& a, const VecDeriv& f)
{
    for (unsigned int i=0; i<f.size(); i++)
    {
        a[i] = f[i] / mass;
    }
}

template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::addForce(VecDeriv& f, const VecCoord& /*x*/, const VecDeriv& /*v*/)
{
    // weight
    const double* g = this->getContext()->getGravity();
    Deriv theGravity;
    DataTypes::set
    ( theGravity, g[0], g[1], g[2]);
    Deriv mg = theGravity * mass;
#if 0
    // velcity-based stuff
    Core::Context::SpatialVelocity vframe = getContext()->getSpatialVelocity();
    Core::Context::Vec aframe = getContext()->getLinearAcceleration() ;
    /*	cerr<<"UniformMass<DataTypes, MassType>::computeForce(), vFrame in local coordinates = "<<vframe<<endl;
    	cerr<<"UniformMass<DataTypes, MassType>::computeForce(), aFrame in local coordinates = "<<aframe<<endl;
    	cerr<<"UniformMass<DataTypes, MassType>::computeForce(), getContext()->getLocalToWorld() = "<<getContext()->getLocalToWorld()<<endl;*/

    // project back to local frame
    vframe = getContext()->getLocalToWorld() / vframe;
    aframe = getContext()->getLocalToWorld().backProjectVector( aframe );
//         	cerr<<"UniformMass<DataTypes, MassType>::computeForce(), vFrame = "<<vframe<<endl;
//         	cerr<<"UniformMass<DataTypes, MassType>::computeForce(), aFrame = "<<aframe<<endl;
//         	cerr<<"UniformMass<DataTypes, MassType>::computeForce(), mg = "<<mg<<endl;

    // add weight and inertia force
    for (unsigned int i=0; i<f.size(); i++)
    {
        f[i] += mg + Core::inertiaForce(vframe,aframe,mass,x[i],v[i]);
        //cerr<<"UniformMass<DataTypes, MassType>::computeForce() = "<<mg + inertiaForce(vframe,aframe,mass,x[i],v[i])<<endl;
    }
#else
    for (unsigned int i=0; i<f.size(); i++)
    {
        f[i] += mg;
    }
#endif
}

template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::draw()
{
    if (!getContext()->getShowBehaviorModels()) return;
    VecCoord& x = *this->mmodel->getX();
    glDisable (GL_LIGHTING);
    glPointSize(2);
    glColor4f (1,1,1,1);
    glBegin (GL_POINTS);
    for (unsigned int i=0; i<x.size(); i++)
    {
        GL::glVertexT(x[i]);
    }
    glEnd();
}


} // namespace Components

} // namespace Sofa

#endif
