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
    : totalMass(0)
{}


template <class DataTypes, class MassType>
UniformMass<DataTypes, MassType>::UniformMass(Core::MechanicalModel<DataTypes>* mmodel)
    : Core::Mass<DataTypes>(mmodel), totalMass(0)
{}

template <class DataTypes, class MassType>
UniformMass<DataTypes, MassType>::~UniformMass()
{}

template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::setMass(const MassType& m)
{
    this->mass = m;
}

template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::setTotalMass(double m)
{
    this->totalMass = m;
}

template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::init()
{
    this->Core::Mass<DataTypes>::init();
    if (this->totalMass>0 && this->mmodel!=NULL)
    {
        this->mass = (MassType)(this->totalMass / this->mmodel->getX()->size());
    }
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
void UniformMass<DataTypes, MassType>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    // weight
    const double* g = this->getContext()->getLocalGravity();
    Deriv theGravity;
    DataTypes::set
    ( theGravity, g[0], g[1], g[2]);
    Deriv mg = theGravity * mass;
    //cerr<<"UniformMass<DataTypes, MassType>::addForce, mg = "<<mass<<" * "<<theGravity<<" = "<<mg<<endl;

    // velocity-based stuff
    Core::Context::SpatialVector vframe = getContext()->getVelocityInWorld();
    Core::Context::Vec3 aframe = getContext()->getVelocityBasedLinearAccelerationInWorld() ;
//     cerr<<"UniformMass<DataTypes, MassType>::computeForce(), vFrame in world coordinates = "<<vframe<<endl;
    //cerr<<"UniformMass<DataTypes, MassType>::computeForce(), aFrame in world coordinates = "<<aframe<<endl;
//     cerr<<"UniformMass<DataTypes, MassType>::computeForce(), getContext()->getLocalToWorld() = "<<getContext()->getPositionInWorld()<<endl;

    // project back to local frame
    vframe = getContext()->getPositionInWorld() / vframe;
    aframe = getContext()->getPositionInWorld().backProjectVector( aframe );
//     cerr<<"UniformMass<DataTypes, MassType>::computeForce(), vFrame in local coordinates= "<<vframe<<endl;
//     cerr<<"UniformMass<DataTypes, MassType>::computeForce(), aFrame in local coordinates= "<<aframe<<endl;
//     cerr<<"UniformMass<DataTypes, MassType>::computeForce(), mg in local coordinates= "<<mg<<endl;

    // add weight and inertia force
    for (unsigned int i=0; i<f.size(); i++)
    {
        //f[i] += mg;
        f[i] += mg + Core::inertiaForce(vframe,aframe,mass,x[i],v[i]);
        //cerr<<"UniformMass<DataTypes, MassType>::computeForce(), vframe = "<<vframe<<", aframe = "<<aframe<<", x = "<<x[i]<<", v = "<<v[i]<<endl;
        //cerr<<"UniformMass<DataTypes, MassType>::computeForce() = "<<mg + Core::inertiaForce(vframe,aframe,mass,x[i],v[i])<<endl;
    }

}

template <class DataTypes, class MassType>
double UniformMass<DataTypes, MassType>::getKineticEnergy( const VecDeriv& v )
{
    double e=0;
    for (unsigned int i=0; i<v.size(); i++)
    {
        e+= v[i]*mass*v[i];
    }
    //cerr<<"UniformMass<DataTypes, MassType>::getKineticEnergy = "<<e/2<<endl;
    return e/2;
}

template <class DataTypes, class MassType>
double UniformMass<DataTypes, MassType>::getPotentialEnergy( const VecCoord& x )
{
    double e = 0;
    // gravity
    Vec3d g ( this->getContext()->getLocalGravity() );
    Deriv theGravity;
    DataTypes::set
    ( theGravity, g[0], g[1], g[2]);
    //cerr<<"UniformMass<DataTypes, MassType>::getPotentialEnergy, theGravity = "<<theGravity<<endl;
    for (unsigned int i=0; i<x.size(); i++)
    {
        /*        cerr<<"UniformMass<DataTypes, MassType>::getPotentialEnergy, mass = "<<mass<<endl;
                cerr<<"UniformMass<DataTypes, MassType>::getPotentialEnergy, x = "<<x[i]<<endl;
                cerr<<"UniformMass<DataTypes, MassType>::getPotentialEnergy, remove "<<theGravity*mass*x[i]<<endl;*/
        e -= theGravity*mass*x[i];
    }
    return e;
}


template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::draw()
{
    if (!getContext()->getShowBehaviorModels())
        return;
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



