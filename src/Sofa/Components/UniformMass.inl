#ifndef SOFA_COMPONENTS_UNIFORMMASS_INL
#define SOFA_COMPONENTS_UNIFORMMASS_INL

#include "UniformMass.h"
#include "Scene.h"
#include "GL/template.h"
#include "Common/RigidTypes.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

template <class DataTypes, class MassType>
UniformMass<DataTypes, MassType>::UniformMass()
    : mmodel(NULL)
{
    DataTypes::set(gravity,0,-9.8,0);
}


template <class DataTypes, class MassType>
UniformMass<DataTypes, MassType>::UniformMass(Core::MechanicalModel<DataTypes>* mmodel, const std::string& /*name*/)
    : mmodel(mmodel)
{
    DataTypes::set(gravity,0,-9.8,0);
}

template <class DataTypes, class MassType>
UniformMass<DataTypes, MassType>::~UniformMass()
{
}

template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::setMechanicalModel(Core::MechanicalModel<DataTypes>* mm)
{
    this->mmodel = mm;
}

template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::setMass(const MassType& m)
{
    this->mass = m;
}

// -- Mass interface
template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::addMDx()
{
    VecDeriv& res = *mmodel->getF();
    VecDeriv& dx = *mmodel->getDx();
    for (unsigned int i=0; i<dx.size(); i++)
    {
        res[i] += dx[i] * mass;
    }
}

template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::accFromF()
{
    VecDeriv& a = *mmodel->getDx();
    VecDeriv& f = *mmodel->getF();
    for (unsigned int i=0; i<f.size(); i++)
    {
        a[i] = f[i] / mass;
    }
}

template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::computeForce()
{
    VecDeriv& f = *mmodel->getF();
    Deriv mg = gravity * mass;
    for (unsigned int i=0; i<f.size(); i++)
    {
        f[i] += mg;
    }
}

template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::setGravity( const Deriv& g )
{
    this->gravity = g;
}

template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::draw()
{
    if (!Scene::getInstance()->getShowBehaviorModels()) return;
    VecCoord& x = *mmodel->getX();
    glDisable (GL_LIGHTING);
    glPointSize(5);
    glColor4f (1,1,1,1);
    glBegin (GL_POINTS);
    for (unsigned int i=0; i<x.size(); i++)
    {
        GL::glVertexT(x[i]);
    }
    glEnd();
}

// Specialization for rigids
template <>
void UniformMass<RigidTypes, RigidMass>::draw();

} // namespace Components

} // namespace Sofa

#endif
