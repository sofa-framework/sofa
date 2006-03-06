#ifndef SOFA_CORE_MECHANICALOBJECT_INL
#define SOFA_CORE_MECHANICALOBJECT_INL

#include "MechanicalObject.h"
#include "Sofa/Abstract/Encoding.inl"
#include <assert.h>

namespace Sofa
{

namespace Core
{

template <class DataTypes>
MechanicalObject<DataTypes>::MechanicalObject()
{
    x = new VecCoord;
    v = new VecDeriv;
    f = new VecDeriv;
    dx = new VecDeriv;
    // default size is 1
    x->resize(1);
    v->resize(1);
    f->resize(1);
    dx->resize(1);
}

template <class DataTypes>
MechanicalObject<DataTypes>::~MechanicalObject()
{
}

template <class DataTypes>
void MechanicalObject<DataTypes>::addMapping(Core::BasicMapping *mMap)
{
    assert(mMap);
    mappings.push_back(mMap);
}

template <class DataTypes>
void MechanicalObject<DataTypes>::addForceField(Core::ForceField *mFField)
{
    assert(mFField);
    forcefields.push_back(mFField);
}

template <class DataTypes>
void MechanicalObject<DataTypes>::init()
{
    this->propagateX();
    this->propagateV();
}

template <class DataTypes>
void MechanicalObject<DataTypes>::beginIteration(double dt)
{
    MappingIt it = mappings.begin();
    MappingIt itEnd = mappings.end();
    for (; it != itEnd; it++)
    {
        (*it)->beginIteration(dt);
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::endIteration(double dt)
{
    MappingIt it = mappings.begin();
    MappingIt itEnd = mappings.end();
    for (; it != itEnd; it++)
    {
        (*it)->endIteration(dt);
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::propagateX()
{
    MappingIt it = mappings.begin();
    MappingIt itEnd = mappings.end();
    for (; it != itEnd; it++)
    {
        (*it)->propagateX();
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::propagateV()
{
    MappingIt it = mappings.begin();
    MappingIt itEnd = mappings.end();
    for (; it != itEnd; it++)
    {
        (*it)->propagateV();
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::propagateDx()
{
    MappingIt it = mappings.begin();
    MappingIt itEnd = mappings.end();
    for (; it != itEnd; it++)
    {
        (*it)->propagateDx();
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::resetForce()
{
    f->resize(x->size());
    for (unsigned int i=0; i<f->size(); i++)
        (*f)[i] = typename VecDeriv::value_type();
    MappingIt it = mappings.begin();
    MappingIt itEnd = mappings.end();
    for (; it != itEnd; it++)
        (*it)->resetForce();
}

template <class DataTypes>
void MechanicalObject<DataTypes>::accumulateForce()
{
    {
        MappingIt it = mappings.begin();
        MappingIt itEnd = mappings.end();
        for (; it != itEnd; it++)
            (*it)->accumulateForce();
    }
    {
        ForceFieldIt it = forcefields.begin();
        ForceFieldIt itEnd = forcefields.end();
        for (; it != itEnd; it++)
            (*it)->addForce();
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::accumulateDf()
{
    {
        MappingIt it = mappings.begin();
        MappingIt itEnd = mappings.end();
        for (; it != itEnd; it++)
            (*it)->accumulateDf();
    }
    {
        ForceFieldIt it = forcefields.begin();
        ForceFieldIt itEnd = forcefields.end();
        for (; it != itEnd; it++)
            (*it)->addDForce();
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::applyConstraints()
{
}

template <class DataTypes>
void MechanicalObject<DataTypes>::setObject(Abstract::BehaviorModel* obj)
{
    MappingIt it = mappings.begin();
    MappingIt itEnd = mappings.end();
    for (; it != itEnd; it++)
        (*it)->setObject(obj);
}

} // namespace Core

} // namespace Sofa

#endif
