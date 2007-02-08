#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_REPULSIVESPRINGFORCEFIELD_H
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_REPULSIVESPRINGFORCEFIELD_H

#include <sofa/component/forcefield/StiffSpringForceField.h>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

template<class DataTypes>
class RepulsiveSpringForceField : public forcefield::StiffSpringForceField<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef typename forcefield::StiffSpringForceField<DataTypes>::Mat3 Mat3;
public:

    RepulsiveSpringForceField(core::componentmodel::behavior::MechanicalState<DataTypes>* object1, core::componentmodel::behavior::MechanicalState<DataTypes>* object2)
        : forcefield::StiffSpringForceField<DataTypes>(object1, object2)
    {
    }

    RepulsiveSpringForceField(core::componentmodel::behavior::MechanicalState<DataTypes>* object)
        : forcefield::StiffSpringForceField<DataTypes>(object)
    {
    }

    virtual void addForce();
    virtual double getPotentialEnergy();
};

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif
