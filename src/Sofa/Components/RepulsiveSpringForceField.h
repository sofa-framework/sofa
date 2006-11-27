#ifndef SOFA_COMPONENTS_REPULSIVESPRINGFORCEFIELD_H
#define SOFA_COMPONENTS_REPULSIVESPRINGFORCEFIELD_H

#include "Sofa/Components/StiffSpringForceField.h"

namespace Sofa
{

namespace Components
{

template<class DataTypes>
class RepulsiveSpringForceField : public StiffSpringForceField<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef typename StiffSpringForceField<DataTypes>::Mat3 Mat3;
public:

    RepulsiveSpringForceField(Core::MechanicalModel<DataTypes>* object1, Core::MechanicalModel<DataTypes>* object2)
        : StiffSpringForceField<DataTypes>(object1, object2)
    {
    }

    RepulsiveSpringForceField(Core::MechanicalModel<DataTypes>* object)
        : StiffSpringForceField<DataTypes>(object)
    {
    }

    virtual void addForce();
    virtual double getPotentialEnergy();
};

} // namespace Components

} // namespace Sofa

#endif
