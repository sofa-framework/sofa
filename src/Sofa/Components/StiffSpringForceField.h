#ifndef SOFA_COMPONENTS_STIFFSPRINGFORCEFIELD_H
#define SOFA_COMPONENTS_STIFFSPRINGFORCEFIELD_H

#include "Sofa/Components/SpringForceField.h"

namespace Sofa
{

namespace Components
{

template<class DataTypes>
class StiffSpringForceField : public SpringForceField<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    class Mat3 : public fixed_array<Deriv,3>
    {
    public:
        Deriv operator*(const Deriv& v)
        {
            return Deriv((*this)[0]*v,(*this)[1]*v,(*this)[2]*v);
        }
    };

protected:
    std::vector<Mat3> dfdx;

public:
    StiffSpringForceField(Core::MechanicalObject<DataTypes>* object1, Core::MechanicalObject<DataTypes>* object2, const char* filename, const std::string& name)
        : SpringForceField<DataTypes>(object1, object2, filename, name)
    {
    }

    StiffSpringForceField(Core::MechanicalObject<DataTypes>* object, const char* filename, const std::string& name)
        : SpringForceField<DataTypes>(object, filename, name)
    {
    }

    StiffSpringForceField(Core::MechanicalObject<DataTypes>* object1, Core::MechanicalObject<DataTypes>* object2)
        : SpringForceField<DataTypes>(object1, object2)
    {
    }

    StiffSpringForceField(Core::MechanicalObject<DataTypes>* object)
        : SpringForceField<DataTypes>(object)
    {
    }

    virtual void addForce();

    virtual void addDForce();
};

} // namespace Components

} // namespace Sofa

#endif
