#ifndef SOFA_COMPONENTS_LagrangianMultiplierContactConstraint_H
#define SOFA_COMPONENTS_LagrangianMultiplierContactConstraint_H

#include "Sofa-old/Core/InteractionForceField.h"
#include "LagrangianMultiplierConstraint.h"
#include "Sofa-old/Abstract/VisualModel.h"

#include <vector>

namespace Sofa
{

namespace Components
{

template<class DataTypes>
class LagrangianMultiplierAttachConstraint : public LagrangianMultiplierConstraint<DataTypes>, public Core::InteractionForceField, public Abstract::VisualModel
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef Common::StdVectorTypes<Real, Real, Real> LMTypes;
    typedef typename LMTypes::VecCoord LMVecCoord;
    typedef typename LMTypes::VecDeriv LMVecDeriv;

protected:
    Core::MechanicalModel<DataTypes>* object1;
    Core::MechanicalModel<DataTypes>* object2;

    struct ConstraintData
    {
        int m1, m2;   ///< the two attached points
    };

    std::vector<ConstraintData> constraints;

public:

    LagrangianMultiplierAttachConstraint(Core::MechanicalModel<DataTypes>* object1, Core::MechanicalModel<DataTypes>* object2)
        : object1(object1), object2(object2)
    {
    }

    LagrangianMultiplierAttachConstraint(Core::MechanicalModel<DataTypes>* object)
        : object1(object), object2(object)
    {
    }

    Core::MechanicalModel<DataTypes>* getObject1() { return object1; }
    Core::MechanicalModel<DataTypes>* getObject2() { return object2; }
    Core::BasicMechanicalModel* getMechModel1() { return object1; }
    Core::BasicMechanicalModel* getMechModel2() { return object2; }

    void clear(int reserve = 0)
    {
        constraints.clear();
        if (reserve)
            constraints.reserve(reserve);
        this->lambda->resize(0);
    }

    void addConstraint(int m1, int m2);

    virtual void addForce();

    virtual void addDForce();

    virtual double getPotentialEnergy();

    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }
};

} // namespace Components

} // namespace Sofa

#endif
