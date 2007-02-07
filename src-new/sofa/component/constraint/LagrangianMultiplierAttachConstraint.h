#ifndef SOFA_COMPONENT_CONSTRAINT_LAGRANGIANMULTIPLIERATTACHCONSTRAINT_H
#define SOFA_COMPONENT_CONSTRAINT_LAGRANGIANMULTIPLIERATTACHCONSTRAINT_H

#include <sofa/core/componentmodel/behavior/InteractionForceField.h>
#include <sofa/component/constraint/LagrangianMultiplierConstraint.h>
#include <sofa/core/VisualModel.h>
#include <vector>


namespace sofa
{

namespace component
{

namespace constraint
{

template<class DataTypes>
class LagrangianMultiplierAttachConstraint : public LagrangianMultiplierConstraint<DataTypes>, public core::componentmodel::behavior::InteractionForceField, public core::VisualModel
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef defaulttype::StdVectorTypes<Real, Real, Real> LMTypes;
    typedef typename LMTypes::VecCoord LMVecCoord;
    typedef typename LMTypes::VecDeriv LMVecDeriv;

protected:
    core::componentmodel::behavior::MechanicalState<DataTypes>* object1;
    core::componentmodel::behavior::MechanicalState<DataTypes>* object2;

    struct ConstraintData
    {
        int m1, m2;   ///< the two attached points
    };

    std::vector<ConstraintData> constraints;

public:

    LagrangianMultiplierAttachConstraint(core::componentmodel::behavior::MechanicalState<DataTypes>* object1, core::componentmodel::behavior::MechanicalState<DataTypes>* object2)
        : object1(object1), object2(object2)
    {
    }

    LagrangianMultiplierAttachConstraint(core::componentmodel::behavior::MechanicalState<DataTypes>* object)
        : object1(object), object2(object)
    {
    }

    core::componentmodel::behavior::MechanicalState<DataTypes>* getObject1() { return object1; }
    core::componentmodel::behavior::MechanicalState<DataTypes>* getObject2() { return object2; }
    core::componentmodel::behavior::BaseMechanicalState* getMechModel1() { return object1; }
    core::componentmodel::behavior::BaseMechanicalState* getMechModel2() { return object2; }

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

} // namespace constraint

} // namespace component

} // namespace sofa

#endif
