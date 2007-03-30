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
    typedef typename core::componentmodel::behavior::MechanicalState<DataTypes> MechanicalState;

protected:
    MechanicalState* object1;
    MechanicalState* object2;

    struct ConstraintData
    {
        int m1, m2;   ///< the two attached points
    };

    std::vector<ConstraintData> constraints;

public:

    LagrangianMultiplierAttachConstraint(MechanicalState* m1=NULL, MechanicalState* m2=NULL)
        : object1(m1), object2(m2)
    {
    }

    MechanicalState* getObject1() { return object1; }
    MechanicalState* getObject2() { return object2; }
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

    /// Pre-construction check method called by ObjectFactory.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (arg->getAttribute("object1") || arg->getAttribute("object2"))
        {
            if (dynamic_cast<MechanicalState*>(arg->findObject(arg->getAttribute("object1",".."))) == NULL)
                return false;
            if (dynamic_cast<MechanicalState*>(arg->findObject(arg->getAttribute("object2",".."))) == NULL)
                return false;
        }
        else
        {
            if (dynamic_cast<MechanicalState*>(context->getMechanicalState()) == NULL)
                return false;
        }
        return core::componentmodel::behavior::InteractionForceField::canCreate(obj, context, arg);
    }

    /// Construction method called by ObjectFactory.
    template<class T>
    static void create(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        core::componentmodel::behavior::InteractionForceField::create(obj, context, arg);
        if (arg && (arg->getAttribute("object1") || arg->getAttribute("object2")))
        {
            obj->object1 = dynamic_cast<MechanicalState*>(arg->findObject(arg->getAttribute("object1","..")));
            obj->object2 = dynamic_cast<MechanicalState*>(arg->findObject(arg->getAttribute("object2","..")));
        }
        else if (context)
        {
            obj->object1 =
                obj->object2 =
                        dynamic_cast<MechanicalState*>(context->getMechanicalState());
        }
    }

};

} // namespace constraint

} // namespace component

} // namespace sofa

#endif
