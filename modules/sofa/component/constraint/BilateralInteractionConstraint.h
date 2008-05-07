#ifndef SOFA_COMPONENT_CONSTRAINT_BILATERALINTERACTIONCONSTRAINT_H
#define SOFA_COMPONENT_CONSTRAINT_BILATERALINTERACTIONCONSTRAINT_H

#include <sofa/core/componentmodel/behavior/InteractionConstraint.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/VisualModel.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace constraint
{

template<class DataTypes>
class BilateralInteractionConstraint : public core::componentmodel::behavior::InteractionConstraint
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecConst VecConst;
    typedef typename DataTypes::SparseVecDeriv SparseVecDeriv;
    typedef typename DataTypes::SparseDeriv SparseDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef typename core::componentmodel::behavior::MechanicalState<DataTypes> MechanicalState;

protected:
    MechanicalState* object1;
    MechanicalState* object2;
    bool yetIntegrated;

    Coord dfree;
    unsigned int cid;

    Data<int> m1;
    Data<int> m2;

public:

    BilateralInteractionConstraint(MechanicalState* object1, MechanicalState* object2)
        : object1(object1), object2(object2), yetIntegrated(false)
        , m1(initData(&m1, 0, "first_point","index of the constraint on the first model"))
        , m2(initData(&m2, 0, "second_point","index of the constraint on the second model"))
    {
    }

    BilateralInteractionConstraint(MechanicalState* object)
        : object1(object), object2(object), yetIntegrated(false)
        , m1(initData(&m1, 0, "first_point","index of the constraint on the first model"))
        , m2(initData(&m2, 0, "second_point","index of the constraint on the second model"))
    {
    }

    BilateralInteractionConstraint()
        : object1(NULL), object2(NULL), yetIntegrated(false)
        , m1(initData(&m1, 0, "first_point","index of the constraint on the first model"))
        , m2(initData(&m2, 0, "second_point","index of the constraint on the second model"))
    {
    }

    virtual ~BilateralInteractionConstraint()
    {
    }

    MechanicalState* getObject1() { return object1; }
    MechanicalState* getObject2() { return object2; }
    core::componentmodel::behavior::BaseMechanicalState* getMechModel1() { return object1; }
    core::componentmodel::behavior::BaseMechanicalState* getMechModel2() { return object2; }

    virtual void init();

    virtual void applyConstraint(unsigned int & /*constraintId*/, double & /*unused*/);

    virtual void getConstraintValue(double* v /*, unsigned int &numContacts */);

    virtual void getConstraintId(long* id, unsigned int &offset);

    virtual void getConstraintType(bool* type, unsigned int &offset);

    // Previous Constraint Interface
    virtual void projectResponse() {}
    virtual void projectVelocity() {}
    virtual void projectPosition() {}
    virtual void projectFreeVelocity() {}
    virtual void projectFreePosition() {}

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
        return core::componentmodel::behavior::InteractionConstraint::canCreate(obj, context, arg);
    }

    /// Construction method called by ObjectFactory.
    template<class T>
    static void create(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        core::componentmodel::behavior::InteractionConstraint::create(obj, context, arg);
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

    void draw();
};
} // namespace constraint

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONSTRAINT_BILATERALINTERACTIONCONSTRAINT_H
