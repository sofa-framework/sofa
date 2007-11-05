#ifndef SOFA_COMPONENT_CONSTRAINT_UNILATERALINTERACTIONCONSTRAINT_H
#define SOFA_COMPONENT_CONSTRAINT_UNILATERALINTERACTIONCONSTRAINT_H

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
class UnilateralInteractionConstraint : public core::componentmodel::behavior::InteractionConstraint, public core::VisualModel
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

    struct Contact
    {
        int m1, m2;		///< the two extremities of the spring: masses m1 and m2
        Deriv norm;		///< contact normal, from m1 to m2
        Deriv t;		///< added for friction
        Deriv s;		///< added for friction
        Real dt;
        Real delta;		///< QP * normal - contact distance
        Real dfree;		///< QPfree * normal - contact distance
        Real dfree_t;   ///< QPfree * t
        Real dfree_s;   ///< QPfree * s
        unsigned int id;
        double mu;		///< angle for friction

        // for visu
        Coord P, Q;
        Coord Pfree, Qfree;
    };

    sofa::helper::vector<Contact> contacts;
    Real epsilon;

public:

    unsigned int constraintId;

    UnilateralInteractionConstraint(MechanicalState* object1, MechanicalState* object2)
        : object1(object1), object2(object2), epsilon(Real(0.001))
    {
    }

    UnilateralInteractionConstraint(MechanicalState* object)
        : object1(object), object2(object), epsilon(Real(0.001))
    {
    }

    UnilateralInteractionConstraint()
        : object1(NULL), object2(NULL), epsilon(Real(0.001))
    {
    }

    virtual ~UnilateralInteractionConstraint()
    {
    }

    MechanicalState* getObject1() { return object1; }
    MechanicalState* getObject2() { return object2; }
    core::componentmodel::behavior::BaseMechanicalState* getMechModel1() { return object1; }
    core::componentmodel::behavior::BaseMechanicalState* getMechModel2() { return object2; }

    void clear(int reserve = 0)
    {
        contacts.clear();
        if (reserve)
            contacts.reserve(reserve);
    }

    virtual void applyConstraint(unsigned int & /*contactId*/, double & /*mu*/);

    virtual void addContact(double mu, Deriv norm, Coord P, Coord Q, Real contactDistance, int m1, int m2, Coord Pfree = Coord(), Coord Qfree = Coord());

    virtual void getConstraintValue(double* v /*, unsigned int &numContacts */);

    // Previous Constraint Interface
    virtual void projectResponse() {};
    virtual void projectVelocity() {};
    virtual void projectPosition() {};

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

    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }


};
} // namespace constraint

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONSTRAINT_UNILATERALINTERACTIONCONSTRAINT_H
