#ifndef SOFA_COMPONENT_CONSTRAINT_LAGRANGIANMULTIPLIERCONTACTCONSTRAINT_H
#define SOFA_COMPONENT_CONSTRAINT_LAGRANGIANMULTIPLIERCONTACTCONSTRAINT_H

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
class LagrangianMultiplierContactConstraint : public LagrangianMultiplierConstraint<DataTypes>, public core::componentmodel::behavior::InteractionForceField, public core::VisualModel
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

    struct Contact
    {
        int m1, m2;   ///< the two extremities of the spring: masses m1 and m2
        Deriv norm;   ///< contact normal, from m1 to m2
        Real dist;    ///< minimum distance between the points
        Real ks;      ///< spring stiffness
        Real mu_s;    ///< coulomb friction coefficient (currently unused)
        Real mu_v;    ///< viscous friction coefficient
        Real pen0;     ///< penetration at start of timestep
        Real pen;     ///< current penetration
    };

    std::vector<Contact> contacts;

public:

    LagrangianMultiplierContactConstraint(core::componentmodel::behavior::MechanicalState<DataTypes>* object1, core::componentmodel::behavior::MechanicalState<DataTypes>* object2)
        : object1(object1), object2(object2)
    {
    }

    LagrangianMultiplierContactConstraint(core::componentmodel::behavior::MechanicalState<DataTypes>* object)
        : object1(object), object2(object)
    {
    }

    core::componentmodel::behavior::MechanicalState<DataTypes>* getObject1() { return object1; }
    core::componentmodel::behavior::MechanicalState<DataTypes>* getObject2() { return object2; }
    core::componentmodel::behavior::BaseMechanicalState* getMechModel1() { return object1; }
    core::componentmodel::behavior::BaseMechanicalState* getMechModel2() { return object2; }

    void clear(int reserve = 0)
    {
        contacts.clear();
        if (reserve)
            contacts.reserve(reserve);
        this->lambda->resize(0);
    }

    void addContact(int m1, int m2, const Deriv& norm, Real dist, Real ks, Real mu_s = 0.0f, Real mu_v = 0.0f);

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
