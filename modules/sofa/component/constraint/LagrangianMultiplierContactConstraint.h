#ifndef SOFA_COMPONENT_CONSTRAINT_LAGRANGIANMULTIPLIERCONTACTCONSTRAINT_H
#define SOFA_COMPONENT_CONSTRAINT_LAGRANGIANMULTIPLIERCONTACTCONSTRAINT_H

#include <sofa/core/componentmodel/behavior/PairInteractionForceField.h>
#include <sofa/component/constraint/LagrangianMultiplierConstraint.h>
#include <vector>



namespace sofa
{

namespace component
{

namespace constraint
{

template<class DataTypes>
class LagrangianMultiplierContactConstraint : public LagrangianMultiplierConstraint<DataTypes>, public core::componentmodel::behavior::PairInteractionForceField<DataTypes>, public virtual core::objectmodel::BaseObject
{
public:
    typedef typename core::componentmodel::behavior::PairInteractionForceField<DataTypes> Inherit;
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

    sofa::helper::vector<Contact> contacts;

public:

    LagrangianMultiplierContactConstraint(MechanicalState* m1=NULL, MechanicalState* m2=NULL)
        : Inherit(m1, m2)
    {
    }

    MechanicalState* getObject1() { return this->mstate1; }
    MechanicalState* getObject2() { return this->mstate2; }

    void clear(int reserve = 0)
    {
        contacts.clear();
        if (reserve)
            contacts.reserve(reserve);
        this->lambda->resize(0);
    }

    void addContact(int m1, int m2, const Deriv& norm, Real dist, Real ks, Real mu_s = 0.0f, Real mu_v = 0.0f);

    virtual void addForce(VecDeriv& f1, VecDeriv& f2, const VecCoord& x1, const VecCoord& x2, const VecDeriv& v1, const VecDeriv& v2);

    virtual void addDForce(VecDeriv& df1, VecDeriv& df2, const VecDeriv& dx1, const VecDeriv& dx2);

    virtual double getPotentialEnergy(const VecCoord&, const VecCoord&);

    void draw();

};

} // namespace constraint

} // namespace component

} // namespace sofa

#endif
