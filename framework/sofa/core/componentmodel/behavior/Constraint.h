#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_CONSTRAINT_H
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_CONSTRAINT_H

#include <sofa/core/componentmodel/behavior/BaseConstraint.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{


template<class DataTypes>
class Constraint : public BaseConstraint
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::VecConst VecConst;

    Constraint(MechanicalState<DataTypes> *mm = NULL);

    virtual ~Constraint();

    DataField<Real> endTime;  ///< Time when the constraint becomes inactive (-1 for infinitely active)
    bool isActive() const; ///< if false, the constraint does nothing

    virtual void init();

    virtual void projectResponse(); ///< project dx to constrained space
    virtual void projectVelocity(); ///< project dx to constrained space (dx models a velocity)
    virtual void projectPosition(); ///< project x to constrained space (x models a position)

    virtual void projectResponse(VecDeriv& dx) = 0; ///< project dx to constrained space
    virtual void projectVelocity(VecDeriv& dx)=0; ///< project dx to constrained space (dx models a velocity)
    virtual void projectPosition(VecCoord& x)=0; ///< project x to constrained space (x models a position)

    virtual void applyConstraint(); // Pure virtual would be better
    virtual void applyConstraint(VecConst& /*c*/) {};

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL)
            return false;
        return BaseObject::canCreate(obj, context, arg);
    }

protected:
    MechanicalState<DataTypes> *mstate;
};

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
