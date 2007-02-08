#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_BASECONSTRAINT_H
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_BASECONSTRAINT_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/componentmodel/behavior/BaseMechanicalState.h>

#include <sofa/defaulttype/SofaBaseMatrix.h>
#include <sofa/defaulttype/SofaBaseVector.h>

#include <vector>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

class BaseConstraint : public virtual objectmodel::BaseObject
{
public:
    virtual ~BaseConstraint() { }

    virtual void projectResponse() = 0; ///< project dx to constrained space (dx models an acceleration)
    virtual void projectVelocity() = 0; ///< project dx to constrained space (dx models a velocity)
    virtual void projectPosition() = 0; ///< project x to constrained space (x models a position)

    virtual void applyConstraint() {};

    virtual void applyConstraint(defaulttype::SofaBaseMatrix *, unsigned int &) {};
    virtual void applyConstraint(defaulttype::SofaBaseVector *, unsigned int &) {};

    virtual BaseMechanicalState* getDOFs() { return NULL; }

    virtual void getConstraintValue(double *) {};
};

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
