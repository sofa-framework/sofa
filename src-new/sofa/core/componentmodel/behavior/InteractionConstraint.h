#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_INTERACTIONCONSTRAINT_H
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_INTERACTIONCONSTRAINT_H

#include <sofa/core/componentmodel/behavior/Constraint.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

class InteractionConstraint : public BaseConstraint
{
public:
    virtual BaseMechanicalState* getMechModel1() = 0;
    virtual BaseMechanicalState* getMechModel2() = 0;
};

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
