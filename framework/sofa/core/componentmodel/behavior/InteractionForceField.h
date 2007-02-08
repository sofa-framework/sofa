#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_INTERACTIONFORCEFIELD_H
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_INTERACTIONFORCEFIELD_H

#include <sofa/core/componentmodel/behavior/BaseForceField.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/defaulttype/SofaBaseMatrix.h>
#include <sofa/defaulttype/SofaBaseVector.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

class InteractionForceField : public BaseForceField
{
public:
    virtual BaseMechanicalState* getMechModel1() = 0;
    virtual BaseMechanicalState* getMechModel2() = 0;

    // ForceField using matrix interface
    virtual void contributeToMatrixDimension(unsigned int * const, unsigned int * const) {};
    virtual void computeMatrix(sofa::defaulttype::SofaBaseMatrix *, double , double , double, unsigned int &) {};
    virtual void computeVector(sofa::defaulttype::SofaBaseVector *, unsigned int &) {};
    virtual void matResUpdatePosition(sofa::defaulttype::SofaBaseVector *, unsigned int &) {};
};

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
