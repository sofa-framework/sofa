#ifndef SOFA_CORE_INTERACTIONFORCEFIELD_H
#define SOFA_CORE_INTERACTIONFORCEFIELD_H

#include "ForceField.h"
#include "Sofa-old/Components/Common/SofaBaseMatrix.h"
#include "Sofa-old/Components/Common/SofaBaseVector.h"

namespace Sofa
{

namespace Core
{

class InteractionForceField : public BasicForceField
{
public:
    virtual BasicMechanicalModel* getMechModel1() = 0;
    virtual BasicMechanicalModel* getMechModel2() = 0;

    // ForceField using matrix interface
    virtual void contributeToMatrixDimension(unsigned int * const, unsigned int * const) {};
    virtual void computeMatrix(Sofa::Components::Common::SofaBaseMatrix *, double , double , double, unsigned int &) {};
    virtual void computeVector(Sofa::Components::Common::SofaBaseVector *, unsigned int &) {};
    virtual void matResUpdatePosition(Sofa::Components::Common::SofaBaseVector *, unsigned int &) {};
};

} // namespace Core

} // namespace Sofa

#endif
