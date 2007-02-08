#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_BASEFORCEFIELD_H
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_BASEFORCEFIELD_H

#include <sofa/core/objectmodel/BaseObject.h>
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

class BaseForceField : public virtual objectmodel::BaseObject
{
public:
    virtual ~BaseForceField() {}

    virtual void addForce() = 0;

    virtual void addDForce() = 0;

    virtual double getPotentialEnergy() =0;


    virtual void computeMatrix(sofa::defaulttype::SofaBaseMatrix *, double , double , double, unsigned int &) {};

    virtual void contributeToMatrixDimension(unsigned int * const, unsigned int * const) {};

    virtual void computeVector(sofa::defaulttype::SofaBaseVector *, unsigned int &) {};

    virtual void matResUpdatePosition(sofa::defaulttype::SofaBaseVector *, unsigned int &) {};
};

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
