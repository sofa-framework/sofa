#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_BASEMASS_H
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_BASEMASS_H

#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

class BaseMass : public virtual objectmodel::BaseObject
{
public:
    virtual ~BaseMass() { }

    virtual void addMDx() = 0; ///< f += M dx using dof->getF() and dof->getDx()

    virtual void accFromF() = 0; ///< dx = M^-1 f using dof->getF() and dof->getDx()

    virtual double getKineticEnergy() = 0;  ///< vMv/2 using dof->getV()

};

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
