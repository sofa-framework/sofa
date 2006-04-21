#ifndef SOFA_ABSTRACT_BEHAVIORMODEL_H
#define SOFA_ABSTRACT_BEHAVIORMODEL_H

#include "BaseObject.h"

namespace Sofa
{

namespace Abstract
{

/*! \class BehaviorModel
 *  \brief An interface inherited by all BehaviorModel
 *  \author Fonteneau Sylvere
 *  \version 0.1
 *  \date    02/22/2004
 *
 *  <P>This Interface is used for MechanicalGroup and "black-box" BehaviorModel<BR>
 *  All behavior model inherit of this Interface, and each object has to implement the updatePosition method
 *  <BR>updatePosition corresponds to the computation of a new simulation step<BR>
 */

class BehaviorModel : public virtual BaseObject
{
public:
    virtual ~BehaviorModel() {}

    virtual void init() = 0;

    /// Computation of a new simulation step.
    virtual void updatePosition(double dt) = 0;

    virtual void applyTranslation(double /*dx*/, double /*dy*/, double /*dz*/) { }
    virtual void applyScale(double /*sx*/, double /*sy*/, double /*sz*/, double /*smass*/) { }
    virtual void applyRotation(double /*ax*/, double /*ay*/, double /*az*/, double /*angle*/) { }
};

} // namespace Abstract

} // namespace Sofa

#endif
