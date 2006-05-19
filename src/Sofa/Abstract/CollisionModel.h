#ifndef SOFA_ABSTRACT_COLLISIONMODEL_H
#define SOFA_ABSTRACT_COLLISIONMODEL_H

#include <vector>
#include "CollisionElement.h"
#include "BaseObject.h"

namespace Sofa
{

namespace Abstract
{

class BehaviorModel;

/*! \class CollisionModel
  *  \brief An interface inherited by CollisionObject
  *  \author Fonteneau Sylvere
  *  \version 0.1
  *  \date    02/22/2004
  *
  *  <P>This Interface is used for CollisionModel like SPHModel, ...<BR>
  *  All collision model inherit of this Interface. <BR>
  *  The collision model has a view on mapping between a CollisionModel and a BehaviorModel. <br>
  *  Because when a collision is detected by the environment collision, <BR>
  *  we have only access to a collision model by element of a collision model <BR>
  *  Then, when we have to report collision to a BehaviorModel <BR>
  *  we use the mapping between CollisionModel and BehaviorModel <BR>
  *  </P>
  */

class CollisionModel : public virtual BaseObject
{
public:
    virtual ~CollisionModel() { }

    virtual std::vector<CollisionElement*> & getCollisionElements() = 0;

    virtual CollisionModel* getNext() = 0;
    virtual CollisionModel* getPrevious() = 0;

    virtual bool isActive() { return true; }

    virtual void computeSphereVolume() {}

    virtual void computeBoundingBox() {}

    virtual void computeContinueBoundingBox() {}

    virtual BehaviorModel* getObject() = 0;

    CollisionModel* getFirst()
    {
        CollisionModel *cm = this;
        CollisionModel *cm2;
        while ((cm2 = cm->getPrevious())!=NULL)
            cm = cm2;
        return cm;
    }

    CollisionModel* getLast()
    {
        CollisionModel *cm = this;
        CollisionModel *cm2;
        while ((cm2 = cm->getNext())!=NULL)
            cm = cm2;
        return cm;
    }
};

} // namespace Abstract

} // namespace Sofa

#endif
