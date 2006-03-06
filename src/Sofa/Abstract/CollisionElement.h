#ifndef SOFA_ABSTRACT_COLLISIONELEMENT_H
#define SOFA_ABSTRACT_COLLISIONELEMENT_H

#include <vector>

namespace Sofa
{

namespace Abstract
{

class CollisionModel;

class CollisionElement
{
public:
    virtual ~CollisionElement() { }

    virtual CollisionModel* getCollisionModel() = 0;

    virtual void getBBox(double* minVect, double* maxVect) = 0;

    bool isSelfCollis(CollisionElement* elem) {return getCollisionModel() == elem->getCollisionModel();};
};

} // namespace Abstract

} // namespace Sofa

#endif
