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
    CollisionElement() : mTimeStamp(-1) {}

    virtual ~CollisionElement() { }

    virtual CollisionModel* getCollisionModel() = 0;

    virtual void getBBox(double* minVect, double* maxVect) = 0;
    virtual void getContinuousBBox(double* minVect, double* maxVect, double /*dt*/) { getBBox(minVect, maxVect); }

    /// Test if collisions with another element should be tested.
    /// Default is to reject any self-collisions.
    virtual bool canCollideWith(CollisionElement* elem) {return getCollisionModel() != elem->getCollisionModel();}

    //bool isSelfCollis(CollisionElement* elem) {return getCollisionModel() == elem->getCollisionModel();}

    static void clearAllVisits()
    {
        CurrentTimeStamp(1);
    }

    bool visited() const
    {
        return mTimeStamp == CurrentTimeStamp();
    }

    void setVisited()
    {
        mTimeStamp = CurrentTimeStamp();
    }

protected:
    int mTimeStamp;

    static int CurrentTimeStamp(int incr=0)
    {
        static int ts = 0;
        ts += incr;
        return ts;
    }
};

} // namespace Abstract

} // namespace Sofa

#endif
