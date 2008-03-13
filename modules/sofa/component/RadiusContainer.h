#ifndef RADIUSCONTAINER_H_
#define RADIUSCONTAINER_H_

namespace sofa
{

namespace component
{

class RadiusContainer : public virtual sofa::core::objectmodel::BaseObject
{
public:
    virtual double getRadius(unsigned int index) = 0;
};

}

}
#endif /*RADIUSCONTAINER_H_*/
