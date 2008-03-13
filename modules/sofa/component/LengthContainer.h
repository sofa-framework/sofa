#ifndef LENGTHCONTAINER_H_
#define LENGTHCONTAINER_H_

namespace sofa
{

namespace component
{

class LengthContainer : public virtual sofa::core::objectmodel::BaseObject
{
public:
    virtual double getLength(unsigned int index) = 0;
};

}

}
#endif /*LENGTHCONTAINER_H_*/
