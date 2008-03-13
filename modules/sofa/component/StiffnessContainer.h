#ifndef STIFFNESSCONTAINER_H_
#define STIFFNESSCONTAINER_H_

namespace sofa
{

namespace component
{

class StiffnessContainer : public virtual sofa::core::objectmodel::BaseObject
{
public:
    virtual double getStiffness(unsigned int index) = 0;
};

}

}

#endif /*STIFFNESSCONTAINER_H_*/
