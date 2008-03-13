#ifndef POISSONCONTAINER_
#define POISSONCONTAINER_

namespace sofa
{

namespace component
{

class PoissonContainer : public virtual sofa::core::objectmodel::BaseObject
{
public:
    virtual double getPoisson(unsigned int index) = 0;
};

}

}


#endif /*POISSONCONTAINER_*/
