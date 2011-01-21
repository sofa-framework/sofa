#ifndef VISUALMODELPOLICY_H
#define VISUALMODELPOLICY_H

#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace gui
{
namespace qt
{
namespace viewer
{

class VisualModelPolicy
{
public:
    virtual void load() = 0;
    virtual void unload() = 0;
protected:

};


class OglModelPolicy : public VisualModelPolicy
{
protected:
    sofa::core::ObjectFactory::ClassEntry* classVisualModel;
public:
    void load()
    {
        // Replace generic visual models with OglModel
        sofa::core::ObjectFactory::AddAlias("VisualModel", "OglModel", true,
                &classVisualModel);

    }
    void unload()
    {
        sofa::core::ObjectFactory::ResetAlias("VisualModel", classVisualModel);

    }
};

}
}
}
}



#endif // VISUALMODELPOLICY_H
