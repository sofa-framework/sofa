#ifndef OGRESOFAVIEWER_H
#define OGRESOFAVIEWER_H

#include <sofa/gui/qt/viewer/VisualModelPolicy.h>
#include <sofa/gui/qt/viewer/SofaViewer.h>
namespace sofa
{
namespace gui
{
namespace qt
{
namespace viewer
{

class OgreVisualModelPolicy : public VisualModelPolicy
{
protected:
    sofa::core::ObjectFactory::ClassEntry* classVisualModel;
    sofa::core::ObjectFactory::ClassEntry* classOglModel;
public:
    void load()
    {
        // Replace OpenGL visual models with OgreVisualModel
        sofa::core::ObjectFactory::AddAlias("OglModel", "OgreVisualModel", true, &classOglModel);
        sofa::core::ObjectFactory::AddAlias("VisualModel", "OgreVisualModel", true, &classVisualModel);
    }

    void unload()
    {
        sofa::core::ObjectFactory::ResetAlias("OglModel", classOglModel);
        sofa::core::ObjectFactory::ResetAlias("VisualModel", classVisualModel);
    }

};

typedef sofa::gui::qt::viewer::CustomPolicySofaViewer< OgreVisualModelPolicy > OgreSofaViewer;

}
}
}
}


#endif // OGRESOFAVIEWER_H
