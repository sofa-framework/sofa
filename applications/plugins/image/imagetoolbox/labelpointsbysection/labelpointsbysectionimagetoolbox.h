#ifndef LABELPOINTSBYSECTIONIMAGETOOLBOX_H
#define LABELPOINTSBYSECTIONIMAGETOOLBOX_H

#include <sofa/component/component.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>

#include "labelpointsbysectionimagetoolboxaction.h"
#include "../labelimagetoolbox.h"



#include "initImage.h"




namespace sofa
{

namespace component
{

namespace engine
{

class SOFA_IMAGE_API LabelPointsBySectionImageToolBox: public LabelImageToolBox
{
public:
    SOFA_CLASS(LabelPointsBySectionImageToolBox,LabelImageToolBox);
    
    LabelPointsBySectionImageToolBox():LabelImageToolBox()
        , d_ip(initData(&d_ip, "imageposition",""))
        , d_p(initData(&d_p, "3Dposition",""))
        , d_axis(initData(&d_axis, (unsigned int)4,"axis",""))
        , d_value(initData(&d_value,"value",""))
    {
    
    }
    
    virtual void init()
    {
        addOutput(&d_ip);
        addOutput(&d_p);
        addOutput(&d_axis);
        addOutput(&d_value);
        
    }
    
    virtual sofa::gui::qt::LabelImageToolBoxAction* createTBAction(QObject*parent)
    {
        return new sofa::gui::qt::LabelPointsBySectionImageToolBoxAction(this,parent);
    }
    
public:
    Data<sofa::defaulttype::Vec3d> d_ip;
    Data<sofa::defaulttype::Vec3d> d_p;
    Data<unsigned int> d_axis;
    Data<std::string> d_value;
};


}}}

#endif // LabelPointsBySectionImageToolBox_H
