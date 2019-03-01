#ifndef LABELPOINTIMAGETOOLBOX_H
#define LABELPOINTIMAGETOOLBOX_H

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>

#include <QDataStream>
#include "labelpointimagetoolboxaction.h"
#include "../labelimagetoolbox.h"




#include <image/image_gui/config.h>




namespace sofa
{

namespace component
{

namespace engine
{

class SOFA_IMAGE_GUI_API LabelPointImageToolBox: public LabelImageToolBox
{
public:
    SOFA_CLASS(LabelPointImageToolBox,LabelImageToolBox);
    
    LabelPointImageToolBox():LabelImageToolBox()
        , d_ip(initData(&d_ip, "imageposition",""))
        , d_p(initData(&d_p, "3Dposition",""))
        , d_axis(initData(&d_axis, (unsigned int)4,"axis",""))
        , d_value(initData(&d_value,"value",""))
    {
    
    }
    
    void init() override
    {
        addOutput(&d_ip);
        addOutput(&d_p);
        addOutput(&d_axis);
        addOutput(&d_value);
        
    }
    
    sofa::gui::qt::LabelImageToolBoxAction* createTBAction(QWidget*parent=NULL) override
    {
        return new sofa::gui::qt::LabelPointImageToolBoxAction(this,parent);
    }
    
public:
    Data<sofa::defaulttype::Vec3d> d_ip;
    Data<sofa::defaulttype::Vec3d> d_p;
    Data<unsigned int> d_axis;
    Data<std::string> d_value;
};


}}}

#endif // LABELPOINTIMAGETOOLBOX_H


