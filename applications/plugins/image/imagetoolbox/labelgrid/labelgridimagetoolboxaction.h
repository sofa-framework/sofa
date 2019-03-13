#ifndef LABELGRIDIMAGETOOLBOXACTION_H
#define LABELGRIDIMAGETOOLBOXACTION_H

#include <image/image_gui/config.h>

#include <QSpinBox>
#include <QPushButton>


#include "../labelimagetoolboxaction.h"
//#include "labelpointimagetoolbox.h"

namespace sofa
{

namespace component
{

namespace engine
{
class SOFA_IMAGE_GUI_API LabelGridImageToolBoxNoTemplated;
}}}

namespace sofa
{
namespace gui
{
namespace qt
{

class SOFA_IMAGE_GUI_API LabelGridImageToolBoxAction : public LabelImageToolBoxAction
{
Q_OBJECT

    //QGraphicsLineItem *lineH[3], *lineV[3];
    QGraphicsPathItem *path[3];

    QSpinBox * mainAxisSpin;
    QSpinBox * secondAxisSpin;

public:
    LabelGridImageToolBoxAction(sofa::component::engine::LabelImageToolBox* lba,QObject *parent);
    ~LabelGridImageToolBoxAction() override;
    
    sofa::component::engine::LabelGridImageToolBoxNoTemplated* LGITB();

private:
    void createMainCommands();
    //void createBoxCommands();
    void createGridCommands();
    //void createNormalCommands();

    void lineTo(const unsigned int axis,const sofa::defaulttype::Vec3d& imageposition);
    void moveTo(const unsigned int axis,const sofa::defaulttype::Vec3d& imageposition);


public slots:
    void addOnGraphs() override;
    void updateGraphs() override;
    void updateColor() override;

private slots:
    void selectionPointEvent(int mouseevent, const unsigned int axis,const sofa::defaulttype::Vec3d& imageposition,const sofa::defaulttype::Vec3d& position3D,const QString& value);
    
    void executeButtonClick();
    
private:
    QPushButton* select;
    
};

}
}
}

#endif // LabelGridIMAGETOOLBOXACTION_H



