#ifndef CONTOURIMAGETOOLBOXACTION_H
#define CONTOURIMAGETOOLBOXACTION_H

#include <image/image_gui/config.h>

#include <QPushButton>
#include <QSpinBox>
#include <QAction>
#include <QGroupBox>

#include <QGraphicsLineItem>

#include <sofa/defaulttype/VecTypes.h>
#include "../labelimagetoolboxaction.h"
//#include "contourimagetoolbox.h"

namespace sofa
{

namespace component
{

namespace engine
{
class SOFA_IMAGE_GUI_API ContourImageToolBoxNoTemplated;
}}}

namespace sofa
{
namespace gui
{
namespace qt
{

class SOFA_IMAGE_GUI_API ContourImageToolBoxAction : public LabelImageToolBoxAction
{
Q_OBJECT

    //QGraphicsLineItem *lineH[3], *lineV[3];

    QGraphicsPathItem *cursor[3];
    QGraphicsPathItem *path[3];
    
    QPushButton* select;
    QSpinBox *vecX, *vecY, *vecZ, *radius;
    QDoubleSpinBox *threshold;
    QGroupBox *posGroup, *radiusGroup, *thresholdGroup;

    sofa::defaulttype::Vec3i sectionPosition;

public:
    typedef sofa::defaulttype::Vec<3,unsigned int> PixCoord;
    typedef helper::vector<PixCoord> VecPixCoord;

    ContourImageToolBoxAction(sofa::component::engine::LabelImageToolBox* lba,QObject *parent);
    ~ContourImageToolBoxAction() override;
    
    sofa::component::engine::ContourImageToolBoxNoTemplated* CITB();

    void setImageSize(int,int,int);
    
private:
    void createMainCommands();
    void createPosition();
    void createRadius();
    void createThreshold();

    QPainterPath drawCursor(double x,double y);
    QPainterPath drawSegment(VecPixCoord &v, unsigned int axis);
    void drawSegment();

public slots:
    void addOnGraphs() override;
    void updateGraphs() override;
    void updateColor() override;
    void optionChangeSection(sofa::defaulttype::Vec3i) override;

    
private slots:
    void selectionPointButtonClick(bool);
    void selectionPointEvent(int mouseevent, const unsigned int axis,const sofa::defaulttype::Vec3d& imageposition,const sofa::defaulttype::Vec3d& position3D,const QString& value);
    void sectionButtonClick();
    
    void positionModified();
    void radiusModified();
    void thresholdModified();
    
private:
    
};

}
}
}

#endif // CONTOURIMAGETOOLBOXACTION_H
