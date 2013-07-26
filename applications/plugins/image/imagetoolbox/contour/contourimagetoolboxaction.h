#ifndef CONTOURIMAGETOOLBOXACTION_H
#define CONTOURIMAGETOOLBOXACTION_H

#include <QtGui>
#include <QAction>
#include <QGraphicsLineItem>



#include "../labelimagetoolboxaction.h"
//#include "contourimagetoolbox.h"

#include "initImage.h"

namespace sofa
{

namespace component
{

namespace engine
{
class ContourImageToolBoxNoTemplated;
}}}

namespace sofa
{
namespace gui
{
namespace qt
{

class SOFA_IMAGE_API ContourImageToolBoxAction : public LabelImageToolBoxAction
{
Q_OBJECT

    QGraphicsLineItem *lineH[3], *lineV[3];
    
    QSpinBox *vecX, *vecY, *vecZ, *radius;
    QDoubleSpinBox *threshold;
    QGroupBox *posGroup, *radiusGroup, *thresholdGroup;

public:
    ContourImageToolBoxAction(sofa::component::engine::LabelImageToolBox* lba,QObject *parent);
    ~ContourImageToolBoxAction();
    
    sofa::component::engine::ContourImageToolBoxNoTemplated* CITB();

    void setImageSize(int,int,int);
    
private:
    void createPosition();
    void createRadius();
    void createThreshold();

public slots:
    virtual void addOnGraphs();
    virtual void updateGraphs();
    
private slots:
    void selectionPointButtonClick(bool);
    void selectionPointEvent(int mouseevent, const unsigned int axis,const sofa::defaulttype::Vec3d& imageposition,const sofa::defaulttype::Vec3d& position3D,const QString& value);
    void sectionButtonClick();
    
    void positionModified();
    void radiusModified();
    void thresholdModified();
    
};

}
}
}

#endif // CONTOURIMAGETOOLBOXACTION_H
