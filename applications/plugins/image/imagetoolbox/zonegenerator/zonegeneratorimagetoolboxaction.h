#ifndef ZONEGENERATORIMAGETOOLBOXACTION_H
#define ZONEGENERATORIMAGETOOLBOXACTION_H

#include <QPushButton>
#include <QLineEdit>
#include <QAction>
#include <QGraphicsLineItem>



#include "../labelimagetoolboxaction.h"
//#include "zonegeneratorimagetoolbox.h"

#include <image/image_gui/config.h>

namespace sofa
{

namespace component
{

namespace engine
{
class SOFA_IMAGE_GUI_API ZoneGeneratorImageToolBoxNoTemplated;
}}}

namespace sofa
{
namespace gui
{
namespace qt
{

class SOFA_IMAGE_GUI_API ZoneGeneratorImageToolBoxAction : public LabelImageToolBoxAction
{
Q_OBJECT

    //QGraphicsLineItem *lineH[3], *lineV[3];


    QPushButton* generate;
    QLineEdit *seed;
    QLineEdit *density;

public:

    ZoneGeneratorImageToolBoxAction(sofa::component::engine::LabelImageToolBox* lba,QObject *parent);
    ~ZoneGeneratorImageToolBoxAction() override;
    
    sofa::component::engine::ZoneGeneratorImageToolBoxNoTemplated* ZGITB();

    //void setImageSize(int,int,int);
    
private:
    void createMainCommands();

    //QPainterPath drawCursor(double x,double y);
    //QPainterPath drawSegment(VecPixCoord &v, unsigned int axis);
    //void drawSegment();

public slots:
    void addOnGraphs() override;
    void updateGraphs() override;
    void updateColor() override{}
    void optionChangeSection(sofa::defaulttype::Vec3i) override;

    
private slots:
    void generatePointButtonClick();
    void selectionPointEvent(int mouseevent, const unsigned int axis,const sofa::defaulttype::Vec3d& imageposition,const sofa::defaulttype::Vec3d& position3D,const QString& value);
    //void sectionButtonClick();
    
    //void positionModified();
    //void radiusModified();
    //void thresholdModified();
    
private:
    
};

}
}
}

#endif // ZoneGeneratorImageToolBoxACTION_H
