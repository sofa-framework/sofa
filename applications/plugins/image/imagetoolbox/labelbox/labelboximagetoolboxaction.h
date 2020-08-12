

#ifndef LABELBOXIMAGETOOLBOXACTION_H
#define LABELBOXIMAGETOOLBOXACTION_H

#include <QPushButton>
#include <QGraphicsLineItem>


#include "../labelimagetoolboxaction.h"
//#include "labelboximagetoolbox.h"

#include <image/image_gui/config.h>

namespace sofa
{

namespace component
{

namespace engine
{
class SOFA_IMAGE_GUI_API LabelBoxImageToolBox;
}}}

namespace sofa
{
namespace gui
{
namespace qt
{

class SOFA_IMAGE_GUI_API LabelBoxImageToolBoxAction : public LabelImageToolBoxAction
{
Q_OBJECT

    QGraphicsPathItem *path[3];
    QGraphicsPolygonItem *poly[12];
    QGraphicsRectItem *rec[3];
    bool sectionIsInclude[3];
    sofa::defaulttype::Vec3i sectionPosition;

public:
    LabelBoxImageToolBoxAction(sofa::component::engine::LabelImageToolBox* lba,QObject *parent);
    ~LabelBoxImageToolBoxAction() override;
    
    sofa::component::engine::LabelBoxImageToolBox* LBITB();

private:

    void createMainCommandWidget();
    void validView();

public slots:
    void addOnGraphs() override;
    void updateGraphs() override;
    void updateColor() override;
    void optionChangeSection(sofa::defaulttype::Vec3i) override;

private slots:
    void selectionPointButtonClick(bool);
    void selectionPointEvent(int mouseevent, const unsigned int axis,const sofa::defaulttype::Vec3d& imageposition,const sofa::defaulttype::Vec3d& position3D,const QString& value);
    void deleteButtonClick();
    void saveButtonClick();
    void loadButtonClick();
    
    
private:
    QPushButton* select;
    
};

}
}
}

#endif // LabelBoxIMAGETOOLBOXACTION_H


