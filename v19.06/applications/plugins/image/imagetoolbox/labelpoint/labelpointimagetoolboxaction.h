#ifndef LABELPOINTIMAGETOOLBOXACTION_H
#define LABELPOINTIMAGETOOLBOXACTION_H

#include <QPushButton>

#include "../labelimagetoolboxaction.h"

#include <image/image_gui/config.h>

namespace sofa
{

namespace component
{

namespace engine
{
class SOFA_IMAGE_GUI_API LabelPointImageToolBox;
}}}

namespace sofa
{
namespace gui
{
namespace qt
{

class SOFA_IMAGE_GUI_API LabelPointImageToolBoxAction : public LabelImageToolBoxAction
{
Q_OBJECT

    QGraphicsLineItem *lineH[3], *lineV[3];

public:
    LabelPointImageToolBoxAction(sofa::component::engine::LabelImageToolBox* lba,QObject *parent);
    ~LabelPointImageToolBoxAction() override;
    
    sofa::component::engine::LabelPointImageToolBox* LPITB();

private:

public slots:
    void addOnGraphs() override;
    void updateGraphs() override;
    void updateColor() override;
    
private slots:
    void selectionPointButtonClick(bool);
    void selectionPointEvent(int mouseevent, const unsigned int axis,const sofa::defaulttype::Vec3d& imageposition,const sofa::defaulttype::Vec3d& position3D,const QString& value);
    void sectionButtonClick();
    
    
private:
    QPushButton* select;
    
};

}
}
}

#endif // LABELPOINTIMAGETOOLBOXACTION_H



