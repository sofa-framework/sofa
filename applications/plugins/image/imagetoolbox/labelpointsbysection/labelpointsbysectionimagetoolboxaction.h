#ifndef LABELPOINTSBYSECTIONIMAGETOOLBOXACTION_H
#define LABELPOINTSBYSECTIONIMAGETOOLBOXACTION_H

#include <QAction>
#include <QGraphicsLineItem>


#include "../labelimagetoolboxaction.h"
//#include "LabelPointsBySectionImageToolBox.h"

#include "initImage.h"

namespace sofa
{

namespace component
{

namespace engine
{
class LabelPointsBySectionImageToolBox;
}}}

namespace sofa
{
namespace gui
{
namespace qt
{

class SOFA_IMAGE_API LabelPointsBySectionImageToolBoxAction : public LabelImageToolBoxAction
{
Q_OBJECT

    QGraphicsLineItem *lineH[3], *lineV[3];

public:
    LabelPointsBySectionImageToolBoxAction(sofa::component::engine::LabelImageToolBox* lba,QObject *parent);
    ~LabelPointsBySectionImageToolBoxAction();
    
    sofa::component::engine::LabelPointsBySectionImageToolBox* LPBSITB();
    
    
    void createListPointWidget();
    
private:

public slots:
    virtual void addOnGraphs();
    virtual void updateGraphs();
    virtual void updateColor();
    
private slots:
    void selectionPointButtonClick(bool);
    void selectionPointEvent(int mouseevent, const unsigned int axis,const sofa::defaulttype::Vec3d& imageposition,const sofa::defaulttype::Vec3d& position3D,const QString& value);
    void sectionButtonClick();
    
    
private:
    QAction* select;
    
};

}
}
}

#endif // LABELPOINTSBYSECTIONIMAGETOOLBOXACTION_H
