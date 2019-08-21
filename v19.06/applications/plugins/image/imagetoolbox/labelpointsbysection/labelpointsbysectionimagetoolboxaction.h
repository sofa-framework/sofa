

#ifndef LABELPOINTSBYSECTIONIMAGETOOLBOXACTION_H
#define LABELPOINTSBYSECTIONIMAGETOOLBOXACTION_H

#undef CursorShape // conflicts with X.h, CursorShape
#undef Bool // conflicts with macro Bool in Xlib.h

#include <QAction>
#include <QGraphicsLineItem>

#include "tablewidget.h"

#include "../labelimagetoolboxaction.h"
//#include "LabelPointsBySectionImageToolBox.h"



#include <image/image_gui/config.h>

namespace sofa
{

namespace component
{

namespace engine
{
class SOFA_IMAGE_GUI_API LabelPointsBySectionImageToolBox;
}}}

namespace sofa
{
namespace gui
{
namespace qt
{

class SOFA_IMAGE_GUI_API LabelPointsBySectionImageToolBoxAction : public LabelImageToolBoxAction
{
Q_OBJECT
public:
    typedef TableWidgetForLabelPointBySectionToolBoxAction::Point Point;
    typedef TableWidgetForLabelPointBySectionToolBoxAction::VecCoord VecPointSection;
    typedef TableWidgetForLabelPointBySectionToolBoxAction::MapSection MapSection;

private:
    //QGraphicsLineItem *lineH[3], *lineV[3];

    QGraphicsPathItem* path[3],*oldpath[3];

    
    TableWidgetForLabelPointBySectionToolBoxAction * tablewidget;
    
    QPushButton *select, *xyAxis, *xzAxis, *zyAxis;

    MapSection mapsection;

    bool addPoints;
    int currentSlide;
    int oldSlide;
public:
    LabelPointsBySectionImageToolBoxAction(sofa::component::engine::LabelImageToolBox* lba,QObject *parent);
    ~LabelPointsBySectionImageToolBoxAction() override;
    
    sofa::component::engine::LabelPointsBySectionImageToolBox* LPBSITB();
    
    void createMainCommandWidget();
    void createListPointWidget();
    void createAxisSelectionWidget();

private:
    int currentAxis();
    void setAxis(int);

public slots:
    void addOnGraphs() override;
    void updateGraphs() override;
    void updateColor() override;
    void optionChangeSection(sofa::defaulttype::Vec3i) override;
    void changeSection(int);
    sofa::defaulttype::Vec3i changeSection2(int,bool force=false);
    void mouseMove(const unsigned int axis,const sofa::defaulttype::Vec3d& imageposition,const sofa::defaulttype::Vec3d& position3D,const QString& value) override;
    void addToPath(const unsigned int axis,const sofa::defaulttype::Vec3d& imageposition,bool forceMoveTo=false);

private slots:
    void selectionPointButtonClick(bool);
    void selectionPointEvent(int mouseevent, const unsigned int axis,const sofa::defaulttype::Vec3d& imageposition,const sofa::defaulttype::Vec3d& position3D,const QString& value);
    //void sectionButtonClick();
    void axisChecked(bool b);

    void updateData();
    void reloadData();
    void loadFileData();
    void saveFileData();
private:
    
};

}
}
}

#endif // LABELPOINTSBYSECTIONIMAGETOOLBOXACTION_H


