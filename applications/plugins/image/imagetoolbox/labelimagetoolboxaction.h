#ifndef LABELIMAGETOOLBOXACTION_H
#define LABELIMAGETOOLBOXACTION_H

#include <QAction>
#include <QObject>
#include <QGraphicsScene>
#include <QList>

#include <sofa/defaulttype/VecTypes.h>
//#include "labelimagetoolbox.h"


namespace sofa
{

namespace component
{

namespace engine
{
class LabelImageToolBox;
}}}

namespace sofa
{
namespace gui
{
namespace qt
{


class LabelImageToolBoxAction : public QObject
{
    Q_OBJECT
    
protected:
    QList<QAction*> l_actions;
    sofa::component::engine::LabelImageToolBox *p_label;
    
    sofa::defaulttype::Vec3i d_section;
    
    QGraphicsScene *GraphXY;
    QGraphicsScene *GraphXZ;
    QGraphicsScene *GraphZY;
    

    
public:
    explicit LabelImageToolBoxAction(sofa::component::engine::LabelImageToolBox* lba,QObject *parent = 0);
    
     QList<QAction*>& getActions(){return l_actions;}
     //QList<QWidget*>& getWidgets(){return l_widgets;}
    
private:
    
    
signals:
    
    
public slots:
    
    void setVisible(bool v);
    void buttonSelectedOff();
    void setGraphScene(QGraphicsScene *XY,QGraphicsScene *XZ,QGraphicsScene *ZY);
    virtual void addOnGraphs()=0;
    virtual void updateGraphs()=0;
    virtual void updateColor()=0;
    QColor color();
    
    void guiChangeSection(defaulttype::Vec3i s)
    {
        d_section = s;
    }
    
signals:
    void clickImage(int mouseevent, const unsigned int axis,const sofa::defaulttype::Vec3d& imageposition,const sofa::defaulttype::Vec3d& position3D,const QString& value);
    
    void sectionChangeGui(sofa::defaulttype::Vec3i);
    
    
private:
    QAction *a_color;
    
    void createColorAction();
private slots:
    void clickColor();
    void selectColor(QColor c);
    
};



}
}
}

#endif // LABELIMAGETOOLBOXACTION_H*/
