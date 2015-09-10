#ifndef LABELIMAGETOOLBOXACTION_H
#define LABELIMAGETOOLBOXACTION_H

#include <QtGui>
#include <image/image_gui/config.h>
#include <sofa/defaulttype/VecTypes.h>
//#include "labelimagetoolbox.h"


namespace sofa
{

namespace component
{

namespace engine
{
class SOFA_IMAGE_GUI_API LabelImageToolBox;
}}}

namespace sofa
{
namespace gui
{
namespace qt
{


class SOFA_IMAGE_GUI_API LabelImageToolBoxAction : public QObject//QGroupBox//QWidget
{
    Q_OBJECT
    
protected:
    //QList<QAction*> l_actions;

    typedef QVBoxLayout Layout;

    sofa::component::engine::LabelImageToolBox *p_label;
    Layout *mainlayout;

    
    sofa::defaulttype::Vec3i d_section;
    
    QGraphicsScene *GraphXY;
    QGraphicsScene *GraphXZ;
    QGraphicsScene *GraphZY;
    

    
public:
    explicit LabelImageToolBoxAction(sofa::component::engine::LabelImageToolBox* lba,QObject *parent = 0);
    
    QLayout * layout(){return mainlayout;}

     //QList<QAction*>& getActions(){return l_actions;}
     //QList<QWidget*>& getWidgets(){return l_widgets;}
    
protected:
    inline void addWidget(QWidget *w)
    {
        mainlayout->addWidget(w);
    }

    inline void addLayout(QLayout *w)
    {
        mainlayout->addLayout(w);
    }
    
    inline void addStretch()
    {
        mainlayout->addStretch();
    }

signals:
    
    
public slots:
    
    void buttonSelectedOff();
    void setGraphScene(QGraphicsScene *XY,QGraphicsScene *XZ,QGraphicsScene *ZY);
    virtual void addOnGraphs()=0;
    virtual void updateGraphs()=0;
    virtual void updateColor()=0;
    QColor color();
    void clickColor();

    virtual void mouseMove(const unsigned int /*axis*/,const sofa::defaulttype::Vec3d& /*imageposition*/,const sofa::defaulttype::Vec3d& /*position3D*/,const QString& /*value*/){}
    virtual void optionChangeSection(sofa::defaulttype::Vec3i){}
    
signals:
    void clickImage(int mouseevent, const unsigned int axis,const sofa::defaulttype::Vec3d& imageposition,const sofa::defaulttype::Vec3d& position3D,const QString& value);

    void sectionChanged(sofa::defaulttype::Vec3i);
    
    void guiChangeSection(defaulttype::Vec3i s);

    void colorChanged();

    void updateImage();

private:
    //QPushButton *a_color;
    
    //void createColorAction();
private slots:

    void selectColor(QColor c);

    
};



}
}
}

#endif // LABELIMAGETOOLBOXACTION_H*/
