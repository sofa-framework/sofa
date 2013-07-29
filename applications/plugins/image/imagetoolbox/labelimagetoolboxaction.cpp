#include <QColorDialog>

#include "labelimagetoolboxaction.h"
#include "labelimagetoolbox.h"

namespace sofa
{
namespace gui
{
namespace qt
{

LabelImageToolBoxAction::LabelImageToolBoxAction(sofa::component::engine::LabelImageToolBox* lba,QObject *parent) : QObject(parent),
    p_label(lba)
{
    createColorAction();
}

void LabelImageToolBoxAction::setVisible(bool v)
{
    for(int i=0;i<l_actions.size();i++)
    {
        l_actions[i]->setVisible(v);
    }
    
    /*for(int i=0;i<l_widgets.size();i++)
    {
        l_widgets[i]->setEnabled(v);
        l_widgets[i]->setVisible(false);
    }*/
}


void LabelImageToolBoxAction::buttonSelectedOff()
{
    for(int i=0;i<l_actions.size();i++)
    {
        l_actions[i]->setChecked(false);
    }
}

void LabelImageToolBoxAction::setGraphScene(QGraphicsScene *XY,QGraphicsScene *XZ,QGraphicsScene *ZY)
{
    GraphXY = XY;
    GraphXZ = XZ;
    GraphZY = ZY;
    
    addOnGraphs();
    updateGraphs();
}


    void LabelImageToolBoxAction::createColorAction()
    {
        QPixmap pix(20,20);
        
        sofa::defaulttype::Vec4f v = p_label->d_color.getValue();
        QColor c;
        c.setRgbF(v.x(),v.y(),v.z(),v.w());
        
        pix.fill(c);
        QIcon icon(pix);
        QString text("color");
        a_color = new QAction(icon,text,this);
        l_actions.append(a_color);
        
        connect(a_color,SIGNAL(triggered()),this,SLOT(clickColor()));
    }
    
    void LabelImageToolBoxAction::clickColor()
    {
        sofa::defaulttype::Vec4f v = p_label->d_color.getValue();
        QColor c;
        c.setRgbF(v.x(),v.y(),v.z(),v.w());
        
        QColorDialog *diag = new QColorDialog();
        diag->setOption(QColorDialog::ShowAlphaChannel,true);
        diag->setCurrentColor(c);
        
        connect(diag,SIGNAL(colorSelected(QColor)),this,SLOT(selectColor(QColor)));
        
        diag->show();
    }

    void LabelImageToolBoxAction::selectColor(QColor c)
    {
        QColorDialog *diag = qobject_cast<QColorDialog*>(sender());
        
        sofa::defaulttype::Vec4f v(c.redF(),c.greenF(),c.blueF(),c.alphaF());
        p_label->d_color.setValue(v);
        
        QPixmap pix(20,20);
        pix.fill(c);
        QIcon icon(pix);
        a_color->setIcon(icon);
        
        
        diag->deleteLater();
        this->updateColor();
    }
    
    QColor LabelImageToolBoxAction::color()
    {
        sofa::defaulttype::Vec4f v = p_label->d_color.getValue();
        QColor c;
        c.setRgbF(v.x(),v.y(),v.z(),v.w());
        return c;
    }

}
}
}

