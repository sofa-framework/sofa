#ifndef IMAGETOOLBOXBASICACTIONWIDGET_H
#define IMAGETOOLBOXBASICACTIONWIDGET_H

#include <QAction>
#include <QMessageBox>
#include <QToolBar>

#include "imagetoolboxcentralwidget.h"

namespace sofa
{
namespace gui
{
namespace qt
{

class SOFA_IMAGE_GUI_API ImageToolBoxBasicActionWidget: public QToolBar
{
Q_OBJECT

    QAction *graphXY_Visible;
    QAction *graphXZ_Visible;
    QAction *graphZY_Visible;
    QAction *visualModel;
    QAction *help;

public:
    ImageToolBoxBasicActionWidget():QToolBar("BasicAction")
    {
        this->setToolTip("BasicAction");
        
        graphXY_Visible = this->addAction("XY");
        graphXY_Visible->setCheckable(true);
        graphXY_Visible->setChecked(true);
        
        graphXZ_Visible = this->addAction("XZ");
        graphXZ_Visible->setCheckable(true);
        graphXZ_Visible->setChecked(true);
        
        graphZY_Visible = this->addAction("ZY");
        graphZY_Visible->setCheckable(true);
        graphZY_Visible->setChecked(true);
        
        this->addSeparator();
        
        visualModel = this->addAction("VisualModel");
        visualModel->setCheckable(true);
        visualModel->setChecked(true);

        this->addSeparator();

        help = this->addAction("?");
    }
    
    void connectCentralW(ImageToolBoxCentralWidget* cw)
    {
        connect(cw,SIGNAL(setCheckedXY(bool)),graphXY_Visible,SLOT(setChecked(bool)));
        connect(graphXY_Visible,SIGNAL(toggled(bool)),cw,SLOT(setVisibleXY(bool)));
        
        connect(cw,SIGNAL(setCheckedXZ(bool)),graphXZ_Visible,SLOT(setChecked(bool)));
        connect(graphXZ_Visible,SIGNAL(toggled(bool)),cw,SLOT(setVisibleXZ(bool)));
        
        connect(cw,SIGNAL(setCheckedZY(bool)),graphZY_Visible,SLOT(setChecked(bool)));
        connect(graphZY_Visible,SIGNAL(toggled(bool)),cw,SLOT(setVisibleZY(bool)));
        
        //connect(cw,SIGNAL(setCheckedZY(bool)),graphZY_Visible,SLOT(setChecked(bool)));
        connect(visualModel,SIGNAL(toggled(bool)),cw,SLOT(setVisualModel(bool)));

        connect(help,SIGNAL(triggered()),this,SLOT(helpme()));
    }
public slots:
    void helpme()
    {
        QMessageBox::information(this,"HelpMe","Zoom In: ctrl + Drag with mousse left click.\n Zoom Out: ctrl + mouse left double click.");
    }
    
    



};

}
}
}



#endif // IMAGETOOLBOXBASICACTIONWIDGET_H
