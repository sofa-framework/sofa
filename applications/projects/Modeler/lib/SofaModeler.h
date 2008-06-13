
#ifndef SOFA_MODELER_H
#define SOFA_MODELER_H


#include "Modeler.h"
#include "GraphModeler.h"
#include <map>
#include <vector>
#include <string>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/Factory.h>
#include <sofa/simulation/tree/GNode.h>

#include <sofa/gui/qt/GraphListenerQListView.h>

#ifdef SOFA_QT4
#include <Q3ListView>
#include <Q3TextDrag>
#include <QPushButton>
#else
#include <qlistview.h>
#include <qdragobject.h>
#include <qpushbutton.h>
#endif


namespace sofa
{

namespace gui
{

namespace qt
{

#ifdef SOFA_QT4
typedef Q3ListView QListView;
#endif

typedef sofa::core::ObjectFactory::ClassEntry ClassInfo;
typedef sofa::core::ObjectFactory::Creator    ClassCreator;

using sofa::simulation::tree::GNode;


class SofaModeler : public ::Modeler
{

    Q_OBJECT
public :

    SofaModeler();
    ~SofaModeler() {};

public slots:
    void dragComponent();
    void changeComponent(ClassInfo *currentComponent);
    void changeInformation(QListViewItem *);
    void newGNode();

    void fileNew() {graph->fileNew();};
    void fileReload() {graph->fileReload();}
    void fileOpen() {graph->fileOpen();};
    void fileSave() {graph->fileSave();};
    void fileSaveAs() {graph->fileSaveAs();};
    void fileExit() {exit(0);};
    void changeNameWindow(std::string filename);
    void editUndo() {graph->editUndo();}
    void editRedo() {graph->editRedo();}
    ClassInfo* getInfoFromName(std::string name);


    void keyPressEvent ( QKeyEvent * e );
protected:
    GraphModeler *graph;

    std::map< const QObject* , std::pair<ClassInfo*, QObject*> > mapComponents;

};
}
}
}
#endif
