/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This program is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU General Public License as published by the Free   *
* Software Foundation; either version 2 of the License, or (at your option)    *
* any later version.                                                           *
*                                                                              *
* This program is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for     *
* more details.                                                                *
*                                                                              *
* You should have received a copy of the GNU General Public License along with *
* this program; if not, write to the Free Software Foundation, Inc., 51        *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                    *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_GUI_VIEWER_REALGUI_H
#define SOFA_GUI_VIEWER_REALGUI_H

#include "GUI.h"
#include "GUIField.h"


#ifdef SOFA_GUI_QTVIEWER
#include "QtViewer/QtViewer.h"
#endif

#ifdef SOFA_GUI_QGLVIEWER
#include "QtGLViewer/QtGLViewer.h"
#endif

#ifdef SOFA_GUI_QTOGREVIEWER
#include "QtOgreViewer/QtOgreViewer.h"
#endif


#ifdef QT_MODULE_QT3SUPPORT
#include <Q3ListViewItem>
#include <QStackedWidget>
typedef Q3ListViewItem QListViewItem;
typedef QStackedWidget QWidgetStack;
#else
#include <qwidgetstack.h>
#include "qlistview.h"
#endif

#ifdef SOFA_PML
#include <sofa/filemanager/sofapml/PMLReader.h>
#include <sofa/filemanager/sofapml/LMLReader.h>
using namespace sofa::filemanager::pml;
#endif

namespace sofa
{

namespace gui
{

namespace guiviewer
{

using sofa::simulation::tree::GNode;

class GraphListenerQListView;



class RealGUI : public ::GUI
{
    Q_OBJECT

public:


#ifdef SOFA_GUI_QGLVIEWER
    sofa::gui::guiqglviewer::QtGLViewer* viewer;
#elif SOFA_GUI_QTVIEWER
    sofa::gui::qt::QtViewer* viewer;
#elif SOFA_GUI_QTOGREVIEWER
    sofa::gui::qtogreviewer::QtOgreViewer* viewer;
#endif

    RealGUI( const char* filename=NULL);
    ~RealGUI();

    virtual void fileOpen(const char* filename);
    virtual void fileSaveAs(const char* filename);
    virtual void setScene(GNode* groot, const char* filename=NULL);
    virtual void setTitle( const char* windowTitle );

    //public slots:
    virtual void fileOpen();
    //virtual void fileSave();
    virtual void fileSaveAs();
    virtual void fileReload();
    //virtual void filePrint();
    virtual void fileExit();
    virtual void saveXML();
    //virtual void editUndo();
    //virtual void editRedo();
    //virtual void editCut();
    //virtual void editCopy();
    //virtual void editPaste();
    //virtual void editFind();
    //virtual void helpIndex();
    //virtual void helpContents();
    //virtual void helpAbout();

public slots:

    void DoubleClickeItemInSceneView(QListViewItem * item);
    void playpauseGUI(bool value);
    void step();
    void animate();
    void setDt(double);
    void setDt(const QString&);
    void resetScene();
    void slot_showVisual(bool);
    void slot_showBehavior(bool);
    void slot_showCollision(bool);
    void slot_showBoundingCollision(bool);
    void slot_showMapping(bool);
    void slot_showMechanicalMapping(bool);
    void slot_showForceField(bool);
    void slot_showInteractionForceField(bool);
    void slot_showWireFrame(bool);
    void slot_showNormals(bool);
    void exportGraph();
    void exportGraph(sofa::simulation::tree::GNode*);
    void exportOBJ(bool exportMTL=true);
    void dumpState(bool);
    void displayComputationTime(bool);
    void setExportGnuplot(bool);

signals:
    void reload();
    void newFPS(const QString&);
    void newFPS(double);
    void newTime(const QString&);
    void newTime(double);


protected:

    void eventNewStep();
    void eventNewTime();
    void init();
    void keyPressEvent ( QKeyEvent * e );

#ifdef WIN32
    void resizeEvent(QResizeEvent * event );
#endif
    void timerEvent(QTimerEvent *event)
    {
        if (viewer != NULL) viewer->update();
    }

    bool m_dumpState;
    std::ofstream* m_dumpStateStream;
    bool m_exportGnuplot;
    bool _animationOBJ; int _animationOBJcounter;// save a succession of .obj indexed by _animationOBJcounter
    bool m_displayComputationTime;


    GraphListenerQListView* graphListener;
    QTimer* timerStep;
    QLabel* fpsLabel;
    QLabel* timeLabel;
    QWidgetStack* left_stack;

    sofa::simulation::tree::GNode* groot;
    std::string sceneFileName;

private:
    std::map< core::objectmodel::Base*, QWidget* > _alreadyOpen;
    void addViewer(const char* filename=NULL);
    void setGUI(void);
    int id_timer;
#ifdef SOFA_PML
    virtual void pmlOpen(const char* filename, bool resetView=true);
    virtual void lmlOpen(const char* filename);
    PMLReader *pmlreader;
    LMLReader *lmlreader;
#endif
};








} // namespace qt

} // namespace gui

} // namespace sofa

#endif // SOFA_GUI_VIEWER_REALGUI_H
