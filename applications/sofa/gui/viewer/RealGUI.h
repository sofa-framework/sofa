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

#ifdef SOFA_PML
#  include <sofa/filemanager/sofapml/PMLReader.h>
#  include <sofa/filemanager/sofapml/LMLReader.h>
#endif


#include <sofa/gui/SofaGUI.h>

#include <GUI.h>
#include <GraphListenerQListView.h>
#include <SofaViewer.h>
#include <AddObject.h>
#include <ModifyObject.h>
#include <sofa/simulation/tree/xml/XML.h>

#ifdef QT_MODULE_QT3SUPPORT
#include <Q3ListViewItem>
#include <QStackedWidget>
typedef Q3ListViewItem QListViewItem;
typedef QStackedWidget QWidgetStack;
#else
#include <qwidgetstack.h>
#include "qlistview.h"
#endif


namespace sofa
{

namespace gui
{

namespace guiviewer
{

//enum TYPE{ NORMAL, PML, LML};

using sofa::simulation::tree::GNode;
#ifdef SOFA_PML
using namespace sofa::filemanager::pml;
#endif

class RealGUI : public ::GUI, public SofaGUI
{
    Q_OBJECT

    /// @name SofaGUI Interface
    /// @{

public:

    static int InitGUI(const char* name, const std::vector<std::string>& options);
    static SofaGUI* CreateGUI(const char* name, const std::vector<std::string>& options, sofa::simulation::tree::GNode* groot = NULL, const char* filename = NULL);

    int mainLoop();

    int closeGUI();

    sofa::simulation::tree::GNode* currentSimulation();

    /// @}

    const char* viewerName;

    sofa::gui::viewer::SofaViewer* viewer;

    RealGUI( const char* viewername, const std::vector<std::string>& options = std::vector<std::string>() );
    ~RealGUI();


    virtual void fileOpen(const char* filename); //, int TYPE=NORMAL);
    virtual void fileOpen(const char* filename, bool keepParams); //, int TYPE=NORMAL);
    virtual void fileSaveAs(const char* filename);
    virtual void setScene(GNode* groot, const char* filename=NULL, bool keepParams=false);
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
    virtual void viewerOpenGL();
    virtual void viewerQGLViewer();
    virtual void viewerOGRE();
    //virtual void helpIndex();
    //virtual void helpContents();
    //virtual void helpAbout();

public slots:

    void DoubleClickeItemInSceneView(QListViewItem * item);
    void RightClickedItemInSceneView(QListViewItem *item, const QPoint& point, int index);
    void playpauseGUI(bool value);
    void step();
    void animate();
    void setDt(double);
    void setDt(const QString&);
    void resetScene();
    void screenshot();
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
    void currentTabChanged(QWidget*);
    //Used in Context Menu
    void graphAddObject();
    void graphRemoveObject();
    void graphModify();
    void graphCollapse();
    void graphExpand();
    //When adding an object in the graph
    void loadObject(std::string path, double dx, double dy, double dz, double scale=1.0);
    //refresh the visualization window
    void redraw();
    //when a dialog modify object is closed
    void modifyUnlock(int Id);
    void transformObject( GNode *node, double dx, double dy, double dz, double scale=1.0);

signals:
    void reload();
    void newScene();
    void newStep();


protected:

    void eventNewStep();
    void eventNewTime();
    void init();
    void keyPressEvent ( QKeyEvent * e );


    GNode *searchNode(GNode *node, Q3ListViewItem *item_clicked);
    GNode *verifyNode(GNode *node, Q3ListViewItem *item_clicked);
    bool isErasable(core::objectmodel::Base* element);

    bool m_dumpState;
    std::ofstream* m_dumpStateStream;
    bool m_exportGnuplot;
    bool _animationOBJ; int _animationOBJcounter;// save a succession of .obj indexed by _animationOBJcounter
    bool m_displayComputationTime;


    QWidget* currentTab;
    GraphListenerQListView* graphListener;
    QListViewItem *item_clicked;
    GNode *node_clicked;
    QTimer* timerStep;
    QLabel* fpsLabel;
    QLabel* timeLabel;
    QWidgetStack* left_stack;
    AddObject *dialog;


    //these are already stored in the viewe
    //do not duplicate them
    //sofa::simulation::tree::GNode* groot;
    //std::string sceneFileName;
    sofa::simulation::tree::GNode* getScene() { if (viewer) return viewer->getScene(); else return NULL; }

private:
    //Map: Id -> Node currently modified. Used to avoid dependancies during removing actions
    std::map< int, core::objectmodel::Base* >       map_modifyDialogOpened;
    //unique ID to pass to a modify object dialog
    int current_Id_modifyDialog;

    //At initialization: list of the path to the basic objects you can add to the scene
    std::vector< std::string > list_object;

    std::string list_demo[3];

    //Bounding Box of each object
    std::vector< float > list_object_BoundingBox;
    //currently unused: scale is experimental
    float object_Scale[2];

    float initial_time;
    std::list< GNode *> list_object_added;
    std::list< GNode *> list_object_removed;
    std::list< GNode *> list_object_initial;
    std::list< GNode* > list_node_contactPoints;


    bool setViewer(const char* name);
    void addViewer();
    void setGUI(void);


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
