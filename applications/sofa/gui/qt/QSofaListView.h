#ifndef SOFA_GUI_QT_QSOFAGRAPH_H
#define SOFA_GUI_QT_QSOFAGRAPH_H

#ifdef SOFA_QT4
#include <QWidget>
#include <Q3ListView>
#include <Q3ListViewItem>
#include <Q3Header>
#include <QPushButton>
#else
#include <qwidget.h>
#include <qlistview.h>
#include <qheader.h>
#include <qpushbutton.h>
#endif

#include <sofa/simulation/common/Node.h>
#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/core/objectmodel/BaseObject.h>

#ifndef SOFA_QT4
typedef QListView Q3ListView;
typedef QListViewItem Q3ListViewItem;

#endif
#include <map>

namespace sofa
{

namespace gui
{
namespace qt
{

class AddObject;
class GraphListenerQListView;


enum ObjectModelType { typeNode, typeObject, typeData };
typedef union ObjectModel
{
    sofa::simulation::Node* Node;
    core::objectmodel::BaseObject* Object;
    core::objectmodel::BaseData* Data;
} ObjectModel;
struct u_objectmodel
{
    ObjectModelType type;
    ObjectModel ptr;
};

enum SofaListViewAttribute
{
    SIMULATION,
    VISUAL,
    MODELER
};

class QSofaListView : public Q3ListView
{
    Q_OBJECT
public:
    QSofaListView(const SofaListViewAttribute& attribute,
            QWidget* parent=0,
            const char* name=0,
            Qt::WFlags f = 0 );
    ~QSofaListView();

    GraphListenerQListView* getListener() const { return  graphListener_; };
    void Clear(sofa::simulation::Node* rootNode);
    void Freeze();
    void Unfreeze();
public slots:
    void Export();
    void CloseAllDialogs();
    void UpdateOpenedDialogs();
signals:
    void Lock(bool);
    void RequestSaving(sofa::simulation::Node*);
    void currentActivated(bool);
    void RootNodeChanged(sofa::simulation::Node* newroot, const char* newpath);
    void NodeRemoved();
    void Updated();
    void NodeAdded();

protected slots:
    void updateMatchingObjectmodel(Q3ListViewItem* item);
    void SaveNode();
    void collapseNode();
    void expandNode();
    void modifyUnlock(void* Id);
    void RaiseAddObject();
    void RemoveNode();
    void Modify();
    void HideDatas();
    void ShowDatas();
    void DesactivateNode();
    void ActivateNode();
    void loadObject ( std::string path, double dx, double dy, double dz,  double rx, double ry, double rz,double scale );
    void RunSofaRightClicked( Q3ListViewItem *item, const QPoint& point, int index );
    void RunSofaDoubleClicked( Q3ListViewItem*);
protected:
    void collapseNode(Q3ListViewItem* item);
    void expandNode(Q3ListViewItem* item);
    void transformObject ( sofa::simulation::Node *node, double dx, double dy, double dz,  double rx, double ry, double rz, double scale );
    bool isNodeErasable( core::objectmodel::BaseNode* node);
    void graphActivation(bool activate);
    void updateMatchingObjectmodel();
    std::map< void*, Q3ListViewItem* > map_modifyDialogOpened;
    std::map< void*, QDialog* > map_modifyObjectWindow;
    GraphListenerQListView* graphListener_;
    std::vector< std::string > list_object;
    AddObject* AddObjectDialog_;
    u_objectmodel object_;
    SofaListViewAttribute attribute_;

};

} //sofa
} //gui
}//qt

#endif // SOFA_GUI_QT_QSOFAGRAPH_H


