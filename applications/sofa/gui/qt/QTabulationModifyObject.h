#ifndef SOFA_GUI_QT_QTABULATIONMODIFYOBJECT_H
#define SOFA_GUI_QT_QTABULATIONMODIFYOBJECT_H

#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/core/objectmodel/BaseLink.h>
#include <sofa/simulation/common/Node.h>

#ifdef SOFA_QT4
#include <QWidget>
#include <QTextEdit>
#include <Q3GroupBox>
#include <Q3ListViewItem>
#include <Q3ListView>
#else
#include <qwidget.h>
#include <qtextedit.h>
#include <qgroupbox.h>
#include <qlistview.h>
#endif

#ifndef SOFA_QT4
typedef QGroupBox Q3GroupBox;
typedef QTextEdit   Q3TextEdit;
typedef QListView   Q3ListView;
typedef QListViewItem Q3ListViewItem;
#endif

namespace sofa
{
namespace gui
{
namespace qt
{

struct ModifyObjectFlags;
class QTabulationModifyObject : public QWidget
{
    Q_OBJECT
public:
    QTabulationModifyObject(QWidget* parent,
            core::objectmodel::Base *object, Q3ListViewItem* item,
            unsigned int idx=1);

    void externalWidgetAddition(int num) {size+=num;}
    void addData(sofa::core::objectmodel::BaseData *data, const ModifyObjectFlags& flags);
    void addLink(sofa::core::objectmodel::BaseLink *link, const ModifyObjectFlags& flags);
    void addStretch();

    unsigned int getIndex() const {return index;};
    bool isFull() const;
    void setFull() {size+=maxSize;};
    bool isEmpty() const;
    bool isDirty() const;

public slots:
    void setTabDirty(bool=true);
    void updateListViewItem();
    void updateDataValue();
    void updateWidgetValue();

signals:
    void UpdateDatas();
    void UpdateDataWidgets();
    void TabDirty(bool);
    void nodeNameModification(simulation::Node *);
protected:
    core::objectmodel::Base *object;
    Q3ListViewItem* item;


    const unsigned int index;
    unsigned int size;
    const unsigned int maxSize;

    bool dirty;
};


} // qt
} // gui
} //sofa

#endif // SOFA_GUI_QT_QTABULATIONMODIFYOBJECT_H

