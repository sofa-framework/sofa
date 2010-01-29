#ifndef SOFA_GUI_QT_QSOFASTATGRAPH_H
#define SOFA_GUI_QT_QSOFASTATGRAPH_H
#include <sofa/helper/vector.h>

#ifdef SOFA_QT4
#include <QLabel>
#include <QWidget>
#include <Q3ListView>
#include <Q3ListViewItem>
#include <Q3Header>
#else
#include <qlabel.h>
#include <qwidget.h>
#include <qlistview.h>
#include <qheader.h>
#endif

#ifndef SOFA_QT4
typedef QListView Q3ListView;
typedef QListViewItem Q3ListViewItem;
#endif

namespace sofa
{
namespace core
{
class CollisionModel;
namespace objectmodel
{
class Base;
}
}
namespace simulation
{
class Node;
}

namespace gui
{
namespace qt
{
class QSofaStatWidget : public QWidget
{
    Q_OBJECT
public:
    QSofaStatWidget(QWidget* parent);
    void CreateStats(sofa::simulation::Node* root);
protected:
    QLabel* statsLabel;
    Q3ListView* statsCounter;
    void addSummary();
    void addCollisionModelsStat(const sofa::helper::vector< sofa::core::CollisionModel* >& v);
    std::vector<std::pair<core::objectmodel::Base*, Q3ListViewItem*> > items_stats;

};
} //qt
} //gui
} //sofa

#endif //SOFA_GUI_QT_QSOFASTATGRAPH_H
