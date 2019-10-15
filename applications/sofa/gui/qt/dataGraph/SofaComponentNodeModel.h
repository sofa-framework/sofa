#pragma once

#define NODE_EDITOR_SHARED

#include <QtCore/QObject>

#include <nodes/NodeData>
#include <nodes/NodeDataModel>
#include <iostream>
#include <memory>
#include <climits>
#include <map>

using QtNodes::NodeData;
using QtNodes::NodeDataType;
using QtNodes::NodeDataModel;
using QtNodes::PortType;
using QtNodes::PortIndex;

/// The class can potentially incapsulate any user data which
/// need to be transferred within the Node Editor graph
class SofaComponentNodeData : public NodeData
{
public:
    NodeDataType type() const override;
  
};


static const char* ignoredData[] = { "name", "printLog", "tags", "bbox", "listening", "componentState" };

namespace sofa
{
namespace core
{
namespace objectmodel
{
    class BaseObject;
}
}
}

//------------------------------------------------------------------------------

class SofaComponentNodeModel : public NodeDataModel
{
    Q_OBJECT

public:
    SofaComponentNodeModel();
            
    SofaComponentNodeModel(sofa::core::objectmodel::BaseObject* _sofaObject);

    virtual ~SofaComponentNodeModel() {}

    /// Interface for caption
    QString caption() const override {return m_caption;}
    void setCaption(std::string str) {m_caption = QString::fromStdString(str);}

    /// Interface for name
    QString name() const override { return m_uniqName; }


    size_t getNbrData() {return m_data.size();}

    QtNodes::PortIndex getDataInputId(const QString& dataName);

    void parseSofaObjectData();

    size_t getNbrConnections() { return m_dataConnections.size(); }
    const std::map <QString, std::pair < QString, QString> >& getDataConnections() {
        return m_dataConnections;
    }

public:
    unsigned int nPorts(PortType portType) const override;

    NodeDataType dataType(PortType portType, PortIndex portIndex) const override;

    std::shared_ptr<NodeData> outData(PortIndex port) override;
  
    void setInData(std::shared_ptr<NodeData>, int) override;

    QWidget * embeddedWidget() override { return nullptr; }

protected: 
    QString m_caption;
    QString m_uniqName;

    bool debugNodeGraph;
    std::vector < std::pair < QString, QString> > m_data;
    std::map <QString, std::pair < QString, QString> > m_dataConnections;

    sofa::core::objectmodel::BaseObject* m_SofaObject;
};
