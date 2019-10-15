#pragma once

#define NODE_EDITOR_SHARED

#include <QtCore/QObject>

#include <nodes/NodeData>
#include <nodes/NodeDataModel>
#include <iostream>
#include <memory>
#include <climits>

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
        
    SofaComponentNodeModel(std::vector < std::pair < std::string, std::string> > _data);
    
    virtual ~SofaComponentNodeModel() {}

    /// Interface for caption
    QString caption() const override {return m_caption;}
    void setCaption(std::string str) {m_caption = QString::fromStdString(str);}

    /// Interface for name
    QString name() const override { return m_uniqName; }


    void setNbrData(unsigned int nbr) {m_nbrData = nbr;}

    QtNodes::PortIndex getDataInputId(const std::string& dataName);

    void parseSofaObjectData();
    

public:
    unsigned int nPorts(PortType portType) const override;

    NodeDataType dataType(PortType portType, PortIndex portIndex) const override;

    std::shared_ptr<NodeData> outData(PortIndex port) override;
  
    void setInData(std::shared_ptr<NodeData>, int) override;

    QWidget * embeddedWidget() override { return nullptr; }

protected: 
    QString m_caption;
    QString m_uniqName;
    unsigned int m_nbrData;
    std::vector < std::pair < std::string, std::string> > m_data;
};
