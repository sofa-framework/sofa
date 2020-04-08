#pragma once

#define NODE_EDITOR_SHARED

#include <QtCore/QObject>

#include <nodes/NodeData>
#include <nodes/NodeDataModel>

#include <map>

using QtNodes::NodeData;
using QtNodes::NodeDataType;
using QtNodes::NodeDataModel;
using QtNodes::PortType;
using QtNodes::PortIndex;

namespace sofa
{
    namespace core
    {
        namespace objectmodel
        {
            class BaseObject;
            class BaseData;
        }
    }
}

/// The class can potentially incapsulate any user data which
/// need to be transferred within the Node Editor graph
class SofaComponentNodeData : public NodeData
{
public:
    SofaComponentNodeData();
    SofaComponentNodeData(sofa::core::objectmodel::BaseData* bData);

    NodeDataType type() const override;

    sofa::core::objectmodel::BaseData* getData();

protected:
    sofa::core::objectmodel::BaseData* m_bData;
};


static const char* ignoredData[] = { "name", "printLog", "tags", "bbox", "listening", "componentState" };

//------------------------------------------------------------------------------
/**
* This Class is a NodeDataModel specialisation to represent a Sofa component on the QtNodes graph.
* It will take a SOFA BaseObject as target and parse all Data, storing Data, Links and connections with parents components.
*/
class SofaComponentNodeModel : public NodeDataModel
{
    Q_OBJECT

public:
    /// Default empty Object constructor with 0 Data
    SofaComponentNodeModel(std::string name = "EmptyNode");

    /// constructor with a Sofa BaseObject as target
    SofaComponentNodeModel(sofa::core::objectmodel::BaseObject* _sofaObject, bool debugMode = false);

    virtual ~SofaComponentNodeModel() {}

    /// Interface for caption.
    QString caption() const override { return m_caption; }
    void setCaption(std::string str) { m_caption = QString::fromStdString(str); }

    /// Interface for name.
    QString name() const override { return m_uniqName; }

    /// Return the number of Data.
    size_t getNbrData() { return m_data.size(); }

    /// Return the number of connection with other Node components
    size_t getNbrConnections() { return m_dataConnections.size(); }

    /// return the list of connections @sa m_dataConnections
    const std::map <QString, std::pair < QString, QString> >& getDataConnections() { return m_dataConnections; }

    /// Return the PortIndex of a Data given its Name.
    QtNodes::PortIndex getDataInputId(const QString& dataName);

    ///Interface for QtNodes
    ///{
    /// Override method to return the number of ports
    unsigned int nPorts(PortType portType) const override;

    /// Override method to give the type of Data per Port
    NodeDataType dataType(PortType portType, PortIndex portIndex) const override;

    /// Override method to return the NodeData given a port
    std::shared_ptr<NodeData> outData(PortIndex port) override;

    /// Override method to set input Data
    void setInData(std::shared_ptr<NodeData> data, int port) override;

    /// Override method for more advance node gui. Not yet used.
    QWidget* embeddedWidget() override { return nullptr; }
    ///}

protected:
    /// Internal method to parse all Data of a Sofa component and create the corresponding ports
    void parseSofaObjectData();

protected:
    QString m_caption; ///< caption to be display on the Graph
    QString m_uniqName; ///< unique name to refer to this node

    bool debugNodeGraph; ///< parameter to activate graph logs. False by default.

    /// Vector of Data/port hold by this component/Node. vector of pair{DataName, DataType}
    std::vector < std::pair < QString, QString> > m_data;

    /// vector of SofaComponentNodeData class holding pointer to the Data. To replace @sa m_data when api is validated.
    std::vector < std::shared_ptr<SofaComponentNodeData> > m_Nodedata;

    /// Map to store all connection between this node and other. map.key = this data name, map.value = pair{ComponentName, DataName}
    std::map <QString, std::pair < QString, QString> > m_dataConnections;

    /// Pointer to the sofa object.
    sofa::core::objectmodel::BaseObject* m_SofaObject;
};
