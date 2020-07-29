#include "SofaComponentNodeModel.h"
#include <sofa/core/objectmodel/BaseObject.h>

SofaComponentNodeData::SofaComponentNodeData()
    : m_bData(nullptr)
{

}

SofaComponentNodeData::SofaComponentNodeData(sofa::core::objectmodel::BaseData* bData)
    : m_bData(bData)
{

}

sofa::core::objectmodel::BaseData* SofaComponentNodeData::getData()
{
    //m_bData->updateIfDirty(); ??
    return m_bData;
}

NodeDataType SofaComponentNodeData::type() const
{
    return NodeDataType{ "SofaComponentNodeData",
        "My Sofa Node Data" };
}


SofaComponentNodeModel::SofaComponentNodeModel(std::string name)
    : NodeDataModel()
    , debugNodeGraph(true)
    , m_SofaObject(nullptr)
{
    m_uniqName = QString::fromStdString(name);
    m_caption = m_uniqName;
}

SofaComponentNodeModel::SofaComponentNodeModel(sofa::core::objectmodel::BaseObject* _sofaObject, bool debugMode)
    : NodeDataModel()
    , debugNodeGraph(debugMode)
    , m_SofaObject(_sofaObject)
{
    if (m_SofaObject == nullptr)
    {
        msg_error("SofaComponentNodeModel") << "Sofa BaseObject is null, Node graph initialisation failed.";
        m_uniqName = "ErrorNode";
    }
    else
    {
        parseSofaObjectData();
    }
}


void SofaComponentNodeModel::parseSofaObjectData()
{
    if (m_SofaObject == nullptr)
    {
        msg_error("SofaComponentNodeModel") << "Sofa BaseObject is null, Node graph parseSofaObjectData failed.";
        return;
    }

    // first add this object name as first Data (to be used for the links representation)
    QString SObjectName = QString::fromStdString(m_SofaObject->getName());
    m_data.push_back(std::pair<QString, QString>(SObjectName, "name"));

    // parse links
    const sofa::core::objectmodel::Base::VecLink& links = m_SofaObject->getLinks();
    for (auto link : links)
    {
        const std::string& name = link->getName();
        // ignore unnamed link
        if (link->getName().empty())
            continue;

        // ignore link to context
        if (link->getName() == "context")
            continue;

        if (!link->storePath() && 0 == link->getSize())
            continue;

        const std::string valuetype = link->getValueTypeString();

        msg_info_when(debugNodeGraph, "SofaComponentNodeModel") << "## link: " << name << " | link->getSize(): " << link->getSize() << " | valuetype: " << valuetype << " | path: " << link->storePath();

        std::string linkPath = link->getLinkedPath();
        linkPath.erase(0, 1); // remove @
        std::size_t found = linkPath.find_last_of("/");
        if (found != std::string::npos) // remove path
            linkPath.erase(0, found);

        msg_info_when(debugNodeGraph, "SofaComponentNodeModel") << "  # baselink: " << linkPath;
        QString parentObject = QString::fromStdString(linkPath);
        m_dataConnections[SObjectName] = std::pair<QString, QString>(parentObject, parentObject);
    }

    // parse all Data
    sofa::helper::vector<sofa::core::objectmodel::BaseData*> allData = m_SofaObject->getDataFields();
    for (auto data : allData)
    {
        QString name = QString::fromStdString(data->getName());
        QString group = QString::fromStdString(data->getGroup());

        bool toBeIgnored = false;
        for (auto idata : ignoredData)
        {
            if (name.compare(idata) == 0)
            {
                toBeIgnored = true;
                break;
            }
        }

        if (group == "Visualization")
            toBeIgnored = true;


        if (toBeIgnored)
            continue;

        sofa::core::objectmodel::BaseData* pData = data->getParent();
        if (pData)
        {
            QString parentObject = QString::fromStdString(pData->getOwner()->getName());
            QString parentData = QString::fromStdString(pData->getName());
            msg_info_when(debugNodeGraph, "SofaComponentNodeModel") << "- Parent: " << pData->getName() << " owwner: " << pData->getOwner()->getName();
            m_dataConnections[name] = std::pair<QString, QString>(parentObject, parentData);
        }

        if (!group.isEmpty())
        {
            msg_info_when(debugNodeGraph, "SofaComponentNodeModel") << name.toStdString() << " -> " << data->getGroup();
        }
        m_data.push_back(std::pair<QString, QString>(name, QString::fromStdString(data->getValueTypeString())));
        m_Nodedata.push_back(std::make_shared<SofaComponentNodeData>(data));
    }
}


QtNodes::PortIndex SofaComponentNodeModel::getDataInputId(const QString& dataName)
{
    int cpt = 0;
    for (auto data : m_data)
    {
        if (data.first.compare(dataName) == 0)
            return cpt;

        cpt++;
    }

    return QtNodes::INVALID;
}


unsigned int SofaComponentNodeModel::nPorts(PortType portType) const
{
    return m_data.size();
}

NodeDataType SofaComponentNodeModel::dataType(PortType portType, PortIndex portIndex) const
{
    if (portIndex >= 0 && portIndex < m_data.size())
    {
        NodeDataType NType;
        NType.id = m_data[portIndex].second;
        NType.name = m_data[portIndex].first;
        return NType;
    }

    return SofaComponentNodeData().type();
}

std::shared_ptr<NodeData> SofaComponentNodeModel::outData(PortIndex port)
{
    // because the first port is the name of the component not stored in m_Nodedata:
    port--;

    if (port > 0 && port < m_Nodedata.size())
        return m_Nodedata[port];
    else {
        msg_warning(m_caption.toStdString()) << "Method SofaComponentNodeModel::outData port: " << port << " out of bounds: " << m_Nodedata.size();
        return nullptr;
    }
}

void SofaComponentNodeModel::setInData(std::shared_ptr<NodeData> data, int port)
{
    auto parentNodeData = std::dynamic_pointer_cast<SofaComponentNodeData>(data);
    if (parentNodeData == nullptr)
    {
        msg_warning(m_caption.toStdString()) << "Method SofaComponentNodeModel::setInData SofaComponentNodeData cast failed.";
        return;
    }

    // because the first port is the name of the component not stored in m_Nodedata:
    port--;

    if (port < 0 || port >= m_Nodedata.size())
    {
        msg_warning(m_caption.toStdString()) << "Method SofaComponentNodeModel::setInData port: " << port << " out of bounds: " << m_Nodedata.size();
        return;
    }

    // here you will implement the Data setParent in SOFA!
    std::shared_ptr<SofaComponentNodeData> childNodeData = this->m_Nodedata[port];
    sofa::core::objectmodel::BaseData* childData = childNodeData->getData();
    sofa::core::objectmodel::BaseData* parentData = parentNodeData->getData();

    msg_info_when(debugNodeGraph, m_caption.toStdString()) << "Here connect: {" << parentData->getOwner()->getName() << ", " << parentData->getName() << "} -> {"
        << childData->getOwner()->getName() << ", " << childData->getName() << "}";

}