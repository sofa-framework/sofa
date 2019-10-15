#include "SofaComponentNodeModel.h"

NodeDataType SofaComponentNodeData::type() const
{
    return NodeDataType{ "SofaComponentNodeData",
        "My Sofa Node Data" };
}


SofaComponentNodeModel::SofaComponentNodeModel()
    : NodeDataModel()
    , m_nbrData(0)
{}

SofaComponentNodeModel::SofaComponentNodeModel(std::vector < std::pair < std::string, std::string> > _data)
    : NodeDataModel()
    , m_data(_data)
{
    m_nbrData = m_data.size();
}

QtNodes::PortIndex SofaComponentNodeModel::getDataInputId(const std::string& dataName)
{
    int cpt = 0;
    for (auto data : m_data)
    {
        if (data.first == dataName)
            return cpt;

        cpt++;
    }

    return QtNodes::INVALID;
}


unsigned int SofaComponentNodeModel::nPorts(PortType portType) const
{
    return m_nbrData;
}

NodeDataType SofaComponentNodeModel::dataType(PortType portType, PortIndex portIndex) const
{
    if (portIndex >= 0 && portIndex < m_data.size())
    {
        QString name = QString::fromStdString(m_data[portIndex].first);
        QString type = QString::fromStdString(m_data[portIndex].second);
        NodeDataType NType;
        NType.id = type;
        NType.name = name;
        return NType;
    }

    return SofaComponentNodeData().type();
}

std::shared_ptr<NodeData> SofaComponentNodeModel::outData(PortIndex port)
{
    return std::make_shared<SofaComponentNodeData>();
}

void SofaComponentNodeModel::setInData(std::shared_ptr<NodeData>, int)
{
    //
}