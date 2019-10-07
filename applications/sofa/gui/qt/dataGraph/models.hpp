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
class MyNodeData : public NodeData
{
public:

  NodeDataType
  type() const override
  {
    return NodeDataType {"MyNodeData",
                         "My Node Data"};
  }
};

class SimpleNodeData : public NodeData
{
public:

  NodeDataType
  type() const override
  {
    return NodeDataType {"SimpleData",
                         "Simple Data"};
  }
};

//------------------------------------------------------------------------------

class DefaultObjectModel : public NodeDataModel
{
    Q_OBJECT

public:
    DefaultObjectModel()
        : NodeDataModel()
        , m_nbrData(2)
    {}

    DefaultObjectModel(std::vector < std::pair < std::string, std::string> > _data)
        : NodeDataModel()
        , m_data(_data)
    {
        m_nbrData = m_data.size();
    }

    virtual ~DefaultObjectModel() {}

    QString caption() const override
    {
        return m_caption;
    }

    void setCaption(std::string str)
    {
        m_caption = QString::fromStdString(str);
    }

    QString name() const override
    {
        return QString("DefaultObjectModel");
    }

    void setNbrData(unsigned int nbr)
    {
        m_nbrData = nbr;
    }

    QtNodes::PortIndex getDataInputId(const std::string& dataName)
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

public:

  unsigned int
  nPorts(PortType portType) const override
  {
      return m_nbrData;
  }

  NodeDataType
  dataType(PortType portType,
           PortIndex portIndex) const override
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
          
      return MyNodeData().type();
  }

  std::shared_ptr<NodeData>
  outData(PortIndex port) override
  {
    
      return std::make_shared<MyNodeData>();
  }

  void
  setInData(std::shared_ptr<NodeData>, int) override
  {
    //
  }

  QWidget *
  embeddedWidget() override { return nullptr; }

protected: 
    QString m_caption;
    unsigned int m_nbrData;
    std::vector < std::pair < std::string, std::string> > m_data;
};
