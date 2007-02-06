#ifndef SOFA_COMPONENTS_XML_OBJECTNODE_H
#define SOFA_COMPONENTS_XML_OBJECTNODE_H

#include "Node.h"
#include "Sofa-old/Abstract/BaseObject.h"

namespace Sofa
{

namespace Components
{

namespace XML
{

class ObjectNode : public Node<Abstract::BaseObject>
{
public:
    ObjectNode(const std::string& name, const std::string& type, BaseNode* parent=NULL);

    virtual ~ObjectNode();

    virtual bool initNode();

    virtual const char* getClass() const;
};

} // namespace XML

} // namespace Components

} // namespace Sofa

#endif
