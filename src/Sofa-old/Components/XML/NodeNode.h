#ifndef SOFA_COMPONENTS_XML_NODENODE_H
#define SOFA_COMPONENTS_XML_NODENODE_H

#include "Node.h"
#include <Sofa-old/Abstract/BaseNode.h>
#include "Sofa-old/Core/ForceField.h"

namespace Sofa
{

namespace Components
{

namespace XML
{

class NodeNode : public Node<Abstract::BaseNode>
{
public:
    NodeNode(const std::string& name, const std::string& type, BaseNode* parent=NULL);

    virtual ~NodeNode();

    virtual bool setParent(BaseNode* newParent);

    virtual bool initNode();

    virtual bool init();

    virtual const char* getClass() const;
};

} // namespace XML

} // namespace Components

} // namespace Sofa

#endif
