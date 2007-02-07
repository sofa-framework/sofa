#ifndef SOFA_SIMULATION_TREE_XML_NODEELEMENT_H
#define SOFA_SIMULATION_TREE_XML_NODEELEMENT_H

#include <sofa/simulation/tree/xml/Element.h>
#include <sofa/simulation/tree/xml/BaseElement.h>
#include <sofa/core/componentmodel/behavior/ForceField.h>

namespace sofa
{

namespace simulation
{

namespace tree
{

namespace xml
{

class NodeElement : public Node<xml::BaseElement>
{
public:
    NodeNode(const std::string& name, const std::string& type, BaseElement* parent=NULL);

    virtual ~NodeNode();

    virtual bool setParent(BaseElement* newParent);

    virtual bool initNode();

    virtual bool init();

    virtual const char* getClass() const;
};

} // namespace xml

} // namespace tree

} // namespace simulation

} // namespace sofa

#endif
