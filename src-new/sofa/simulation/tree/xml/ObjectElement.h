#ifndef SOFA_SIMULATION_TREE_XML_OBJECTELEMENT_H
#define SOFA_SIMULATION_TREE_XML_OBJECTELEMENT_H

#include <sofa/simulation/tree/xml/Element.h>
#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa
{

namespace simulation
{

namespace tree
{

namespace xml
{

class ObjectElement : public Element<core::objectmodel::BaseObject>
{
public:
    ObjectElement(const std::string& name, const std::string& type, BaseElement* parent=NULL);

    virtual ~ObjectElement();

    virtual bool initNode();

    virtual const char* getClass() const;
};

} // namespace xml

} // namespace tree

} // namespace simulation

} // namespace sofa

#endif
