#ifndef SOFA_SIMULATION_TREE_XML_ELEMENT_INL
#define SOFA_SIMULATION_TREE_XML_ELEMENT_INL

#include "Element.h"

namespace sofa
{

namespace simulation
{

namespace tree
{

namespace xml
{

template<class Object>
bool Element<Object>::initNode()
{
    //Object *obj = Factory< std::string, Object, Node<Object>* >::getInstance()->createObject(this->getType(), this);
    Object *obj = Factory::CreateObject(this->getType(), this);
    if (obj != NULL)
    {
        setObject(obj);
        obj->setName(getName());
        return true;
    }
    else return false;
}

//template<class Object> class Factory< std::string, Object, Node<Object>* >;


} // namespace xml

} // namespace tree

} // namespace simulation

} // namespace sofa

#endif
