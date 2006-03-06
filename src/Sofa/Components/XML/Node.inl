#ifndef SOFA_COMPONENTS_XML_NODE_INL
#define SOFA_COMPONENTS_XML_NODE_INL

#include "Node.h"
#include "../Common/Factory.inl"

namespace Sofa
{

namespace Components
{

namespace XML
{

using namespace Common;

template<class Object>
bool Node<Object>::initNode()
{
    //Object *obj = Factory< std::string, Object, Node<Object>* >::getInstance()->createObject(this->getType(), this);
    Object *obj = Factory::CreateObject(this->getType(), this);
    if (obj != NULL)
    {
        setObject(obj);
        return true;
    }
    else return false;
}

//template<class Object> class Factory< std::string, Object, Node<Object>* >;


} // namespace XML

} // namespace Components

} // namespace Sofa

#endif
