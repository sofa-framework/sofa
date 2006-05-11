#include "Sofa/Components/Gravity.h"
#include "XML/PropertyNode.h"

#include <math.h>

namespace Sofa
{

namespace Components
{

using namespace Common;
using namespace Core;


void create(Gravity*& obj, XML::Node<Core::Property>* /*arg*/)
{
    obj = new Gravity();
}

SOFA_DECL_CLASS(Gravity)

Creator<XML::PropertyNode::Factory, Gravity> GravityClass("Gravity");

} // namespace Components

} // namespace Sofa


