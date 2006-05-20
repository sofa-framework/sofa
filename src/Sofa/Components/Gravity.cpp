#include "Sofa/Components/Gravity.h"
#include "Sofa/Components/Common/Vec3Types.h"
#include <Sofa/Components/Graph/GNode.h>

#include <math.h>

namespace Sofa
{

namespace Components
{


using namespace Common;
using namespace Core;


Gravity::Gravity():Abstract::ContextObject()
{
}

const Gravity::Vec3&  Gravity::getGravity() const
{
    return gravity_;
}

Gravity* Gravity::setGravity( const Vec3& g )
{
    gravity_=g;
    return this;
}

void Gravity::apply()
{
    getContext()->setGravity( gravity_ );
}
//         void create(Gravity*& obj, XML::Node<Components::Graph::Property>* /*arg*/)
//         {
//             // TODO: read the parameters before
//             obj = new Gravity("gravity",0);
//         }
//
//         SOFA_DECL_CLASS(Gravity)
//
//                 Creator<XML::PropertyNode::Factory, Gravity> GravityClass("Gravity");

} // namespace Components

} // namespace Sofa

