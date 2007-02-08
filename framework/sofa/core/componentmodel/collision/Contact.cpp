#include <sofa/core/componentmodel/collision/Contact.h>
#include <sofa/helper/Factory.inl>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace collision
{

using namespace sofa::defaulttype;

//template class Factory< std::string, Contact, std::pair<core::CollisionModel*,core::CollisionModel*> >;

Contact* Contact::Create(const std::string& type, core::CollisionModel* model1, core::CollisionModel* model2, Intersection* intersectionMethod)
{
    return Factory::CreateObject(type,std::make_pair(std::make_pair(model1,model2),intersectionMethod));
}

} // namespace collision

} // namespace componentmodel

} // namespace core

} // namespace sofa

