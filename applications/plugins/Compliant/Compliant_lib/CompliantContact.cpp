

#include "CompliantContact.inl"
#include <sofa/component/collision/BarycentricContactMapper.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace defaulttype;
using namespace sofa::helper;
using simulation::Node;
using namespace sofa::defaulttype;
using namespace core::collision;
using simulation::Node;

// TODO figure out what is this ?
sofa::core::collision::DetectionOutput::ContactId Identifier::cpt=0;
std::list<sofa::core::collision::DetectionOutput::ContactId> Identifier::availableId;

SOFA_DECL_CLASS(CompliantContact)

Creator<Contact::Factory, CompliantContact<PointModel, PointModel> > PointPointCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<LineModel, SphereModel> > LineSphereCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<LineModel, PointModel> > LinePointCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<LineModel, LineModel> > LineLineCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<TriangleModel, SphereModel> > TriangleSphereCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<TriangleModel, PointModel> > TrianglePointCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<TriangleModel, LineModel> > TriangleLineCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<TriangleModel, TriangleModel> > TriangleTriangleCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<SphereModel, SphereModel> > SphereSphereCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<SphereModel, PointModel> > SpherePointCompliantContactClass("CompliantContact",true);

} // namespace collision

} // namespace component

} // namespace sofa
