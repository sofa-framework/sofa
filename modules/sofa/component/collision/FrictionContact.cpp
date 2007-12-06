#include <sofa/component/collision/FrictionContact.inl>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace defaulttype;
using namespace sofa::helper;
using simulation::tree::GNode;

unsigned int Identifier::cpt=0;
std::list<unsigned int> Identifier::availableId;

SOFA_DECL_CLASS(FrictionContact)

Creator<Contact::Factory, FrictionContact<PointModel, PointModel> > PointPointFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<LineModel, SphereModel> > LineSphereFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<LineModel, PointModel> > LinePointFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<LineModel, LineModel> > LineLineFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TriangleMeshModel, SphereModel> > TriangleMeshSphereFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TriangleMeshModel, PointModel> > TriangleMeshPointFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TriangleMeshModel, LineModel> > TriangleMeshLineFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TriangleMeshModel, TriangleMeshModel> > TriangleMeshTriangleMeshFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TriangleMeshModel, TriangleSetModel> > TriangleMeshTriangleSetFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TriangleSetModel, SphereModel> > TriangleSetSphereFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TriangleSetModel, PointModel> > TriangleSetPointFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TriangleSetModel, LineModel> > TriangleSetLineFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TriangleSetModel, TriangleMeshModel> > TriangleSetTriangleMeshFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TriangleSetModel, TriangleSetModel> > TriangleSetTriangleSetFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<SphereModel, SphereModel> > SphereSphereFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<SphereModel, PointModel> > SpherePointFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<SphereTreeModel, SphereTreeModel> > SphereTreeSphereTreeFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<SphereTreeModel, TriangleSetModel> > SphereTreeTriangleSetFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<SphereTreeModel, TriangleMeshModel> > SphereTreeTriangleMeshFrictionContactClass("FrictionContact", true);


} // namespace collision

} // namespace component

} // namespace sofa
