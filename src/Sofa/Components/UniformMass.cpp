#include "UniformMass.inl"
#include "Sofa/Components/XML/MassNode.h"
#include "Sofa/Components/Common/Vec3Types.h"
#include "Sofa/Components/Common/RigidTypes.h"
#include "Scene.h"
#include "GL/Repere.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

template <>
void UniformMass<RigidTypes, RigidMass>::draw()
{
    if (!Scene::getInstance()->getShowBehaviorModels()) return;
    VecCoord& x = *mmodel->getX();
    for (unsigned int i=0; i<x.size(); i++)
    {
        Quat orient = x[i].getOrientation();
        RigidTypes::Vec3& center = x[i].getCenter();
        orient[3] = -orient[3];

        static GL::Axis *axis = new GL::Axis(center, orient);

        axis->update(center, orient);
        axis->draw();
    }
}

SOFA_DECL_CLASS(UniformMass)

template class UniformMass<Vec3dTypes,double>;
template class UniformMass<Vec3fTypes,float>;
template class UniformMass<RigidTypes,RigidMass>;

namespace Common   // \todo Why this must be inside Common namespace
{

template<class DataTypes, class MassType>
void create(UniformMass<DataTypes, MassType>*& obj, XML::Node<Core::Mass>* arg)
{
    XML::createWithParent< UniformMass<DataTypes, MassType>, Core::MechanicalModel<DataTypes> >(obj, arg);
    if (obj!=NULL)
    {
        if (arg->getAttribute("gravity"))
        {
            double x=0;
            double y=0;
            double z=0;
            sscanf(arg->getAttribute("gravity"),"%lf %lf %lf",&x,&y,&z);
            typename DataTypes::Deriv g;
            DataTypes::set(g,x,y,z);
            obj->setGravity(g);
        }
        if (arg->getAttribute("mass"))
        {
            obj->setMass((MassType)atof(arg->getAttribute("mass")));
        }
    }
}
}

Creator< XML::MassNode::Factory, UniformMass<Vec3dTypes,double> > UniformMass3dClass("UniformMass",true);
Creator< XML::MassNode::Factory, UniformMass<Vec3fTypes,float > > UniformMass3fClass("UniformMass",true);
Creator< XML::MassNode::Factory, UniformMass<RigidTypes,RigidMass> > UniformMassRigidClass("UniformMass",true);

} // namespace Components

} // namespace Sofa
