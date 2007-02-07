#include "PlaneForceField.inl"
#include "Common/Vec3Types.h"
#include "Sofa-old/Core/MechanicalObject.h"
#include "Common/ObjectFactory.h"

namespace Sofa
{

namespace Components
{

SOFA_DECL_CLASS(PlaneForceField)

using namespace Common;

template class PlaneForceField<Vec3dTypes>;
template class PlaneForceField<Vec3fTypes>;

template<class DataTypes>
void create(PlaneForceField<DataTypes>*& obj, ObjectDescription* arg)
{
    XML::createWithParent< PlaneForceField<DataTypes>, Core::MechanicalModel<DataTypes> >(obj, arg);
    if (obj!=NULL)
    {
        if (arg->getAttribute("stiffness")) obj->setStiffness((typename PlaneForceField<DataTypes>::Real)atof(arg->getAttribute("stiffness")));
        if (arg->getAttribute("damping")) obj->setDamping((typename PlaneForceField<DataTypes>::Real)atof(arg->getAttribute("damping")));
        double x=0,y=0,z=0,d=0;
        if (arg->getAttribute("normal"))
            sscanf(arg->getAttribute("normal"),"%lf %lf %lf",&x,&y,&z);
        else
            z=1;
        if (arg->getAttribute("d")) d = atof(arg->getAttribute("d"));

        typename DataTypes::Deriv normal;
        DataTypes::set(normal,x,y,z);
        obj->setPlane(normal,(typename PlaneForceField<DataTypes>::Real)d);
    }
}

Creator< ObjectFactory, PlaneForceField<Vec3dTypes> > PlaneForceFieldVec3dClass("PlaneForceField", true);
Creator< ObjectFactory, PlaneForceField<Vec3fTypes> > PlaneForceFieldVec3fClass("PlaneForceField", true);

} // namespace Components

} // namespace Sofa
