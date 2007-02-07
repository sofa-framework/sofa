#include <sofa/component/forcefield/PlaneForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/simulation/tree/xml/ObjectFactory.h>
#include <sofa/component/MechanicalObject.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

SOFA_DECL_CLASS(PlaneForceField)

using namespace sofa::defaulttype;

template class PlaneForceField<Vec3dTypes>;
template class PlaneForceField<Vec3fTypes>;

template<class DataTypes>
void create(PlaneForceField<DataTypes>*& obj, simulation::tree::xml::ObjectDescription* arg)
{
    simulation::tree::xml::createWithParent< PlaneForceField<DataTypes>, core::componentmodel::behavior::MechanicalState<DataTypes> >(obj, arg);
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

Creator< simulation::tree::xml::ObjectFactory, PlaneForceField<Vec3dTypes> > PlaneForceFieldVec3dClass("PlaneForceField", true);
Creator< simulation::tree::xml::ObjectFactory, PlaneForceField<Vec3fTypes> > PlaneForceFieldVec3fClass("PlaneForceField", true);

} // namespace forcefield

} // namespace component

} // namespace sofa
