#include "Sofa/Components/Common/ObjectFactory.h"
#include "Sofa/Components/Common/Vec3Types.h"
#include "Sofa/Components/Common/RigidTypes.h"
#include "Sofa/Components/Common/LaparoscopicRigidTypes.h"
#include "Sofa/Components/MassSpringLoader.h"
#include "Sofa/Core/MechanicalObject.inl"

namespace Sofa
{

namespace Components
{

using namespace Core;
using namespace Common;

SOFA_DECL_CLASS(MechanicalObject)

template<class DataTypes>
class MechanicalObjectLoader : public MassSpringLoader
{
public:
    MechanicalObject<DataTypes>* dest;
    int index;
    MechanicalObjectLoader(MechanicalObject<DataTypes>* dest) : dest(dest), index(0) {}

    virtual void addMass(double px, double py, double pz, double vx, double vy, double vz, double /*mass*/, double /*elastic*/, bool /*fixed*/, bool /*surface*/)
    {
        dest->resize(index+1);
        DataTypes::set((*dest->getX())[index], px, py, pz);
        DataTypes::set((*dest->getV())[index], vx, vy, vz);
        ++index;
    }
};




namespace Common   // \todo Why this must be inside Common namespace
{

template<class DataTypes>
void create(MechanicalObject<DataTypes>*& obj, ObjectDescription* arg)
{
    obj = new MechanicalObject<DataTypes>();
    if (arg->getAttribute("filename"))
    {
        MechanicalObjectLoader<DataTypes> loader(obj);
        loader.load(arg->getAttribute("filename"));
    }
    if (obj!=NULL)
    {
        if (arg->getAttribute("scale")!=NULL)
            obj->applyScale(atof(arg->getAttribute("scale")));
        if (arg->getAttribute("dx")!=NULL || arg->getAttribute("dy")!=NULL || arg->getAttribute("dz")!=NULL)
            obj->applyTranslation(atof(arg->getAttribute("dx","0.0")),atof(arg->getAttribute("dy","0.0")),atof(arg->getAttribute("dz","0.0")));
    }
}
}

Creator< ObjectFactory, MechanicalObject<Vec3fTypes> > MechanicalObjectVec3fClass("MechanicalObjectVec3f",true);
Creator< ObjectFactory, MechanicalObject<Vec3dTypes> > MechanicalObjectVec3dClass("MechanicalObjectVec3d",true);
Creator< ObjectFactory, MechanicalObject<Vec3dTypes> > MechanicalObjectVec3Class("MechanicalObjectVec3",true);
Creator< ObjectFactory, MechanicalObject<Vec3dTypes> > MechanicalObjectClass("MechanicalObject",true);
Creator< ObjectFactory, MechanicalObject<RigidTypes> > MechanicalObjectRigidClass("MechanicalObjectRigid",true);
Creator< ObjectFactory, MechanicalObject<LaparoscopicRigidTypes> > MechanicalObjectLaparoscopicRigidClass("LaparoscopicObject",true);
} // namespace Components



template <>
void Core::MechanicalObject<Components::Common::Vec3dTypes>::getIndicesInSpace(std::vector<unsigned>& indices,Real xmin,Real xmax,Real ymin,Real ymax,Real zmin,Real zmax) const
{
    const VecCoord& x = *getX();
    for( unsigned i=0; i<x.size(); ++i )
    {
        if( x[i][0] >= xmin && x[i][0] <= xmax && x[i][1] >= ymin && x[i][1] <= ymax && x[i][2] >= zmin && x[i][2] <= zmax )
        {
            indices.push_back(i);
        }
    }
}
template <>
void Core::MechanicalObject<Components::Common::Vec3fTypes>::getIndicesInSpace(std::vector<unsigned>& indices,Real xmin,Real xmax,Real ymin,Real ymax,Real zmin,Real zmax) const
{
    const VecCoord& x = *getX();
    for( unsigned i=0; i<x.size(); ++i )
    {
        if( x[i][0] >= xmin && x[i][0] <= xmax && x[i][1] >= ymin && x[i][1] <= ymax && x[i][2] >= zmin && x[i][2] <= zmax )
        {
            indices.push_back(i);
        }
    }
}

// overload for rigid bodies: use the center
template<>
void Core::MechanicalObject<Components::Common::RigidTypes>::getIndicesInSpace(std::vector<unsigned>& indices,Real xmin,Real xmax,Real ymin,Real ymax,Real zmin,Real zmax) const
{
    const VecCoord& x = *getX();
    for( unsigned i=0; i<x.size(); ++i )
    {
        if( x[i].getCenter()[0] >= xmin && x[i].getCenter()[0] <= xmax && x[i].getCenter()[1] >= ymin && x[i].getCenter()[1] <= ymax && x[i].getCenter()[2] >= zmin && x[i].getCenter()[2] <= zmax )
        {
            indices.push_back(i);
        }
    }
}

// g++ 4.1 requires template instantiations to be declared on a parent namespace from the template class.

template class Core::MechanicalObject<Components::Common::Vec3dTypes>;
template class Core::MechanicalObject<Components::Common::Vec3fTypes>;
template class Core::MechanicalObject<Components::Common::RigidTypes>;
template class Core::MechanicalObject<Components::Common::LaparoscopicRigidTypes>;

} // namespace Sofa
