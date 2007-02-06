#include "CudaTypes.h"
#include "CudaMechanicalObject.inl"
#include "Sofa/Components/Common/ObjectFactory.h"
#include "Sofa/Components/MassSpringLoader.h"

namespace Sofa
{

namespace Components
{

using namespace Core;

// \todo This code is duplicated Sofa/Components/MechanicalObject.cpp

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
        obj->parseFields(arg->getAttributeMap() );
        if (arg->getAttribute("scale")!=NULL)
            obj->applyScale(atof(arg->getAttribute("scale")));
        if (arg->getAttribute("dx")!=NULL || arg->getAttribute("dy")!=NULL || arg->getAttribute("dz")!=NULL)
            obj->applyTranslation(atof(arg->getAttribute("dx","0.0")),atof(arg->getAttribute("dy","0.0")),atof(arg->getAttribute("dz","0.0")));
    }
}
} // namespace Common

} // namespace Components

namespace Contrib
{

namespace CUDA
{

using namespace Core;
using namespace Components::Common;

SOFA_DECL_CLASS(MechanicalObjectCuda)

Creator< ObjectFactory, MechanicalObject<CudaVec3fTypes> > MechanicalObjectVec3fClass("MechanicalObjectCudaVec3f",true);
Creator< ObjectFactory, MechanicalObject<CudaVec3Types> > MechanicalObjectVec3Class("MechanicalObjectCudaVec3",true);
Creator< ObjectFactory, MechanicalObject<CudaVec3Types> > MechanicalObjectClass("MechanicalObjectCuda",true);
//Creator< ObjectFactory, MechanicalObject<CudaRigidTypes> > MechanicalObjectRigidClass("MechanicalObjectCudaRigid",true);

} // namespace CUDA

} // namespace Contrib

} // namespace Sofa
