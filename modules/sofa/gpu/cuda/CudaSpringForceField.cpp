#include "CudaTypes.h"
#include "Sofa-old/Components/Common/ObjectFactory.h"
#include "CudaSpringForceField.inl"

namespace Sofa
{

namespace Components
{

// \todo This code is duplicated Sofa/Components/*SpringForceField.cpp

namespace Common   // \todo Why this must be inside Common namespace
{
template<class DataTypes>
void create(SpringForceField<DataTypes>*& obj, ObjectDescription* arg)
{
    XML::createWithParent< SpringForceField<DataTypes>, Core::MechanicalModel<DataTypes> >(obj, arg);
    if (obj == NULL) // try the InteractionForceField initialization
        XML::createWith2Objects< SpringForceField<DataTypes>, Core::MechanicalModel<DataTypes>, Core::MechanicalModel<DataTypes> >(obj, arg);
    if (obj != NULL)
    {
        if (arg->getAttribute("filename"))
            obj->load(arg->getAttribute("filename"));
    }
}

template<class DataTypes>
void create(StiffSpringForceField<DataTypes>*& obj, ObjectDescription* arg)
{
    XML::createWithParent< StiffSpringForceField<DataTypes>, Core::MechanicalModel<DataTypes> >(obj, arg);
    if (obj == NULL) // try the InteractionForceField initialization
        XML::createWith2Objects< StiffSpringForceField<DataTypes>, Core::MechanicalModel<DataTypes>, Core::MechanicalModel<DataTypes> >(obj, arg);
    if (obj != NULL)
    {
        if (arg->getAttribute("filename"))
            obj->load(arg->getAttribute("filename"));
    }
}

template<class DataTypes>
void create(MeshSpringForceField<DataTypes>*& obj, ObjectDescription* arg)
{
    XML::createWithParent< MeshSpringForceField<DataTypes>, Core::MechanicalModel<DataTypes> >(obj, arg);
    if (obj!=NULL)
    {
        if (arg->getAttribute("stiffness"))          obj->setStiffness         ((typename DataTypes::Coord::value_type)atof(arg->getAttribute("stiffness")));
        if (arg->getAttribute("linesStiffness"))     obj->setLinesStiffness    ((typename DataTypes::Coord::value_type)atof(arg->getAttribute("linesStiffness")));
        if (arg->getAttribute("trianglesStiffness")) obj->setTrianglesStiffness((typename DataTypes::Coord::value_type)atof(arg->getAttribute("trianglesStiffness")));
        if (arg->getAttribute("quadsStiffness"))     obj->setQuadsStiffness    ((typename DataTypes::Coord::value_type)atof(arg->getAttribute("quadsStiffness")));
        if (arg->getAttribute("tetrasStiffness"))    obj->setTetrasStiffness   ((typename DataTypes::Coord::value_type)atof(arg->getAttribute("tetrasStiffness")));
        if (arg->getAttribute("cubesStiffness"))     obj->setCubesStiffness    ((typename DataTypes::Coord::value_type)atof(arg->getAttribute("cubesStiffness")));
        if (arg->getAttribute("damping"))            obj->setDamping           ((typename DataTypes::Coord::value_type)atof(arg->getAttribute("damping")));
        if (arg->getAttribute("linesDamping"))       obj->setLinesDamping      ((typename DataTypes::Coord::value_type)atof(arg->getAttribute("linesDamping")));
        if (arg->getAttribute("trianglesDamping"))   obj->setTrianglesDamping  ((typename DataTypes::Coord::value_type)atof(arg->getAttribute("trianglesDamping")));
        if (arg->getAttribute("quadsDamping"))       obj->setQuadsDamping      ((typename DataTypes::Coord::value_type)atof(arg->getAttribute("quadsDamping")));
        if (arg->getAttribute("tetrasDamping"))      obj->setTetrasDamping     ((typename DataTypes::Coord::value_type)atof(arg->getAttribute("tetrasDamping")));
        if (arg->getAttribute("cubesDamping"))       obj->setCubesDamping      ((typename DataTypes::Coord::value_type)atof(arg->getAttribute("cubesDamping")));
    }
}
}
}

namespace Contrib
{

namespace CUDA
{
using namespace Components::Common;
using namespace Components;

SOFA_DECL_CLASS(SpringForceFieldCuda)

Creator< ObjectFactory, SpringForceField<CudaVec3fTypes> > SpringForceFieldCuda3fClass("SpringForceField",true);
Creator< ObjectFactory, StiffSpringForceField<CudaVec3fTypes> > StiffSpringForceFieldCuda3fClass("StiffSpringForceField",true);
Creator< ObjectFactory, MeshSpringForceField<CudaVec3fTypes> > MeshSpringForceFieldCuda3fClass("MeshSpringForceField",true);

} // namespace CUDA

} // namespace Contrib

} // namespace Sofa
