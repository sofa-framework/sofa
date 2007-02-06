#include "MeshSpringForceField.inl"
#include "Common/Vec3Types.h"
#include "Common/ObjectFactory.h"

namespace Sofa
{

namespace Components
{

SOFA_DECL_CLASS(MeshSpringForceField)

using namespace Common;

template class MeshSpringForceField<Vec3dTypes>;
template class MeshSpringForceField<Vec3fTypes>;

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

Creator< ObjectFactory, MeshSpringForceField<Vec3dTypes> > MeshSpringForceFieldVec3dClass("MeshSpringForceField", true);
Creator< ObjectFactory, MeshSpringForceField<Vec3fTypes> > MeshSpringForceFieldVec3fClass("MeshSpringForceField", true);

} // namespace Components

} // namespace Sofa
