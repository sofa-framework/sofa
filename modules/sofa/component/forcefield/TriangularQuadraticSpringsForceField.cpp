#include <sofa/component/forcefield/TriangularQuadraticSpringsForceField.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/topology/MeshTopology.h>
#include <fstream> // for reading the file
#include <iostream> //for debugging
#include <vector>
#include <sofa/defaulttype/Vec3Types.h>



// #define DEBUG_TRIANGLEFEM

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;

using std::cerr;
using std::cout;
using std::endl;

SOFA_DECL_CLASS(TriangularQuadraticSpringsForceField)

using namespace sofa::defaulttype;

template class TriangularQuadraticSpringsForceField<Vec3dTypes>;
template class TriangularQuadraticSpringsForceField<Vec3fTypes>;


// Register in the Factory
int TriangularQuadraticSpringsForceFieldClass = core::RegisterObject("Quadratic Springs on a Triangular Mesh")
        .add< TriangularQuadraticSpringsForceField<Vec3dTypes> >()
        .add< TriangularQuadraticSpringsForceField<Vec3fTypes> >()
        ;



} // namespace forcefield

} // namespace Components

} // namespace Sofa

