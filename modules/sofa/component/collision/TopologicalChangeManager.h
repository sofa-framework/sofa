#ifndef SOFA_COMPONENT_COLLISION_TOPOLOGICALCHANGEMANAGER_H
#define SOFA_COMPONENT_COLLISION_TOPOLOGICALCHANGEMANAGER_H

#include <sofa/core/CollisionElement.h>

#include <sofa/core/BehaviorModel.h>

#include <sofa/component/topology/TriangleSetTopology.h>
#include <sofa/component/topology/TetrahedronSetTopology.h>
#include <sofa/component/topology/QuadSetTopology.h>
#include <sofa/component/topology/HexahedronSetTopology.h>

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Vec3Types.h>

#include <sofa/component/MechanicalObject.h>
#include <sofa/simulation/tree/GNode.h>

// TopologicalChangeManager class to handle topological changes

namespace sofa
{

namespace component
{
namespace collision
{
class TriangleSetModel;
}

namespace topology
{
template<class T>
class TriangleSetTopology;
template<class T>
class TetrahedronSetTopology;
template<class T>
class HexahedronSetTopology;
template<class T>
class QuadSetTopology;
class QuadSetTopologyContainer;
class TetrahedronSetTopologyContainer;
class HexahedronSetTopologyContainer;
}
}

namespace component
{

namespace collision
{
using namespace sofa::defaulttype;

class TopologicalChangeManager
{
public:


protected:


public:
    TopologicalChangeManager();

    ~TopologicalChangeManager();

    // Handle Removing of topological element (from any type of topology)
    void removeItemsFromCollisionModel(sofa::core::CollisionElementIterator) const;

    // Intermediate method to handle cutting
    bool incisionTriangleSetTopology(topology::TriangleSetTopology< Vec3Types >*);

    // Handle Cutting (activated only for a triangular topology), using global variables to register the two last input points
    bool incisionTriangleModel(sofa::core::CollisionElementIterator, Vector3&, bool, bool);


private:
    // Global variables to register the two last input points (for incision along one segment in a triangular mesh)
    struct Incision
    {
        Vec<3,double> a_init;
        Vec<3,double> b_init;
        unsigned int ind_ta_init;
        unsigned int ind_tb_init;

        bool is_first_cut;

        unsigned int b_last_init;
        sofa::helper::vector< unsigned int > b_p12_last_init;
        sofa::helper::vector< unsigned int > b_i123_last_init;

        unsigned int a_last_init;
        sofa::helper::vector< unsigned int >  a_p12_last_init;
        sofa::helper::vector< unsigned int >  a_i123_last_init;
    }	incision;

};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
