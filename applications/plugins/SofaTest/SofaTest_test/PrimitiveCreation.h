#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <sofa/helper/ArgumentParser.h>
#include <sofa/helper/UnitTest.h>
#include <sofa/helper/vector_algebra.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/system/PluginManager.h>

//#include <sofa/simulation/tree/TreeSimulation.h>
#ifdef SOFA_HAVE_DAG
#include <sofa/simulation/graph/DAGSimulation.h>
#endif
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/xml/initXml.h>

#include <sofa/gui/GUIManager.h>
#include <sofa/gui/Main.h>
#include <sofa/helper/system/FileRepository.h>

#include <ComponentMain/init.h>
#include <MiscMapping/SubsetMultiMapping.h>
#include <MiscMapping/DistanceMapping.h>
#include <MiscMapping/DistanceFromTargetMapping.h>
#include <BaseTopology/MeshTopology.h>
#include <BaseTopology/EdgeSetTopologyContainer.h>
#include <BaseCollision/SphereModel.h>
#include <BaseTopology/CubeTopology.h>
#include <BaseVisual/VisualStyle.h>
#include <ImplicitOdeSolver/EulerImplicitSolver.h>
#include <ExplicitOdeSolver/EulerSolver.h>
#include <BaseLinearSolver/CGLinearSolver.h>
#include <BaseCollision/OBBModel.h>
#include <sofa/simulation/tree/tree.h>
#include <sofa/simulation/tree/TreeSimulation.h>

//Using double by default, if you have SOFA_FLOAT in use in you sofa-default.cfg, then it will be FLOAT.
#include <sofa/component/typedef/Sofa_typedef.h>
//#include <plugins/SceneCreator/SceneCreator.h>


#include <sofa/simulation/common/Simulation.h>
#include <MiscCollision/DefaultCollisionGroupManager.h>
#include <sofa/simulation/tree/GNode.h>

#include <BaseTopology/MeshTopology.h>
#include <MeshCollision/MeshIntTool.h>

#include "Sofa_test.h"

namespace sofa{

typedef Vector3 Vec3;

/**
  *\brief Makes up an OBBModel containing just one OBB. angles and order are the rotations used to make up this OBB.
  *
  *\param p the center of the OBB
  *\param angles it is of size 3 and contains the rotations around axes, i.e., angles[0] contains rotation around x axis etc...
  *\param order it is the order we rotate, i.e, if we want to rotate first around z axis, then x axis and then y axis order will be {2,0,1}
  *\param v it is the velocity of the OBB
  *\param extents it contains half-extents of the OBB
  *\param father it is a node that will contain the returned OBBModel
  */
sofa::component::collision::OBBModel::SPtr makeOBB(const Vec3 & p,const double *angles,const int *order,const Vec3 &v,const Vec3 &extents, sofa::simulation::Node::SPtr &father);

sofa::component::collision::TriangleModel::SPtr makeTri(const Vec3 & p0,const Vec3 & p1,const Vec3 & p2,const Vec3 & v, sofa::simulation::Node::SPtr &father);

sofa::component::collision::CapsuleModel::SPtr makeCap(const Vec3 & p0,const Vec3 & p1,double radius,const Vec3 & v,
                                                                   sofa::simulation::Node::SPtr & father);

sofa::component::collision::RigidSphereModel::SPtr makeRigidSphere(const Vec3 & p,SReal radius,const Vec3 &v,const double *angles,const int *order,
                                                                            sofa::simulation::Node::SPtr & father);

sofa::component::collision::SphereModel::SPtr makeSphere(const Vec3 & p,SReal radius,const Vec3 & v,sofa::simulation::Node::SPtr & father);
}
