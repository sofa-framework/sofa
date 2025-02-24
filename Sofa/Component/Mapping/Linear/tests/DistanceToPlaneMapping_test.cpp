/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/component/mapping/linear/DistanceToPlaneMapping.h>
#include <sofa/testing/BaseTest.h>

#include <boost/function_types/components.hpp>

#include "sofa/core/MechanicalParams.h"
using sofa::testing::BaseTest;


#include <sofa/simulation/graph/DAGSimulation.h>
using sofa::simulation::graph::DAGSimulation;
using sofa::simulation::Node ;
using sofa::core::objectmodel::New ;
using sofa::core::objectmodel::BaseData ;

#include <sofa/component/statecontainer/MechanicalObject.h>
using sofa::component::statecontainer::MechanicalObject ;

#include <sofa/simulation/Node.h>
#include <sofa/linearalgebra/EigenSparseMatrix.h>
#include <sofa/core/trait/DataTypes.h>


using sofa::defaulttype::Vec1Types;
using sofa::defaulttype::Vec3Types;
using sofa::defaulttype::Rigid3Types;



template <class In>
sofa::simulation::Simulation* createSimpleScene(typename sofa::component::mapping::linear::DistanceToPlaneMapping<In>::SPtr mapping)
{
    sofa::simulation::Simulation* simu = sofa::simulation::getSimulation();

    const Node::SPtr node = simu->createNewGraph("root");

    typename MechanicalObject<In>::SPtr mechanical = New<MechanicalObject<In>>();
    mechanical->resize(10);
    auto inPos = mechanical->writePositions();
    inPos.resize(10);
    auto inRestPos = mechanical->writeRestPositions();
    inRestPos.resize(10);
    node->addObject(mechanical);


    const Node::SPtr nodeMapping = node->createChild("nodeToMap");
    typename MechanicalObject<sofa::defaulttype::Vec1Types>::SPtr targetMechanical = New<MechanicalObject<sofa::defaulttype::Vec1Types>>();
    nodeMapping->addObject(targetMechanical);
    mapping->setFrom(mechanical.get());
    mapping->setTo(targetMechanical.get());
    nodeMapping->addObject(mapping);

    EXPECT_NO_THROW(
        sofa::simulation::node::initRoot(node.get())
    );
    return simu;
}




GTEST_TEST(DistanceToPlaneMapping_Tests_Vec3d, init)
{
    typename sofa::component::mapping::linear::DistanceToPlaneMapping<Vec3Types>::SPtr mapping = New<sofa::component::mapping::linear::DistanceToPlaneMapping<Vec3Types>>();
    mapping->d_planeNormal.setValue(sofa::type::Vec3(1,2,5));
    auto simu = createSimpleScene<Vec3Types>(mapping);
    EXPECT_DOUBLE_EQ(mapping->d_planeNormal.getValue().norm(),1.0);
    EXPECT_EQ(mapping->getFrom()[0]->getSize(),10);
    EXPECT_EQ(mapping->getTo()[0]->getSize(),mapping->getFrom()[0]->getSize());
}

GTEST_TEST(DistanceToPlaneMapping_Tests_Vec3d, apply)
{
    typename sofa::component::mapping::linear::DistanceToPlaneMapping<Vec3Types>::SPtr mapping = New<sofa::component::mapping::linear::DistanceToPlaneMapping<Vec3Types>>();

    sofa::type::Vec3 planePoint{-1, 1, 2};
    sofa::type::Vec3 planeNormal{5, 1, 2}; planeNormal = planeNormal/planeNormal.norm();
    sofa::type::Vec3 planeTangent1{-1, 5, 0}; planeTangent1 = planeTangent1/planeTangent1.norm();
    sofa::type::Vec3 planeTangent2 = cross(planeNormal, planeTangent1);

    sofa::DataVecCoord_t<Vec3Types> inPos{sofa::DataVecCoord_t<Vec3Types>::InitData()};

    std::vector<double> dists{0.2,-0.5,-2.1,0.5};

    inPos.setValue({planePoint + planeNormal * dists[0],
                       planePoint + planeNormal * dists[1] + planeTangent1 * 10,
                       planePoint + planeNormal * dists[2] + planeTangent2 * 10,
                       planePoint + planeNormal * dists[3] + planeTangent1 * 2 + planeTangent2 * 4});
    sofa::DataVecCoord_t<Vec1Types> outVec; outVec.beginEdit()->resize(4);

    mapping->d_planeNormal.setValue(planeNormal);
    mapping->d_planePoint.setValue(planePoint);
    mapping->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
    mapping->apply(sofa::core::MechanicalParams::defaultInstance(),outVec,inPos);

    for (unsigned i = 0; i<outVec.getValue().size(); ++i)
    {
        EXPECT_DOUBLE_EQ(outVec.getValue()[i][0], dists[i]);
    }
}

GTEST_TEST(DistanceToPlaneMapping_Tests_Rigid3d, init)
{
    typename sofa::component::mapping::linear::DistanceToPlaneMapping<Rigid3Types>::SPtr mapping = New<sofa::component::mapping::linear::DistanceToPlaneMapping<Rigid3Types>>();
    mapping->d_planeNormal.setValue(Rigid3Types::Deriv(sofa::type::Vec3(1,2,5),sofa::type::Vec3(1,2,3)));

    auto simu = createSimpleScene<Rigid3Types>(mapping);
    EXPECT_EQ(mapping->getFrom()[0]->getSize(),10);
    EXPECT_EQ(mapping->getTo()[0]->getSize(),mapping->getFrom()[0]->getSize());
}


