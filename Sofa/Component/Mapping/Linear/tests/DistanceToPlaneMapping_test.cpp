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

#include "sofa/core/ConstraintParams.h"
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

static constexpr SReal EPSILON=1e-12;

using sofa::defaulttype::Vec1Types;

template <class DataType>
class PlaneMappingTest : public testing::Test
{

    typedef sofa::type::Vec<sofa::Deriv_t<DataType>::spatial_dimensions,typename sofa::Deriv_t<DataType>::value_type> PlaneNormalType;
    typedef sofa::type::Vec<sofa::Coord_t<DataType>::spatial_dimensions,typename sofa::Coord_t<DataType>::value_type> PlanePointType;
public:
    sofa::simulation::Simulation* createSimpleScene(typename sofa::component::mapping::linear::DistanceToPlaneMapping<DataType>::SPtr mapping)
    {


        sofa::simulation::Simulation* simu = sofa::simulation::getSimulation();

        const Node::SPtr node = simu->createNewGraph("root");

        typename MechanicalObject<DataType>::SPtr mechanical = New<MechanicalObject<DataType>>();
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

    PlaneNormalType getPseudoRandomNormal()
    {
        constexpr std::array<SReal,6> randomValuesForNormal = {1, 5.1, -6, 0, 9.5, 4};
        PlaneNormalType returnVec;
        for (unsigned i = 0; i< sofa::Deriv_t<DataType>::spatial_dimensions; ++i)
        {
            returnVec[i] = randomValuesForNormal[i];
        }
        return returnVec;
    }


    PlanePointType getPseudoRandomPoint()
    {
        constexpr std::array<SReal,7> randomValuesForPoint = {-2.5, 1.4, 3, 0, 0.7, 12, 7.07};
        PlanePointType returnVec;
        for (unsigned i = 0; i< sofa::Coord_t<DataType>::spatial_dimensions; ++i)
        {
            returnVec[i] = randomValuesForPoint[i];
        }
        return returnVec;

    }

    void testInit(PlaneNormalType planeNormal)
    {
        typename sofa::component::mapping::linear::DistanceToPlaneMapping<DataType>::SPtr mapping = New<sofa::component::mapping::linear::DistanceToPlaneMapping<DataType>>();
        mapping->d_planeNormal.setValue(planeNormal);
        this->createSimpleScene(mapping);
        EXPECT_LE(std::fabs(mapping->d_planeNormal.getValue().norm() - 1.0),EPSILON);
        EXPECT_EQ(mapping->getFrom()[0]->getSize(),10);
        EXPECT_EQ(mapping->getTo()[0]->getSize(),mapping->getFrom()[0]->getSize());
    }

    void testApply(PlaneNormalType planeNormal, PlanePointType planePoint)
    {
        typename sofa::component::mapping::linear::DistanceToPlaneMapping<DataType>::SPtr mapping = New<sofa::component::mapping::linear::DistanceToPlaneMapping<DataType>>();

        planeNormal = planeNormal/planeNormal.norm();
        sofa::Deriv_t<DataType> fullNormal;
        DataType::setDPos(fullNormal, planeNormal);

        sofa::Coord_t<DataType> fullPlanePoint;
        DataType::setCPos(fullPlanePoint, planePoint);

        sofa::Deriv_t<DataType> planeTangent1 = fullNormal;
        planeTangent1[0] = planeNormal[1];
        planeTangent1[1] = -planeNormal[0];
        for (unsigned i = 2; i< sofa::Deriv_t<DataType>::size(); ++i)
        {
            planeTangent1[i] = 0;
        }

        planeTangent1 = planeTangent1/planeTangent1.norm();

        sofa::DataVecCoord_t<DataType> inPos{typename sofa::DataVecCoord_t<DataType>::InitData()};

        std::vector<SReal> dists{0.2,-0.5,-2.1,0.5};

        inPos.setValue({fullPlanePoint + fullNormal * dists[0],
                           fullPlanePoint + fullNormal * dists[1] + planeTangent1 * 10,
                           fullPlanePoint + fullNormal * dists[2] ,
                           fullPlanePoint + fullNormal * dists[3] + planeTangent1 * 2 });
        sofa::DataVecCoord_t<Vec1Types> outVec; outVec.beginEdit()->resize(4);

        mapping->d_planeNormal.setValue(planeNormal);
        mapping->d_planePoint.setValue(planePoint);
        mapping->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
        mapping->apply(sofa::core::MechanicalParams::defaultInstance(),outVec,inPos);

        for (unsigned i = 0; i<outVec.getValue().size(); ++i)
        {
            EXPECT_LE(std::fabs(outVec.getValue()[i][0] - dists[i]), EPSILON);
        }
    }

    void testApplyJ(PlaneNormalType planeNormal)
    {
        typename sofa::component::mapping::linear::DistanceToPlaneMapping<DataType>::SPtr mapping = New<sofa::component::mapping::linear::DistanceToPlaneMapping<DataType>>();

        planeNormal = planeNormal/planeNormal.norm();
        sofa::Deriv_t<DataType> fullNormal;
        DataType::setDPos(fullNormal, planeNormal);
        sofa::Deriv_t<DataType> planeTangent1 = fullNormal;
        planeTangent1[0] = planeNormal[1];
        planeTangent1[1] = -planeNormal[0];
        for (unsigned i = 2; i< PlaneNormalType::size(); ++i)
        {
            planeTangent1[i] = 0;
        }

        planeTangent1 = planeTangent1/planeTangent1.norm();

        sofa::DataVecDeriv_t<DataType> inPos{typename sofa::DataVecDeriv_t<DataType>::InitData()};

        std::vector<SReal> dists{0.2,-0.5,-2.1,0.5};

        inPos.setValue({ fullNormal * dists[0],
                            fullNormal * dists[1] + planeTangent1 * 10,
                            fullNormal * dists[2],
                            fullNormal * dists[3] + planeTangent1 * 2});
        sofa::DataVecCoord_t<Vec1Types> outVec; outVec.beginEdit()->resize(4);

        mapping->d_planeNormal.setValue(planeNormal);
        mapping->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
        mapping->applyJ(sofa::core::MechanicalParams::defaultInstance(),outVec,inPos);

        for (unsigned i = 0; i<outVec.getValue().size(); ++i)
        {
            EXPECT_LE(std::fabs(outVec.getValue()[i][0] - dists[i]), EPSILON  );
        }
    }

    void testApplyJT_Force(PlaneNormalType planeNormal)
    {
        typename sofa::component::mapping::linear::DistanceToPlaneMapping<DataType>::SPtr mapping = New<sofa::component::mapping::linear::DistanceToPlaneMapping<DataType>>();

        planeNormal = planeNormal/planeNormal.norm();

        sofa::DataVecDeriv_t<Vec1Types> inPos{sofa::DataVecDeriv_t<Vec1Types>::InitData()};


        inPos.setValue({sofa::type::Vec1(0.2),sofa::type::Vec1(-0.5),sofa::type::Vec1(-2.1),sofa::type::Vec1(0.5)});

        sofa::DataVecDeriv_t<DataType> outVec; outVec.beginEdit()->resize(4);

        mapping->d_planeNormal.setValue(planeNormal);
        mapping->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
        mapping->applyJT(sofa::core::MechanicalParams::defaultInstance(),outVec,inPos);

        for (unsigned i = 0; i<outVec.getValue().size(); ++i)
        {
            for (unsigned j=0; j< PlaneNormalType::size(); ++j)
            {
                EXPECT_LE(std::fabs(outVec.getValue()[i][j] - planeNormal[j] * inPos.getValue()[i][0]),EPSILON);
            }
        }
    }

    void testApplyJT_Constraint(PlaneNormalType planeNormal)
    {
        typename sofa::component::mapping::linear::DistanceToPlaneMapping<DataType>::SPtr mapping = New<sofa::component::mapping::linear::DistanceToPlaneMapping<DataType>>();

        planeNormal = planeNormal/planeNormal.norm();

        sofa::DataMatrixDeriv_t<Vec1Types> inMat{sofa::DataMatrixDeriv_t<Vec1Types>::InitData()};
        sofa::DataMatrixDeriv_t<DataType> outMat{typename sofa::DataMatrixDeriv_t<DataType>::InitData()};


        auto writeMatrixIn = sofa::helper::getWriteAccessor(inMat);

        std::vector<unsigned> dofsIds = {2,4,5,7};

        unsigned lineId = 0;
        for (unsigned i : dofsIds)
        {
            auto o = writeMatrixIn->writeLine(lineId++);
            o.addCol(i,Vec1Types::Deriv( 1 - 2*(lineId%1)));
        }

        mapping->d_planeNormal.setValue(planeNormal);
        mapping->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
        mapping->applyJT(sofa::core::ConstraintParams::defaultInstance(),outMat,inMat);

        const auto readMatrixOut = sofa::helper::getReadAccessor(outMat);
        for (auto rowIt = readMatrixOut->begin(); rowIt != readMatrixOut->end(); ++rowIt)
        {
            auto colIt = rowIt.begin();
            auto colItEnd = rowIt.end();

            EXPECT_GT(dofsIds.size(),rowIt.index());

            EXPECT_FALSE(colIt == colItEnd);

            EXPECT_EQ(colIt.index(),dofsIds[rowIt.index()]);
            EXPECT_EQ(DataType::getDPos(colIt.val()),planeNormal*( 1 - 2*(rowIt.index()%1)));

            if constexpr (sofa::type::isRigidType<DataType>)
            {
                EXPECT_EQ(DataType::getDRot(colIt.val()),typename sofa::Deriv_t<DataType>::Rot() *0 );
            }

            ++colIt;
            EXPECT_TRUE(colIt == colItEnd);
        }
    }
};


using sofa::defaulttype::Vec2Types;
using sofa::defaulttype::Vec3Types;
using sofa::defaulttype::Vec6Types;
using sofa::defaulttype::Rigid3Types;
using sofa::defaulttype::Rigid2Types;


using DataTypes = testing::Types<Vec2Types, Vec3Types, Vec6Types, Rigid3Types, Rigid2Types >;
TYPED_TEST_SUITE(PlaneMappingTest, DataTypes);

TYPED_TEST(PlaneMappingTest, init)
{

    this->testInit(this->getPseudoRandomNormal());
}

TYPED_TEST(PlaneMappingTest, apply)
{
    this->testApply(this->getPseudoRandomNormal(),this->getPseudoRandomPoint());
}

TYPED_TEST(PlaneMappingTest, applyJ)
{
    this->testApplyJ(this->getPseudoRandomNormal());
}

TYPED_TEST(PlaneMappingTest, applyJT_Forces)
{
    this->testApplyJT_Force(this->getPseudoRandomNormal());
}

TYPED_TEST(PlaneMappingTest, applyJT_Constraints)
{
    this->testApplyJT_Constraint(this->getPseudoRandomNormal());
}


