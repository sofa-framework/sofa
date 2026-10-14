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
#include <gtest/gtest.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/core/Mapping.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/simulation/DifferentialOperations.h>
#include <sofa/simulation/MappingGraph.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/component/mapping/nonlinear/SquareMapping.h>

namespace sofa
{

using DataTypes1D = sofa::defaulttype::Vec1Types;
using Coord1D = DataTypes1D::Coord;
using Deriv1D = DataTypes1D::Deriv;
using MechanicalObject1D = component::statecontainer::MechanicalObject<DataTypes1D>;

/**
 * @brief Linear Scale Mapping: y = s * x
 * Forward evaluation:   y = s * x
 * JVP (pushforward):    dy = s * dx
 * VJP (pullback):       dx* = s * dy*
 */
class ScaleMapping : public core::Mapping<DataTypes1D, DataTypes1D>
{
public:
    SOFA_CLASS(ScaleMapping, SOFA_TEMPLATE2(core::Mapping, DataTypes1D, DataTypes1D));

    double scale { 1.0 };

    ScaleMapping(double s = 1.0) : scale(s) {}

    void apply( const core::MechanicalParams* mparams, OutDataVecCoord& dOut, const InDataVecCoord& dIn) override
    {
        const auto in = sofa::helper::getReadAccessor(dIn);
        auto out = sofa::helper::getWriteOnlyAccessor(dOut);
        out.resize(in.size());

        for (size_t i = 0; i < in.size(); ++i)
        {
            out[i] = in[i] * scale;
        }
    }

    void applyJ( const core::MechanicalParams* mparams, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn) override
    {
        const auto in = sofa::helper::getReadAccessor(dIn);
        auto out = sofa::helper::getWriteOnlyAccessor(dOut);
        out.resize(in.size());

        for (size_t i = 0; i < in.size(); ++i)
        {
            out[i] = in[i] * scale;
        }
    }

    void applyJT(const core::MechanicalParams* mparams, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn) override
    {
        const auto in = sofa::helper::getReadAccessor(dIn);
        auto out = sofa::helper::getWriteOnlyAccessor(dOut);

        if (in.size() != out.size())
            out.resize(out.size());

        for (size_t i = 0; i < out.size(); ++i)
        {
            out[i] += in[i] * scale;
        }
    }

    void applyDJT(const core::MechanicalParams*,
                  core::MultiVecDerivId,
                  core::ConstMultiVecDerivId) override
    {
        // Linear mapping: d(J^T) = 0
    }
};

class DifferentialOperationsTest : public ::testing::Test
{
protected:
    simulation::Node::SPtr rootNode;

    void SetUp() override
    {
        rootNode = simulation::getSimulation()->createNewNode("root");
    }
};

/**
 * @test Test Forward-Mode AD (JVP) and Reverse-Mode AD (VJP) on a linear chain:
 * Graph: X --(Scale 2.0)--> Y --(Scale 3.0)--> Z
 * Composition: Z = 6.0 * X
 * JVP: dZ = 6.0 * dX
 * VJP: dX* = 6.0 * dZ*
 */
TEST_F(DifferentialOperationsTest, LinearChainAD)
{
    core::MechanicalParams mparams(*core::ExecParams::defaultInstance());

    auto nodeX = rootNode->createChild("X");
    auto nodeY = nodeX->createChild("Y");
    auto nodeZ = nodeY->createChild("Z");

    auto moX = core::objectmodel::New<MechanicalObject1D>();
    moX->setName("moX");
    moX->resize(1);
    moX->vRealloc(&mparams, sofa::core::vec_id::write_access::dx);
    (*moX->writePositions())[0] = Coord1D(2.0);
    nodeX->addObject(moX);

    auto moY = core::objectmodel::New<MechanicalObject1D>();
    moY->setName("moY");
    moY->resize(1);
    moY->vRealloc(&mparams, sofa::core::vec_id::write_access::dx);
    nodeY->addObject(moY);

    auto mapXY = core::objectmodel::New<ScaleMapping>(2.0);
    mapXY->setFrom(moX.get());
    mapXY->setTo(moY.get());
    nodeY->addObject(mapXY);

    auto moZ = core::objectmodel::New<MechanicalObject1D>();
    moZ->setName("moZ");
    moZ->resize(1);
    moZ->vRealloc(&mparams, sofa::core::vec_id::write_access::dx);
    nodeZ->addObject(moZ);

    auto mapYZ = core::objectmodel::New<ScaleMapping>(3.0);
    mapYZ->setFrom(moY.get());
    mapYZ->setTo(moZ.get());
    nodeZ->addObject(mapYZ);

    rootNode->init(core::ExecParams::defaultInstance());

    simulation::MappingGraph mappingGraph;
    mappingGraph.build(rootNode.get());

    // 1. Forward evaluation (pushforwardCoord)
    simulation::common::DifferentialOperations::pushforwardCoord(
        mappingGraph, mparams, core::MultiVecCoordId(core::vec_id::write_access::position));

    EXPECT_DOUBLE_EQ((*moY->readPositions())[0][0], 4.0);
    EXPECT_DOUBLE_EQ((*moZ->readPositions())[0][0], 12.0);

    // 2. Forward-mode AD / JVP (pushforwardTangent)
    // Seed tangent dX = 1.5
    helper::getWriteAccessor(*moX->write(core::vec_id::write_access::dx))[0] = Deriv1D(1.5);
    helper::getWriteAccessor(*moY->write(core::vec_id::write_access::dx))[0] = Deriv1D(0.0);
    helper::getWriteAccessor(*moZ->write(core::vec_id::write_access::dx))[0] = Deriv1D(0.0);

    simulation::common::DifferentialOperations::pushforwardTangent(
        mappingGraph, mparams, core::MultiVecDerivId(core::vec_id::write_access::dx));

    // dY = 2.0 * 1.5 = 3.0, dZ = 3.0 * 3.0 = 9.0 = 6.0 * 1.5
    EXPECT_DOUBLE_EQ(helper::getReadAccessor(*moY->read(core::vec_id::read_access::dx))[0][0], 3.0);
    EXPECT_DOUBLE_EQ(helper::getReadAccessor(*moZ->read(core::vec_id::read_access::dx))[0][0], 9.0);

    // 3. Reverse-mode AD / VJP (pullbackCotangent)
    // Seed adjoint at output dZ* = 1.0, reset inputs
    helper::getWriteAccessor(*moX->write(core::vec_id::write_access::force))[0] = Deriv1D(0.0);
    helper::getWriteAccessor(*moY->write(core::vec_id::write_access::force))[0] = Deriv1D(0.0);
    helper::getWriteAccessor(*moZ->write(core::vec_id::write_access::force))[0] = Deriv1D(1.0);

    simulation::common::DifferentialOperations::pullbackCotangent(
        mappingGraph, mparams, core::MultiVecDerivId(core::vec_id::write_access::force));

    // dY* += 3.0 * 1.0 = 3.0, dX* += 2.0 * 3.0 = 6.0
    EXPECT_DOUBLE_EQ(helper::getReadAccessor(*moY->read(core::vec_id::read_access::force))[0][0], 3.0);
    EXPECT_DOUBLE_EQ(helper::getReadAccessor(*moX->read(core::vec_id::read_access::force))[0][0], 6.0);
}

/**
 * @test Test Non-linear Function Composition:
 * Graph: X --(Square)--> Y --(Scale 3.0)--> Z
 * Function: Z = 3.0 * X^2
 * At X = 2.0:
 *   Z = 12.0
 * JVP at dX = 1.0:
 *   dZ = d(3*X^2)/dX * dX = 6*X * dX = 12.0
 * VJP with adjoint dZ* = 1.0:
 *   dX* = 6*X * dZ* = 12.0
 */
TEST_F(DifferentialOperationsTest, NonlinearCompositionAD)
{
    core::MechanicalParams mparams(*core::ExecParams::defaultInstance());

    auto nodeX = rootNode->createChild("X");
    auto nodeY = nodeX->createChild("Y");
    auto nodeZ = nodeY->createChild("Z");

    auto moX = core::objectmodel::New<MechanicalObject1D>();
    moX->setName("moX");
    moX->resize(1);
    moX->vRealloc(&mparams, sofa::core::vec_id::write_access::dx);
    (*moX->writePositions())[0] = Coord1D(2.0);
    nodeX->addObject(moX);

    auto moY = core::objectmodel::New<MechanicalObject1D>();
    moY->setName("moY");
    moY->resize(1);
    moY->vRealloc(&mparams, sofa::core::vec_id::write_access::dx);
    nodeY->addObject(moY);

    auto mapXY = core::objectmodel::New<sofa::component::mapping::nonlinear::SquareMapping<DataTypes1D, DataTypes1D>>();
    mapXY->setFrom(moX.get());
    mapXY->setTo(moY.get());
    nodeY->addObject(mapXY);

    auto moZ = core::objectmodel::New<MechanicalObject1D>();
    moZ->setName("moZ");
    moZ->resize(1);
    moZ->vRealloc(&mparams, sofa::core::vec_id::write_access::dx);
    nodeZ->addObject(moZ);

    auto mapYZ = core::objectmodel::New<ScaleMapping>(3.0);
    mapYZ->setFrom(moY.get());
    mapYZ->setTo(moZ.get());
    nodeZ->addObject(mapYZ);

    rootNode->init(core::ExecParams::defaultInstance());

    simulation::MappingGraph mappingGraph;
    mappingGraph.build(rootNode.get());

    // 1. Forward Pass
    simulation::common::DifferentialOperations::pushforwardCoord(
        mappingGraph, mparams, core::MultiVecCoordId(core::vec_id::write_access::position));

    EXPECT_DOUBLE_EQ((*moY->readPositions())[0][0], 4.0);
    EXPECT_DOUBLE_EQ((*moZ->readPositions())[0][0], 12.0);

    // 2. JVP (Pushforward Tangent)
    helper::getWriteAccessor(*moX->write(core::vec_id::write_access::dx))[0] = Deriv1D(1.0);
    helper::getWriteAccessor(*moY->write(core::vec_id::write_access::dx))[0] = Deriv1D(0.0);
    helper::getWriteAccessor(*moZ->write(core::vec_id::write_access::dx))[0] = Deriv1D(0.0);

    simulation::common::DifferentialOperations::pushforwardTangent(
        mappingGraph, mparams, core::MultiVecDerivId(core::vec_id::write_access::dx));

    // dY = 2 * 2.0 * 1.0 = 4.0
    EXPECT_DOUBLE_EQ(helper::getReadAccessor(*moY->read(core::vec_id::read_access::dx))[0][0], 4.0);
    // dZ = 3.0 * 4.0 = 12.0
    EXPECT_DOUBLE_EQ(helper::getReadAccessor(*moZ->read(core::vec_id::read_access::dx))[0][0], 12.0);

    // 3. VJP (Pullback Cotangent)
    helper::getWriteAccessor(*moX->write(core::vec_id::write_access::force))[0] = Deriv1D(0.0);
    helper::getWriteAccessor(*moY->write(core::vec_id::write_access::force))[0] = Deriv1D(0.0);
    helper::getWriteAccessor(*moZ->write(core::vec_id::write_access::force))[0] = Deriv1D(1.0);

    simulation::common::DifferentialOperations::pullbackCotangent(
        mappingGraph, mparams, core::MultiVecDerivId(core::vec_id::write_access::force));

    // dY* = 3.0 * 1.0 = 3.0
    EXPECT_DOUBLE_EQ(helper::getReadAccessor(*moY->read(core::vec_id::read_access::force))[0][0], 3.0);
    // dX* = 2 * 2.0 * 3.0 = 12.0
    EXPECT_DOUBLE_EQ(helper::getReadAccessor(*moX->read(core::vec_id::read_access::force))[0][0], 12.0);
}

/**
 * @test Test pullbackCotangentTangent (Second order directional derivative):
 * Graph: X --(Square)--> Y
 * Pullback cotangent-tangent:
 *   df_x = J^T df_y + d(J^T) f_y
 * When kFactor != 0, applyDJT contributes geometric stiffness: 2.0 * f_y * dx
 */
TEST_F(DifferentialOperationsTest, PullbackCotangentTangentHigherOrder)
{
    core::MechanicalParams mparams(*core::ExecParams::defaultInstance());
    mparams.setKFactor(1.0);

    auto nodeX = rootNode->createChild("X");
    auto nodeY = nodeX->createChild("Y");

    auto moX = core::objectmodel::New<MechanicalObject1D>();
    moX->setName("moX");
    moX->resize(1);
    moX->vRealloc(&mparams, sofa::core::vec_id::write_access::dx);
    (*moX->writePositions())[0] = Coord1D(3.0);
    nodeX->addObject(moX);

    auto moY = core::objectmodel::New<MechanicalObject1D>();
    moY->setName("moY");
    moY->resize(1);
    moY->vRealloc(&mparams, sofa::core::vec_id::write_access::dx);
    nodeY->addObject(moY);

    auto mapXY = core::objectmodel::New<sofa::component::mapping::nonlinear::SquareMapping<DataTypes1D, DataTypes1D>>();
    mapXY->setFrom(moX.get());
    mapXY->setTo(moY.get());
    nodeY->addObject(mapXY);

    rootNode->init(core::ExecParams::defaultInstance());

    simulation::MappingGraph mappingGraph;
    mappingGraph.build(rootNode.get());

    // Set tangent perturbation dx = 0.5 at input X
    helper::getWriteAccessor(*moX->write(core::vec_id::write_access::dx))[0] = Deriv1D(0.5);
    mparams.setDx(core::MultiVecDerivId(core::vec_id::write_access::dx));

    // Output cotangent-tangent df_y = 4.0
    helper::getWriteAccessor(*moX->write(core::vec_id::write_access::force))[0] = Deriv1D(0.0);
    helper::getWriteAccessor(*moY->write(core::vec_id::write_access::force))[0] = Deriv1D(4.0);

    simulation::common::DifferentialOperations::pullbackCotangentTangent(
        mappingGraph, mparams, core::MultiVecDerivId(core::vec_id::write_access::force));

    // 1. J^T * df_y = 2 * x * df_y = 2 * 3.0 * 4.0 = 24.0
    // 2. d(J^T) * df_y = 2.0 * df_y * dx = 2.0 * 4.0 * 0.5 = 4.0
    // Total df_x = 24.0 + 4.0 = 28.0
    EXPECT_DOUBLE_EQ(helper::getReadAccessor(*moX->read(core::vec_id::read_access::force))[0][0], 28.0);
}
}
