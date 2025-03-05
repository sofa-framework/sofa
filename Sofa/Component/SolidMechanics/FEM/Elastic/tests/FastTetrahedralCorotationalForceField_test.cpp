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
#include "BaseTetrahedronFEMForceField_test.h"
#include <sofa/component/solidmechanics/fem/elastic/FastTetrahedralCorotationalForceField.h>

namespace sofa
{

using FastTetrahedralCorotationalForceField3 = sofa::component::solidmechanics::fem::elastic::FastTetrahedralCorotationalForceField<sofa::defaulttype::Vec3Types>;

INSTANTIATE_TYPED_TEST_SUITE_P(
    FastTetrahedralCorotationalForceField_test,
    BaseTetrahedronFEMForceField_test,
    FastTetrahedralCorotationalForceField3
);

class FastTetrahedralCorotationalForceField_test : public BaseTetrahedronFEMForceField_test<FastTetrahedralCorotationalForceField3>
{
public:
    void checkInit() override
    {
        Transformation exp_initRot;
        TetraCoord exp_initPos;
        Transformation exp_curRot;
        MaterialStiffness exp_stiffnessMat;
        StrainDisplacement exp_strainD;

        setupCheckInit(exp_initRot, exp_initPos, exp_curRot, exp_stiffnessMat, exp_strainD);

        Transformation initRot (type::NOINIT);
        Transformation curRot(type::NOINIT);

        typename FastTetrahedralCorotationalForceField3::SPtr tetraFEM = m_root->getTreeObject<FastTetrahedralCorotationalForceField3>();
        ASSERT_TRUE(tetraFEM.get() != nullptr);

        const typename FastTetrahedralCorotationalForceField3::TetrahedronRestInformation& tetraInfo = tetraFEM->d_tetrahedronInfo.getValue()[0];
        initRot.transpose(tetraInfo.restRotation); // TODO check why transposed is stored in this version
        curRot = initRot; // not needed at init.

        // Expected specific values
        TetraCoord exp_shapeVector = { Coord(0, 1, 0), Coord(0, 0, 1), Coord(1, 0, 0), Coord(-1, -1, -1) };
        Transformation exp_linearDfDxDiag[4];
        exp_linearDfDxDiag[0] = { Vec3(64.1026, 0, 0), Vec3(0, 224.359, 0), Vec3(0, 0, 64.1026) };
        exp_linearDfDxDiag[1] = { Vec3(64.1026, 0, -0), Vec3(0, 64.1026, -0), Vec3(-0, -0, 224.359) };
        exp_linearDfDxDiag[2] = { Vec3(224.359, 0, 0), Vec3(0, 64.1026, 0), Vec3(0, 0, 64.1026) };
        exp_linearDfDxDiag[3] = { Vec3(352.5641, 160.25641, 160.25641), Vec3(160.25641, 352.5641, 160.25641), Vec3(160.25641, 160.25641, 352.5641) };

        Transformation exp_linearDfDx[6];
        exp_linearDfDx[0] = { Vec3(0, -0, 0), Vec3(0, 0, 64.1026), Vec3(0, 96.1538, 0) };
        exp_linearDfDx[1] = { Vec3(0, 96.1538, 0), Vec3(64.1026, 0, 0), Vec3(0, 0, 0) };
        exp_linearDfDx[2] = { Vec3(-64.1026, -96.1538, -0), Vec3(-64.1026, -224.359, -64.1026), Vec3(-0, -96.1538, -64.1026) };
        exp_linearDfDx[3] = { Vec3(0, -0, 96.1538), Vec3(-0, 0, 0), Vec3(64.1026, 0, 0) };
        exp_linearDfDx[4] = { Vec3(-64.1026, 0, -96.1538), Vec3(0, -64.1026, -96.1538), Vec3(-64.1026, -64.1026, -224.359) };
        exp_linearDfDx[5] = { Vec3(-224.359, -64.1026, -64.1026), Vec3(-96.1538, -64.1026, -0), Vec3(-96.1538, -0, -64.1026) };


        // check rotations
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                EXPECT_NEAR(exp_initRot[i][j], initRot[i][j], 1e-4);
                EXPECT_NEAR(exp_curRot[i][j], curRot[i][j], 1e-4);
            }
        }

        // check shapeVector
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                EXPECT_NEAR(exp_shapeVector[i][j], tetraInfo.shapeVector[i][j], 1e-4);
            }
        }

        // check DfDx
        for (int id = 0; id < 4; ++id)
        {
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    EXPECT_NEAR(exp_linearDfDxDiag[id][i][j], tetraInfo.linearDfDxDiag[id][i][j], 1e-4);
                    EXPECT_NEAR(exp_linearDfDx[id][i][j], tetraInfo.linearDfDx[id][i][j], 1e-4);
                }
            }
        }
    }

    void checkFEMValues() override
    {
        Transformation exp_initRot;
        TetraCoord exp_initPos;
        Transformation exp_curRot;
        MaterialStiffness exp_stiffnessMat;
        StrainDisplacement exp_strainD;

        setupCheckFEMValues(exp_initRot, exp_initPos, exp_curRot, exp_stiffnessMat, exp_strainD);

        Transformation initRot(type::NOINIT);
        Transformation curRot(type::NOINIT);
        MaterialStiffness stiffnessMat(type::NOINIT);
        StrainDisplacement strainD(type::NOINIT);
        TetraCoord initPosition;

        constexpr sofa::Size elementId = 100;
        computeMatricesCheckInit(initRot, curRot, stiffnessMat, strainD, initPosition, elementId);

        typename FastTetrahedralCorotationalForceField3::SPtr tetraFEM = m_root->getTreeObject<FastTetrahedralCorotationalForceField3>();
        ASSERT_TRUE(tetraFEM.get() != nullptr);

        const typename FastTetrahedralCorotationalForceField3::TetrahedronRestInformation& tetraInfo = tetraFEM->d_tetrahedronInfo.getValue()[100];
        initRot.transpose(tetraInfo.restRotation); // TODO check why transposed is stored in this version
        curRot = tetraInfo.rotation;

        // Expected specific values
        exp_curRot = { Vec3(0.99999985, 0.00032076406, -0.00043657642), Vec3(-0.00033142383, 0.99969634, -0.024639719), Vec3(0.00042854031, 0.024639861, 0.9996963) };
        TetraCoord exp_shapeVector = { Coord(0.3, 0.224999, -0.3), Coord(-0.3, 0, 0.3), Coord(0, 0, -0.3), Coord(0, -0.224999, 0.3) };

        Transformation exp_linearDfDxDiag[4];
        exp_linearDfDxDiag[0] = { Vec3(865.38462, 320.51282, -427.35043), Vec3(320.51282, 678.4188, -320.51282), Vec3(-427.35043, -320.51282, 865.38462) };
        exp_linearDfDxDiag[1] = { Vec3(769.23077, -0, -427.35043), Vec3(-0, 341.88034, 0), Vec3(-427.35043, 0, 769.23077) };
        exp_linearDfDxDiag[2] = { Vec3(170.94017, -0, 0), Vec3(-0, 170.94017, -0), Vec3(0, -0, 598.2906) };
        exp_linearDfDxDiag[3] = { Vec3(267.09402, -0, 0), Vec3(-0, 507.47863, -320.51282), Vec3(0, -320.51282, 694.44444) };

        Transformation exp_linearDfDx[6];
        exp_linearDfDx[0] = { Vec3(-769.23077, -192.30769, 427.35043), Vec3(-128.20513, -341.88034, 128.20513), Vec3(427.35043, 192.30769, -769.23077) };
        exp_linearDfDx[1] = { Vec3(170.94017, 0, -170.94017), Vec3(0, 170.94017, -128.20513), Vec3(-256.41026, -192.30769, 598.2906) };
        exp_linearDfDx[2] = { Vec3(-267.09402, -128.20513, 170.94017), Vec3(-192.30769, -507.47863, 320.51282), Vec3(256.41026, 320.51282, -694.44444) };
        exp_linearDfDx[3] = { Vec3(-170.94017, -0, 170.94017), Vec3(-0, -170.94017, 0), Vec3(256.41026, 0, -598.2906) };
        exp_linearDfDx[4] = { Vec3(170.94017, 128.20513, -170.94017), Vec3(192.30769, 170.94017, -192.30769), Vec3(-256.41026, -128.20513, 598.2906) };
        exp_linearDfDx[5] = { Vec3(-170.94017, 0, -0), Vec3(0, -170.94017, 192.30769), Vec3(-0, 128.20513, -598.2906) };


        // check rotations
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                EXPECT_NEAR(exp_initRot[i][j], initRot[i][j], 1e-4);
                EXPECT_NEAR(exp_curRot[i][j], curRot[i][j], 1e-4);
            }
        }

        // check shapeVector
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                EXPECT_NEAR(exp_shapeVector[i][j], tetraInfo.shapeVector[i][j], 1e-4);
            }
        }

        // check DfDx
        for (int id = 0; id < 4; ++id)
        {
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    EXPECT_NEAR(exp_linearDfDxDiag[id][i][j], tetraInfo.linearDfDxDiag[id][i][j], 1e-4);
                    EXPECT_NEAR(exp_linearDfDx[id][i][j], tetraInfo.linearDfDx[id][i][j], 1e-4);
                }
            }
        }
    }
};

TEST_F(FastTetrahedralCorotationalForceField_test, init)
{
    this->checkInit();
}

TEST_F(FastTetrahedralCorotationalForceField_test, FEMValues)
{
    this->checkFEMValues();
}

TEST_F(FastTetrahedralCorotationalForceField_test, emptyTology)
{
    this->checkEmptyTopology();
}

}
