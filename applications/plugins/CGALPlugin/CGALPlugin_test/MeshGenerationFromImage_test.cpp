/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <SofaTest/Sofa_test.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/simulation/Simulation.h>
#include <SofaSimulationGraph/DAGSimulation.h>
#include <sofa/simulation/Node.h>
#include <sofa/helper/logging/Messaging.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaBaseMechanics/UniformMass.h>
#include <SceneCreator/SceneCreator.h>
#include <SofaBoundaryCondition/ConstantForceField.h>
#include <SofaSimulationCommon/SceneLoaderXML.h>

#define CGALPLUGIN_MESHGENERATIONFROMIMAGE_CPP

#include <CGALPlugin/MeshGenerationFromImage.h>
#include <image/ImageTypes.h>

namespace cgal
{

using namespace sofa;
using namespace modeling;
using core::objectmodel::New;
using sofa::simulation::SceneLoaderXML;
using sofa::core::ExecParams;

struct MeshGenerationFromImage_test : public Sofa_test<>
{
    sofa::simulation::Node::SPtr m_root;
    cgal::MeshGenerationFromImage< defaulttype::Vec3dTypes, defaulttype::ImageUC >::SPtr m_meshGenerator;

    void SetUp()
    {

    }

    void loadSceneDefault()
    {
        /// a scene defined using raw-strings (C++11)
        const char* s1 = R"scene1(
                <Node name="root" gravity="0 0 0" dt="1"  >
                    <RequiredPlugin pluginName="CGALPlugin"/>
                    <RequiredPlugin pluginName="image"/>
                )scene1";

        std::string imageContainerLine = "<ImageContainer name=\"image\" template=\"ImageUC\" filename=\"" + std::string(SOFACGAL_TEST_RESOURCES_DIR) + "image/image-cube.inr\"/>";

        const char *s2 = R"scene2(
                    <MeshGenerationFromImage template="Vec3d" name="generator" printLog="true" drawTetras="false"
                            image="@image.image" transform="@image.transform"
                            cellSize="5" facetSize="5" edgeSize="5" facetApproximation="0.1"  facetAngle="30" cellRatio="3"  ordering="0"
                         />
                </Node>
        )scene2";

        //
        std::string scene1(s1);
        std::string scene2(s2);

        std::string scene = scene1+imageContainerLine+scene2;

        m_root = sofa::simulation::SceneLoaderXML::loadFromMemory(
                "scene", scene.c_str(), scene.size());

        m_root->getContext()->get(m_meshGenerator);

        EXPECT_NE(m_meshGenerator, nullptr);
    }

    void loadSceneWithFeatures()
    {
        /// a scene defined using raw-strings (C++11)
        const char* s1 = R"scene1(
                <Node name="root" gravity="0 0 0" dt="1"  >
                    <RequiredPlugin pluginName="CGALPlugin"/>
                    <RequiredPlugin pluginName="image"/>
                )scene1";

        std::string vtkLoaderLine = "<MeshVTKLoader name=\"loader\" filename=\""+ std::string(SOFACGAL_TEST_RESOURCES_DIR) + "edgePoints.vtk" +"\"/>\n";
        std::string imageContainerLine = "<ImageContainer name=\"image\" template=\"ImageUC\" filename=\"" + std::string(SOFACGAL_TEST_RESOURCES_DIR) + "image/image-cube.inr\"/>";

        const char *s2 = R"scene2(
                    <MeshGenerationFromImage template="Vec3d" name="generator" printLog="true" drawTetras="true"
                            image="@image.image" transform="@image.transform"
                            cellSize="5" facetSize="5" facetApproximation="0.1"  facetAngle="30" cellRatio="3"  ordering="0"
                            label="1 2 3" labelCellSize="0.15 0.15 0.15" labelCellData="100 200 300" features="@loader.position"/>
                </Node>
        )scene2";

        //
        std::string scene1(s1);
        std::string scene2(s2);

        std::string scene = scene1+vtkLoaderLine+imageContainerLine+scene2;

        m_root = sofa::simulation::SceneLoaderXML::loadFromMemory(
                "scene", scene.c_str(), scene.size());

        m_root->getContext()->get(m_meshGenerator);

        EXPECT_NE(m_meshGenerator, nullptr);
    }
};

/*
 * This test needs a fix in VTKLoader (to allow the use of the VERTICES keyword)
 *
TEST_F(MeshGenerationFromImage_test, MeshFromImageWithFeatures_test)
{
    bool result = true;

    loadSceneWithFeatures();
    m_root->init(sofa::core::ExecParams::defaultInstance());

    helper::ReadAccessor< Data<sofa::defaulttype::Vec3dTypes::VecCoord> > features = m_meshGenerator->d_features;
    helper::ReadAccessor< Data<sofa::defaulttype::Vec3dTypes::VecCoord> > nodes = m_meshGenerator->d_newX0;

    double tolerance = 1e-10;
    size_t nfts = features.size();
    bool foundFeatures[nfts];
    msg_info("MeshFromImageWithFeatures_test") << "#features to be tested: " << nfts;
    for (size_t fi = 0; fi < nfts; fi++) {
        foundFeatures[fi] = false;
        const defaulttype::Vec3dTypes::Coord& ft = features[fi];
        for (size_t i = 0; i < nodes.size() && !foundFeatures[fi]; i++)
        {
            const defaulttype::Vec3dTypes::Coord & np = nodes[i];

            double dist = 0.0;
            for (size_t d = 0; d < 3; d++)
                dist += helper::SQR(np[d] - ft[d]);
            dist = sqrt(dist);

            if (dist < tolerance)
            {
                foundFeatures[fi] = true;
            }
        }
    }

    for (size_t fi = 0; fi < nfts; fi++)
    {
        if (!foundFeatures[fi])
            result = false;
    }

    EXPECT_TRUE(result);

}

*/

TEST_F(MeshGenerationFromImage_test, MeshFromImageCellSize)
{
    loadSceneDefault();

    double oldSize = m_meshGenerator->d_cellSize.getValue();
    double newSize = oldSize * 0.1;

    m_root->init(sofa::core::ExecParams::defaultInstance());
    double oldNbTetra = m_meshGenerator->d_tetrahedra.getValue().size();

    m_meshGenerator->d_cellSize.setValue(newSize);
    //frozen is set to True when the computation is done for the first time
    //and prevent us to execute for a second time update().
    //so we need to set it to False.
    m_meshGenerator->d_frozen.setValue(false);
    m_meshGenerator->update();

    double newNbTetra = m_meshGenerator->d_tetrahedra.getValue().size();

    EXPECT_LT(oldNbTetra, newNbTetra);
}

TEST_F(MeshGenerationFromImage_test, MeshFromImageFacetSize)
{
    loadSceneDefault();

    double oldSize = m_meshGenerator->d_facetSize.getValue();
    double newSize = oldSize * 0.01;

    m_root->init(sofa::core::ExecParams::defaultInstance());
    double oldNbTetra = m_meshGenerator->d_tetrahedra.getValue().size();

    m_meshGenerator->d_facetSize.setValue(newSize);
    m_meshGenerator->d_frozen.setValue(false);
    m_meshGenerator->update();
    double newNbTetra = m_meshGenerator->d_tetrahedra.getValue().size();

    EXPECT_LT(oldNbTetra, newNbTetra);
}

TEST_F(MeshGenerationFromImage_test, MeshFromImageLabelSize)
{
    loadSceneDefault();

    sofa::helper::vector<int> newLabel;
    newLabel.push_back(1);
    newLabel.push_back(2);
    newLabel.push_back(3);
    sofa::helper::vector<double> newLabelCellSize;
    double newSize = 0.1;
    newLabelCellSize.push_back(newSize);
    newLabelCellSize.push_back(newSize);
    newLabelCellSize.push_back(newSize);
    sofa::helper::vector<double> newLabelCellData;
    newLabelCellData.push_back(100);
    newLabelCellData.push_back(200);
    newLabelCellData.push_back(300);


    m_root->init(sofa::core::ExecParams::defaultInstance());
    double oldNbTetra = m_meshGenerator->d_tetrahedra.getValue().size();

    m_meshGenerator->d_label.setValue(newLabel);
    m_meshGenerator->d_labelCellSize.setValue(newLabelCellSize);
    m_meshGenerator->d_labelCellData.setValue(newLabelCellData);
    m_meshGenerator->d_frozen.setValue(false);
    m_meshGenerator->update();
    double newNbTetra = m_meshGenerator->d_tetrahedra.getValue().size();

    EXPECT_LT(oldNbTetra, newNbTetra);
}



}// namespace cgal








