/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
//#include <SofaBoundaryCondition/FixedConstraint.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/simulation/Simulation.h>
#include <SofaSimulationGraph/DAGSimulation.h>
#include <sofa/simulation/Node.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaBaseMechanics/UniformMass.h>
#include <SceneCreator/SceneCreator.h>
#include <SofaBoundaryCondition/ConstantForceField.h>
#include <SofaSimulationCommon/SceneLoaderXML.h>

#define CGALPLUGIN_MESHGENERATIONFROMIMAGE_CPP
#include <CGALPlugin/MeshGenerationFromImage.h>
#include <image/ImageTypes.h>


namespace sofa{
namespace {

using namespace modeling;
using core::objectmodel::New;
using sofa::simulation::SceneLoaderXML;
using sofa::core::ExecParams;


//template<typename DataTypes>
//void createUniformMass(simulation::Node::SPtr node, component::container::MechanicalObject<DataTypes>& /*dofs*/)
//{
//    node->addObject(New<component::mass::UniformMass<DataTypes, typename DataTypes::Real> >());
//}
//
//template<>
//void createUniformMass(simulation::Node::SPtr node, component::container::MechanicalObject<defaulttype::Rigid3Types>& /*dofs*/)
//{
//    node->addObject(New<component::mass::UniformMass<defaulttype::Rigid3Types, defaulttype::Rigid3Mass> >());
//}
//
//template<>
//void createUniformMass(simulation::Node::SPtr node, component::container::MechanicalObject<defaulttype::Rigid2Types>& /*dofs*/)
//{
//    node->addObject(New<component::mass::UniformMass<defaulttype::Rigid2Types, defaulttype::Rigid2Mass> >());
//}
//
//template <typename _DataTypes>
struct MeshFromImageWithFeatures_test : public Sofa_test<>
{
    sofa::simulation::Node::SPtr root;
    //typedef _DataTypes DataTypes;
    //typedef component::projectiveconstraintset::FixedConstraint<DataTypes> FixedConstraint;
    //typedef component::forcefield::ConstantForceField<DataTypes> ForceField;
    //typedef component::container::MechanicalObject<DataTypes> MechanicalObject;
    
    
    /*typedef typename MechanicalObject::VecCoord  VecCoord;
    typedef typename MechanicalObject::Coord  Coord;
    typedef typename MechanicalObject::VecDeriv  VecDeriv;
    typedef typename MechanicalObject::Deriv  Deriv;*/
    //typedef typename DataTypes::Real  Real;

    bool test() //double /*epsilon*/, const std::string &/*integrationScheme */)
    {
        cgal::MeshGenerationFromImage< defaulttype::Vec3dTypes, defaulttype::ImageUC >* meshGenerator;

        meshGenerator = dynamic_cast<cgal::MeshGenerationFromImage< defaulttype::Vec3dTypes, defaulttype::ImageUC >*>(root->getObject("generator"));

        helper::ReadAccessor< Data<sofa::defaulttype::Vec3dTypes::VecCoord> > features = meshGenerator->d_features;
        helper::ReadAccessor< Data<sofa::defaulttype::Vec3dTypes::VecCoord> > nodes = meshGenerator->f_newX0;

        double tolerance = 1e-10;
        size_t nfts = features.size();
        bool foundFeatures[nfts];
        std::cout << "#features to be tested: " << nfts << std::endl;
        for (size_t fi = 0; fi < nfts; fi++) {
            foundFeatures[fi] = false;
            const defaulttype::Vec3dTypes::Coord& ft = features[fi];
            for (size_t i = 0; i < nodes.size() && !foundFeatures[fi]; i++) {
                const defaulttype::Vec3dTypes::Coord & np = nodes[i];

                double dist = 0.0;
                for (size_t d = 0; d < 3; d++)
                    dist += helper::SQR(np[d] - ft[d]);
                dist = sqrt(dist);

                if (dist < tolerance) {
                    //std::cout << "Feature: " << features[fi] << std::endl;
                    //std::cout << "Node: [" << i << "]: " << np << std::endl;
                    foundFeatures[fi] = true;
                }
            }
        }

        for (size_t fi = 0; fi < nfts; fi++) {
            if (!foundFeatures[fi])
                return false;
        }

        return true;

	}

    void testDataFields()
    {
        std::cout << "Calling test fields!" << std::endl;
        /*EXPECT_EQ(grabber->d_fileName.getName(), "videoFile");
        EXPECT_EQ(grabber->d_seekFrame.getName(), "seek");
        EXPECT_EQ(grabber->d_camIdx.getName(), "cam_index");
        EXPECT_EQ(grabber->d_paused.getName(), "paused");
        EXPECT_EQ(grabber->d_stopped.getName(), "stopped");

        EXPECT_EQ(grabber->d_videoMode.getName(), "video_mode");
        EXPECT_EQ(grabber->d_fullFrame.getName(), "fullFrame");
        EXPECT_EQ(grabber->d_frame1.getName(), "img1");
        EXPECT_EQ(grabber->d_frame2.getName(), "img2");
        EXPECT_EQ(grabber->d_fps.getName(), "fps");
        EXPECT_EQ(grabber->d_dimensions.getName(), "dimensions");*/
    }

    void SetUp()
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
                            image="@image.image" transform="@image.transform"  features="@loader.position"
                            cellSize="5" edgeSize="5"  facetSize="5" facetApproximation="0.1"  facetAngle="30" cellRatio="3"  ordering="0"
                            label="1 2 3" labelCellSize="0.15 0.15 0.15" labelCellData="100 200 300"/>
                </Node>
        )scene2";

        std::string scene1(s1);
        std::string scene2(s2);

        std::string scene = scene1+vtkLoaderLine+imageContainerLine+scene2;

        std::cout << "Scene: \n " << scene << std::endl;

        root = sofa::simulation::SceneLoaderXML::loadFromMemory(
                "scene", scene.c_str(), scene.size());
        root->init(sofa::core::ExecParams::defaultInstance());

    }
};

// Define the list of DataTypes to instantiate
//using testing::Types;
//typedef Types<
//    defaulttype::Vec1Types,
//    defaulttype::Vec2Types,
//    defaulttype::Vec3Types
//    defaulttype::Vec6Types,
//    defaulttype::Rigid2Types,
//    defaulttype::Rigid3Types
//> DataTypes; // the types to instantiate.
//
//// Test suite for all the instantiations
//TYPED_TEST_CASE(MeshFromImageWithFeatures_test, DataTypes);
//// first test case
//TYPED_TEST( MeshFromImageWithFeatures_test , testValueImplicitWithCG )
//{
//    EXPECT_TRUE(  this->test(1e-8,std::string("Implicit")) );
//}
//
//TYPED_TEST( MeshFromImageWithFeatures_test , testValueExplicit )
//{
//    EXPECT_TRUE(  this->test(1e-8, std::string("Explicit")) );
//}

TEST_F(MeshFromImageWithFeatures_test, test)
{
    EXPECT_TRUE(this->test());
}


//
//#ifdef SOFA_HAVE_METIS
//TYPED_TEST( MeshFromImageWithFeatures_test , testValueImplicitWithSparseLDL )
//{
//    EXPECT_TRUE(  this->test(1e-8, std::string("Implicit_SparseLDL")) );
//}
//#endif
//

}// namespace
}// namespace sofa







