/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include <SofaTest/Sofa_test.h>
#include <SofaTest/DataEngine_test.h>
#include "../types/AffineTypes.h"


#include "../deformationMapping/ImageDeformation.h"
#include "../engine/ComputeWeightEngine.h"
#include "../engine/ComputeDualQuatEngine.h"
#include "../mass/MassFromDensity.h"
#include "../quadrature/GaussPointContainer.h"
#include "../quadrature/ImageGaussPointSampler.h"
#include "../quadrature/TopologyGaussPointSampler.h"
#include "../shapeFunction/ShapeFunctionDiscretizer.h"
#include "../shapeFunction/ImageShapeFunctionSelectNode.h"



#include <SceneCreator/SceneCreator.h>
#include <SofaSimulationGraph/DAGSimulation.h>
#include "../deformationMapping/LinearMapping.h"

namespace sofa {



/// a utility for FlexibleDataEngine test
/// allocating all engine's input Data<Image>
template <typename DataEngineType>
struct FlexibleDataEngine_test : public DataEngine_test<DataEngineType>
{
    typedef core::objectmodel::DDGNode DDGNode;
    typedef DDGNode::DDGLinkContainer DDGLinkContainer;

    // Root of the scene graph
    simulation::Node::SPtr root;


    virtual void init()
    {
        DataEngine_test<DataEngineType>::init();

        const DDGLinkContainer& parent_inputs = this->m_engineInput->DDGNode::getInputs();
        for( unsigned i=0, iend=parent_inputs.size() ; i<iend ; ++i )
        {
            core::objectmodel::BaseData* data = static_cast<core::objectmodel::BaseData*>(parent_inputs[i]);

            const defaulttype::AbstractTypeInfo *typeinfo = data->getValueTypeInfo();

            if( typeinfo->name().find("Image") != std::string::npos || typeinfo->name().find("BranchingImage") != std::string::npos )
            {
                if( !data->isSet() )
                {
                    defaulttype::BaseImage* img = static_cast<defaulttype::BaseImage*>( data->beginEditVoidPtr() );
    //                std::cerr<<data->getName()<<" is a Data<Image>\n";
                    // allocate input
                    img->setDimensions( defaulttype::BaseImage::imCoord(1,1,1,1,1) );
                    data->endEditVoidPtr();
                }
            }
        }


        if( this->root ) modeling::initScene(this->root);
    }


    void openScene( const std::string& fileName )
    {
        this->root = modeling::clearScene();
        this->root = down_cast<sofa::simulation::Node>( sofa::simulation::getSimulation()->load(fileName.c_str()).get() );
    }

};



typedef testing::Types<
TestDataEngine< component::engine::ComputeDualQuatEngine<defaulttype::Rigid3Types> >
,TestDataEngine< component::engine::GaussPointContainer >
,TestDataEngine< component::engine::ImageShapeFunctionSelectNode<defaulttype::ImageUC> >
,TestDataEngine< component::engine::ImageDeformation<defaulttype::ImageUC> >
,TestDataEngine< component::engine::MassFromDensity<defaulttype::Affine3Types,defaulttype::ImageUC> >
,TestDataEngine< component::engine::TopologyGaussPointSampler >
,TestDataEngine< component::engine::ShapeFunctionDiscretizer<defaulttype::ImageUC> >
,TestDataEngine< component::engine::ComputeWeightEngine >
,TestDataEngine< component::engine::ImageGaussPointSampler<defaulttype::Image<SReal>,defaulttype::ImageUC> >
> TestTypes; // the types to instanciate.






/////// A BIG MESS TO MAKE A SPECIFIC TEST FOR EACH ENGINE THAT REQUIRES A SCENE

/// standard test does not need a scene and so does nothing particular
template<class TestDataEngineType>
struct SpecificTest
{
    static void run( TestDataEngineType* ) {}
};

/// specific scene for GaussPointContainer
template<>
struct SpecificTest<FlexibleDataEngine_test< TestDataEngine< component::engine::GaussPointContainer > > >
{
    typedef FlexibleDataEngine_test< TestDataEngine< component::engine::GaussPointContainer > > TestDataEngineType;
    static void run( TestDataEngineType* tested )
    {
        tested->m_engineInput->f_inputVolume.setValue( helper::vector<SReal>(1,1) );
    }
};

/// specific scene for ImageDeformation
template<>
struct SpecificTest<FlexibleDataEngine_test< TestDataEngine< component::engine::ImageDeformation<defaulttype::ImageUC> > > >
{
    typedef FlexibleDataEngine_test< TestDataEngine< component::engine::ImageDeformation<defaulttype::ImageUC> > > TestDataEngineType;
    static void run( TestDataEngineType* tested )
    {
        tested->openScene( std::string(FLEXIBLE_TEST_SCENES_DIR) + "/Engine1.scn" );

        simulation::Node::SPtr childNode = tested->root->getChild("child");

        childNode->addObject( tested->m_engine );
        childNode->addObject( tested->m_engineInput );

        tested->m_engineInput->dimensions.setValue( defaulttype::Vector3(1,1,1) );
        tested->m_engineInput->inputImage.setParent( "@../image.inputImage" );
        tested->m_engineInput->inputTransform.setParent( "@../image.inputTransform" );
    }
};



/// specific scene for MassFromDensity
template<>
struct SpecificTest<FlexibleDataEngine_test< TestDataEngine< component::engine::MassFromDensity<defaulttype::Affine3Types,defaulttype::ImageUC> > > >
{
    typedef FlexibleDataEngine_test< TestDataEngine< component::engine::MassFromDensity<defaulttype::Affine3Types,defaulttype::ImageUC> > > TestDataEngineType;
    static void run( TestDataEngineType* tested )
    {
        tested->openScene( std::string(FLEXIBLE_TEST_SCENES_DIR) + "/Engine2.scn" );

        simulation::Node::SPtr childNode = tested->root->getChild("child");

        childNode->addObject( tested->m_engine );
        childNode->addObject( tested->m_engineInput );
    }
};



/// specific scene for TopologyGaussPointSampler
template<>
struct SpecificTest<FlexibleDataEngine_test< TestDataEngine< component::engine::TopologyGaussPointSampler > > >
{
    typedef FlexibleDataEngine_test< TestDataEngine< component::engine::TopologyGaussPointSampler > > TestDataEngineType;
    static void run( TestDataEngineType* tested )
    {
        tested->openScene( std::string(FLEXIBLE_TEST_SCENES_DIR) + "/Engine1.scn" );

        simulation::Node::SPtr childNode = tested->root->getChild("child");

        childNode->addObject( tested->m_engine );
        childNode->addObject( tested->m_engineInput );

        tested->m_engineInput->f_inPosition.setParent( "@../mesh.position" );
    }
};



/// specific scene for ShapeFunctionDiscretizer
template<>
struct SpecificTest<FlexibleDataEngine_test< TestDataEngine< component::engine::ShapeFunctionDiscretizer<defaulttype::ImageUC> > > >
{
    typedef FlexibleDataEngine_test< TestDataEngine< component::engine::ShapeFunctionDiscretizer<defaulttype::ImageUC> > > TestDataEngineType;
    static void run( TestDataEngineType* tested )
    {
        tested->openScene( std::string(FLEXIBLE_TEST_SCENES_DIR) + "/Engine2.scn" );

        simulation::Node::SPtr childNode = tested->root->getChild("child");

        childNode->addObject( tested->m_engine );
        childNode->addObject( tested->m_engineInput );
    }
};


/// specific scene for ComputeWeightEngine
template<>
struct SpecificTest<FlexibleDataEngine_test< TestDataEngine< component::engine::ComputeWeightEngine > > >
{
    typedef FlexibleDataEngine_test< TestDataEngine< component::engine::ComputeWeightEngine > > TestDataEngineType;
    static void run( TestDataEngineType* tested )
    {
        tested->openScene( std::string(FLEXIBLE_TEST_SCENES_DIR) + "/Engine2.scn" );

        tested->root->addObject( tested->m_engine );
        tested->root->addObject( tested->m_engineInput );

        tested->m_engineInput->l_visualModel.setPath( "@/Visual" );
        tested->m_engineInput->l_shapeFunction.setPath( "@/SF" );
        tested->m_engine->l_visualModel.setPath( "@/Visual" );
        tested->m_engine->l_shapeFunction.setPath( "@/SF" );
    }
};




/// specific scene for ImageGaussPointSampler
template<>
struct SpecificTest<FlexibleDataEngine_test< TestDataEngine< component::engine::ImageGaussPointSampler<defaulttype::Image<SReal>,defaulttype::ImageUC> > > >
{
    typedef FlexibleDataEngine_test< TestDataEngine< component::engine::ImageGaussPointSampler<defaulttype::Image<SReal>,defaulttype::ImageUC> > > TestDataEngineType;
    static void run( TestDataEngineType* tested )
    {
        tested->openScene( std::string(FLEXIBLE_TEST_SCENES_DIR) + "/Engine1.scn" );

        simulation::Node::SPtr childNode = tested->root->getChild("child");

        childNode->addObject( tested->m_engine );
        childNode->addObject( tested->m_engineInput );

        tested->m_engineInput->f_index.setParent("@../SF.indices");
        tested->m_engineInput->f_w.setParent("@../SF.weights");
        tested->m_engineInput->f_transform.setParent("@../SF.transform");
        tested->m_engine->targetNumber.setValue(200);
        tested->m_engine->f_method.beginWriteOnly()->setSelectedItem(2); tested->m_engine->f_method.endEdit();
        tested->m_engine->f_order.setValue(1);
    }
};





// ========= Tests to run for each instanciated type
TYPED_TEST_CASE( FlexibleDataEngine_test, TestTypes );

// test number of call to DataEngine::update
TYPED_TEST( FlexibleDataEngine_test, basic_test )
{
    SpecificTest<TestFixture>::run( this );

    this->run_basic_test();
}




}// namespace sofa
