#include <SofaTest/Sofa_test.h>
#include <sofa/helper/BackTrace.h>

#include <SofaSimulationGraph/DAGSimulation.h>
using sofa::simulation::Simulation ;
using sofa::simulation::Node ;
using sofa::simulation::setSimulation ;
using sofa::core::objectmodel::New ;
using sofa::core::objectmodel::BaseData ;
using sofa::simulation::graph::DAGSimulation;

#include <SofaGeneralEngine/MeshROI.h>
using sofa::component::engine::MeshROI ;

#include <sofa/core/visual/VisualParams.h>
using sofa::core::visual::VisualParams;

#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;

using std::vector;
using std::string;


namespace sofa
{

template <typename _DataTypes>
struct MeshROI_test : public Sofa_test<typename _DataTypes::Real>,
        MeshROI<_DataTypes>
{
    typedef MeshROI<_DataTypes> ThisClass;
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::Coord Coord;


    Simulation* m_simu;
    Node::SPtr m_node;
    typename ThisClass::SPtr m_thisObject;

    void SetUp()
    {
        setSimulation(m_simu = new DAGSimulation());
        m_node = m_simu->createNewGraph("root");
        m_thisObject = New<ThisClass >() ;
        m_node->addObject(m_thisObject) ;
    }


    // It is important to freeze what are the available Data field
    // of a component and rise warning/errors when some one removed.
    void attributesTests()
    {
        m_thisObject->setName("myname") ;
        EXPECT_TRUE(m_thisObject->getName() == "myname") ;

        // List of the supported attributes the user expect to find
        // This list needs to be updated if you add an attribute.
        vector<string> attrnames = {
            "box",
            "position", "edges",  "triangles", "tetrahedra",
            "ROIposition", "ROIedges", "ROItriangles",
            "computeEdges", "computeTriangles", "computeTetrahedra",
            "indices", "edgeIndices", "triangleIndices", "tetrahedronIndices",
            "pointsInROI", "edgesInROI", "trianglesInROI", "tetrahedraInROI",
            "pointsOutROI", "edgesOutROI", "trianglesOutROI", "tetrahedraOutROI",
            "drawBox", "drawPoints", "drawEdges", "drawTriangles", "drawTetrahedra",
            "drawSize",
            "doUpdate"
        };

        for(auto& attrname : attrnames)
            EXPECT_NE( m_thisObject->findData(attrname), nullptr ) << "Missing attribute with name '" << attrname << "'." ;

        return ;
    }


    void initTests()
    {
        EXPECT_NO_THROW(m_thisObject->init());

        VisualParams* vparams = VisualParams::defaultInstance();
        vparams->displayFlags().setShowBehaviorModels(true);
        m_thisObject->d_drawSize.setValue(1);

        EXPECT_NO_THROW(m_thisObject->draw(vparams));
    }


    void computeBoundingBoxTest()
    {
        string scene =
                "<?xml version='1.0'?>"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >       "
                "   <Node name='Level 1'>                                          "
                "       <MeshObjLoader name='loader' filename='mesh/dragon.obj'/>  "
                "       <MeshROI template='Vec3d' name='MeshROI'/>                 "
                "   </Node>                                                        "
                "</Node>                                                           " ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                          scene.c_str(),
                                                          scene.size()) ;

        root->getChild("Level 1")->getObject("MeshROI")->init();
        //Bounding box from Meshlab
        EXPECT_EQ(root->getChild("Level 1")->getObject("MeshROI")->findData("box")->getValueString(),"-11.4529 -7.38909 -5.04461 11.4121 8.31288 5.01514");
    }

};

using testing::Types;
typedef Types<Vec3Types> DataTypes;

TYPED_TEST_CASE(MeshROI_test, DataTypes);

TYPED_TEST(MeshROI_test, attributesTests) {
    ASSERT_NO_THROW(this->attributesTests()) ;
}

TYPED_TEST(MeshROI_test, initTests) {
    ASSERT_NO_THROW(this->initTests()) ;
}

TYPED_TEST(MeshROI_test, computeBoundingBoxTest) {
    ASSERT_NO_THROW(this->computeBoundingBoxTest()) ;
}

}
