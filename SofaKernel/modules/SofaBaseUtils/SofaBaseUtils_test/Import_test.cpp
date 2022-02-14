#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest;

#include <sofa/simulation/Node.h>
#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;
using sofa::simulation::Node ;

using sofa::core::execparams::defaultInstance; 

namespace sofa
{

struct Import_test : public BaseTest
{
    void testFromImport()
    {
        EXPECT_MSG_NOEMIT(Error, Warning);

        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
                 "   <Import fromPlugin='Sofa.Component.SceneUtility' components='RequiredPlugin'/>            \n"
                 "</Node>                                                                        \n" ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                          scene.str().c_str(),
                                                          scene.str().size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(sofa::core::execparams::defaultInstance()) ;
    }

    void testFromImportAs()
    {
        EXPECT_MSG_NOEMIT(Error, Warning);

        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
                 "   <Import fromPlugin='Sofa.Component.SceneUtility' components='RequiredPlugin' as='AliasedName'/> \n"
                 "   <AliasedName name='Sofa.Component.SceneUtility'/>                                         \n"
                 "</Node>                                                                        \n" ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                          scene.str().c_str(),
                                                          scene.str().size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(sofa::core::execparams::defaultInstance()) ;
    }

    void testFromInvalidPluginName()
    {
        EXPECT_MSG_EMIT(Error);

        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
                 "   <Import fromPlugin='NotValidPlugin' components='*'/>                        \n"
                 "</Node>                                                                        \n" ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                          scene.str().c_str(),
                                                          scene.str().size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(sofa::core::execparams::defaultInstance()) ;
    }
};

TEST_F(Import_test, testFromImportSyntax ) { testFromImport(); }
TEST_F(Import_test, testFromImportAsSyntax ) { testFromImportAs(); }
TEST_F(Import_test, testFromInvalidPluginName ) { testFromInvalidPluginName(); }

}
