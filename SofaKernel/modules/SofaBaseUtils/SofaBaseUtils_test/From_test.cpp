#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest;

#include <sofa/simulation/Node.h>
#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;
using sofa::simulation::Node ;

using sofa::core::execparams::defaultInstance; 

namespace sofa
{

struct From_test : public BaseTest
{
    void testFromImport()
    {
        EXPECT_MSG_NOEMIT(Error, Warning);

        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
                 "   <From plugin='SofaBaseUtils' import='RequiredPlugin'/>                      \n"
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
                 "   <From plugin='SofaBaseUtils' import='RequiredPlugin' as='AliasedName'/>     \n"
                 "   <AliasedName name='SofaBaseUtils'/>                                         \n"
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
                 "   <From plugin='NotValidPlugin' import='*'/>                                  \n"
                 "   <AliasedName name='SofaBaseUtils'/>                                         \n"
                 "</Node>                                                                        \n" ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                          scene.str().c_str(),
                                                          scene.str().size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(sofa::core::execparams::defaultInstance()) ;
    }
};

TEST_F(From_test, testFromImportSyntax ) { testFromImport(); }
TEST_F(From_test, testFromImportAsSyntax ) { testFromImportAs(); }
TEST_F(From_test, testFromInvalidPluginName ) { testFromInvalidPluginName(); }

}
