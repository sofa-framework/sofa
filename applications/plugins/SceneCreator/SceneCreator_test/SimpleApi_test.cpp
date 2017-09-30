#include <SofaTest/Sofa_test.h>

#include <SceneCreator/SimpleApi.h>
using namespace sofa ;
using namespace sofa::simpleapi ;

class SimpleApi_test : public sofa::Sofa_test<>
{
public:
    bool test();
};

bool SimpleApi_test::test()
{
    using namespace sofa::simpleapi::components ;

    //Node::SPtr root = createRootNode("root") ;
/*
    createObject(root, OglShader, {{}} ) ;
    createObject(root, MechanicalModel, {{
                                             BaseObject::name, "damien"
                                         }});
*/
    return true;
}

TEST_F(SimpleApi_test, createCubeFailed )
{
    ASSERT_TRUE( test() );
}
