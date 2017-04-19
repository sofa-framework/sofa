#include <SofaTest/Sofa_test.h>
using sofa::Sofa_test;

#include <SofaTest/TestMessageHandler.h>
class TestMessageHandler_test : public Sofa_test<>
{
public:
    void windowsProblem1()
    {
        EXPECT_FALSE(true) << "OUI POURUOI ?";
    }

    void windowsProblem2()
    {
        ADD_FAILURE_AT("fichierbidon.cpp",3) << "POURQUOI ?";
    }

    void defaultTestBehaviorSHOULDFAIL()
    {
        msg_deprecated("HERE") << "This should generate a failure" ;
        msg_warning("HERE") << "This should generate a failure"  ;
        msg_error("HERE") << "This should generate a failure" ;
        ADD_FAILURE_AT("fichierbidon.cpp",5) << "POURQUOI ?";
    }


    void catchingTestBehaviorSHOULDNOTFAIL()
    {
        EXPECT_MSG_EMIT(Warning) ;
        EXPECT_MSG_EMIT(Error) ;

        msg_warning("HERE") << "This should not generate a failure"  ;
        msg_error("HERE") << "This should not generate a test falure" ;
    }

    /// THIS TEST SHOULD FAIL.
    void expectAMessageissingBehaviorSHOULDFAIL()
    {
        EXPECT_MSG_EMIT(Warning) ;
        EXPECT_MSG_EMIT(Error) ;

        //msg_warning("HERE") << "This should not generate a failure"  ;
        //msg_error("HERE") << "This should not generate a test falure" ;
    }

    void noEmitTestBehaviorSHOULDFAIL()
    {
        EXPECT_MSG_NOEMIT(Warning) ;
        EXPECT_MSG_NOEMIT(Error) ;

        msg_warning("HERE") << "This should generate a failure but with line number close to " << __LINE__  ;
        msg_error("HERE") << "This should generate a test falure with line number close to " << __LINE__ ;
    }

    void complexTestBehaviorSHOULDFAIL()
    {
        {
            EXPECT_MSG_EMIT(Warning) ;
            EXPECT_MSG_EMIT(Error) ;

            //msg_warning("HERE") << "This should generate a failure"  ;
            //msg_error("HERE") << "This should generate a test failure" ;
            {
                EXPECT_MSG_NOEMIT(Error) ;
                msg_error("HERE") << "This should generate a test failure" ;
            }
        }

        {
            EXPECT_MSG_NOEMIT(Warning) ;
            EXPECT_MSG_NOEMIT(Error) ;

            msg_warning("HERE") << "This should generate a failure"  ;
            msg_error("HERE") << "This should generate a test falure" ;
        }

    }
};


/// performing the regression test on every plugins/projects
TEST_F(TestMessageHandler_test, windowsProblem1)
{
    this->windowsProblem1();
}


/// performing the regression test on every plugins/projects
TEST_F(TestMessageHandler_test, windowsProblem2)
{
    this->windowsProblem2();
}


/// performing the regression test on every plugins/projects
TEST_F(TestMessageHandler_test, defaultTestBehaviorSHOULDFAIL)
{
    this->defaultTestBehaviorSHOULDFAIL();
}

/// performing the regression test on every plugins/projects
TEST_F(TestMessageHandler_test, catchingTestBehaviorSHOULDNOTFAIL)
{
    this->catchingTestBehaviorSHOULDNOTFAIL();
}

/// performing the regression test on every plugins/projects
TEST_F(TestMessageHandler_test, noEmitTestBehaviorSHOULDFAIL)
{
    this->noEmitTestBehaviorSHOULDFAIL();
}

/// performing the regression test on every plugins/projects
TEST_F(TestMessageHandler_test, complexTestBehaviorSHOULDFAIL)
{
    this->complexTestBehaviorSHOULDFAIL();
}

