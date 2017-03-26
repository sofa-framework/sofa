#include <SofaTest/Sofa_test.h>
using sofa::Sofa_test;

#include <SofaTest/TestMessageHandler.h>
using sofa::helper::logging::GtestMessageHandler ;

/// We can define a default policy for a complete class this way so that if not more
/// expectation are given this generates test failures.
class Sofa_test2 : public Sofa_test<float>
{
    EXPECT_MSG_NOEMIT(Error) ;
    EXPECT_MSG_NOEMIT(Warning) ;
    EXPECT_MSG_NOEMIT(Deprecated) ;
};


class TestMessageHandler_test : public Sofa_test2
{
public:
    void defaultTestBehaviorSHOULDFAIL()
    {
        msg_deprecated("HERE") << "This should generate a failure" ;
        msg_warning("HERE") << "This should generate a failure"  ;
        msg_error("HERE") << "This should generate a failure" ;
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

