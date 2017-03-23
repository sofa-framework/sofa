#include <SofaTest/Sofa_test.h>
using sofa::Sofa_test;

#include <SofaTest/TestMessageHandler.h>
using sofa::helper::logging::GtestMessageHandler ;

/// We can define a default policy for a complete class this way so that if not more
/// expectation are given this generates test failures.
class Sofa_test2 : public Sofa_test<float>
{
    EXPECT_MSG_NOEMIT_V2(Error) ;
    EXPECT_MSG_NOEMIT_V2(Warning) ;
    EXPECT_MSG_NOEMIT_V2(Deprecated) ;
};


class TestMessageHandler_test : public Sofa_test2
{
public:
    void defaultTestBehavior()
    {
        msg_deprecated("HERE") << "This should generate a failure" ;
        msg_warning("HERE") << "This should generate a failure"  ;
        msg_error("HERE") << "This should generate a failure" ;
    }

    void catchingTestBehavior()
    {
        EXPECT_MSG_EMIT_V2(Warning) ;
        EXPECT_MSG_EMIT_V2(Error) ;

        msg_warning("HERE") << "This should not generate a failure"  ;
        msg_error("HERE") << "This should not generate a test falure" ;
    }

    /// THIS TEST SHOULD FAIL.
    void expectAMessageissingBehavior()
    {
        EXPECT_MSG_EMIT_V2(Warning) ;
        EXPECT_MSG_EMIT_V2(Error) ;

        //msg_warning("HERE") << "This should not generate a failure"  ;
        //msg_error("HERE") << "This should not generate a test falure" ;
    }

    void noEmitTestBehavior()
    {
        EXPECT_MSG_NOEMIT_V2(Warning) ;
        EXPECT_MSG_NOEMIT_V2(Error) ;

        msg_warning("HERE") << "This should generate a failure but with line number close to " << __LINE__  ;
        msg_error("HERE") << "This should generate a test falure with line number close to " << __LINE__ ;
    }

    void complexTestBehavior()
    {
        {
            EXPECT_MSG_EMIT_V2(Warning) ;
            EXPECT_MSG_EMIT_V2(Error) ;

            //msg_warning("HERE") << "This should generate a failure"  ;
            //msg_error("HERE") << "This should generate a test failure" ;
            {
                EXPECT_MSG_NOEMIT_V2(Error) ;
                msg_error("HERE") << "This should generate a test failure" ;
            }
        }

        {
            EXPECT_MSG_NOEMIT_V2(Warning) ;
            EXPECT_MSG_NOEMIT_V2(Error) ;

            msg_warning("HERE") << "This should generate a failure"  ;
            msg_error("HERE") << "This should generate a test falure" ;
        }

    }
};

/// performing the regression test on every plugins/projects
TEST_F(TestMessageHandler_test, defaultTestBehavior)
{
    this->defaultTestBehavior();
}

/// performing the regression test on every plugins/projects
TEST_F(TestMessageHandler_test, catchingTestBehavior)
{
    this->catchingTestBehavior();
}

/// performing the regression test on every plugins/projects
TEST_F(TestMessageHandler_test, noEmitTestBehavior)
{
    this->noEmitTestBehavior();
}

/// performing the regression test on every plugins/projects
TEST_F(TestMessageHandler_test, complexTestBehavior)
{
    this->complexTestBehavior();
}

