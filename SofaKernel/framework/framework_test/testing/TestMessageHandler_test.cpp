#include <gtest/gtest-spi.h>

#include <sofa/helper/testing/BaseTest.h>
using sofa::helper::testing::BaseTest ;


////////////// IMPLEMENTS A TEST PREDICATE TO VALIDE THAT A THERE IS AT LEAST ONE MESSAGE
/// THE IS EMITTED TO VALDIATE THE BEHAVIOR OF THE FRAMEWORK.
namespace testing {
// Prints a TestPartResult object. (I'm doing this because the 01/03/2018 the operator from gtest
// is not correctly exporter with a GTEST_API_ So I made my own here.
// operator<< in gtest-test-part.cc
std::ostream& operator<<(std::ostream& os, const TestPartResult& result) {
  return os
      << result.file_name() << ":" << result.line_number() << ": "
      << (result.type() == TestPartResult::kSuccess ? "Success" :
          result.type() == TestPartResult::kFatalFailure ? "Fatal failure" :
          "Non-fatal failure") << ":\n"
      << result.message() << std::endl;
}

namespace internal{



// This predicate-formatter checks that 'results' contains a test part
// failure of the given type and that the failure message contains the
// given substring.
AssertionResult AtLeastOneFailure(const char* /* results_expr */,
                              const char* /* type_expr */,
                              const char* /* substr_expr */,
                              const TestPartResultArray& results,
                              TestPartResult::Type type,
                              const string& substr) {
  const std::string expected(type == TestPartResult::kFatalFailure ?
                        "at least 1 fatal failure" :
                        "at least 1 non-fatal failure");
  Message msg;
  if (results.size() == 0) {
    msg << "Expected: " << expected << "\n"
        << "  Actual: " << results.size() << " failures";
    for (int i = 0; i < results.size(); i++) {
      msg << "\n" << results.GetTestPartResult(i);
    }
    return AssertionFailure() << msg;
  }

  const TestPartResult& r = results.GetTestPartResult(0);
  if (r.type() != type) {
    return AssertionFailure() << "Expected: " << expected << "\n"
                              << "  Actual:\n"
                              << r;
  }

  if (strstr(r.message(), substr.c_str()) == NULL) {
    return AssertionFailure() << "Expected: " << expected << " containing \""
                              << substr << "\"\n"
                              << "  Actual:\n"
                              << r;
  }

  return AssertionSuccess();
}

// A helper class for implementing EXPECT_FATAL_FAILURE() and
// EXPECT_NONFATAL_FAILURE().  Its destructor verifies that the given
// TestPartResultArray contains exactly one failure that has the given
// type and contains the given substring.  If that's not the case, a
// non-fatal failure will be generated.
class GTEST_API_ AnyFailureChecker {
 public:
  // The constructor remembers the arguments.
  AnyFailureChecker(const TestPartResultArray* results,
                       TestPartResult::Type type,
                       const string& substr);
  ~AnyFailureChecker();
 private:
  const TestPartResultArray* const results_;
  const TestPartResult::Type type_;
  const string substr_;

  GTEST_DISALLOW_COPY_AND_ASSIGN_(AnyFailureChecker);
};


// The constructor of SingleFailureChecker remembers where to look up
// test part results, what type of failure we expect, and what
// substring the failure message should contain.
AnyFailureChecker:: AnyFailureChecker(
    const TestPartResultArray* results,
    TestPartResult::Type type,
    const string& substr)
    : results_(results),
      type_(type),
      substr_(substr) {}

// The destructor of SingleFailureChecker verifies that the given
// TestPartResultArray contains exactly one failure that has the given
// type and contains the given substring.  If that's not the case, a
// non-fatal failure will be generated.
AnyFailureChecker::~AnyFailureChecker() {
  EXPECT_PRED_FORMAT3(AtLeastOneFailure, *results_, type_, substr_);
}
} // internal
} // testing

#define EXPECT_ATLEAST_ONE_NONFATAL_FAILURE(statement, substr) \
  do {\
    ::testing::TestPartResultArray gtest_failures;\
    ::testing::internal::AnyFailureChecker gtest_checker(\
        &gtest_failures, ::testing::TestPartResult::kNonFatalFailure, \
        (substr));\
    {\
      ::testing::ScopedFakeTestPartResultReporter gtest_reporter(\
          ::testing::ScopedFakeTestPartResultReporter:: \
          INTERCEPT_ONLY_CURRENT_THREAD, &gtest_failures);\
      if (::testing::internal::AlwaysTrue()) { statement; }\
    }\
  } while (::testing::internal::AlwaysFalse())


class TestMessageHandler_test : public BaseTest
{
public:
    void defaultTestBehavior()
    {
        msg_deprecated("HERE") << "This should generate a failure"  ;
        msg_warning("HERE") << "This should generate a failure" ;
        msg_error("HERE") << "This should generate a failure" ;
    }


    void catchingTestBehavior()
    {
        EXPECT_MSG_EMIT(Warning) ;
        EXPECT_MSG_EMIT(Error) ;

        msg_warning("HERE") << "This should not generate a failure"  ;
        msg_error("HERE") << "This should not generate a test falure" ;
    }

    /// THIS TEST SHOULD FAIL.
    void expectAMessageissingBehavior()
    {
        EXPECT_MSG_EMIT(Warning) ;
        EXPECT_MSG_EMIT(Error) ;

        //msg_warning("HERE") << "This should not generate a failure"  ;
        //msg_error("HERE") << "This should not generate a test falure" ;
    }

    void noEmitTestBehavior()
    {
        EXPECT_MSG_NOEMIT(Warning) ;
        EXPECT_MSG_NOEMIT(Error) ;

        msg_warning("HERE") << "This should generate a failure but with line number close to " << __LINE__  ;
        msg_error("HERE") << "This should generate a test falure with line number close to " << __LINE__ ;
    }

    void noEmitIgnoredTestBehavior()
    {
        EXPECT_MSG_NOEMIT(Error) ;
        EXPECT_MSG_NOEMIT(Warning) ;

        IGNORE_MSG(Warning) ;
        msg_warning("HERE") << "This shouldn't generate a failure " ;

        {
            IGNORE_MSG(Error) ;
            msg_error("HERE") << "This shouldn't generate a failure " ;
        }
    }

    void complexTestBehavior()
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
TEST_F(TestMessageHandler_test, defaultTestBehavior)
{
   EXPECT_ATLEAST_ONE_NONFATAL_FAILURE(this->defaultTestBehavior(), "Message") ;
}

/// performing the regression test on every plugins/projects
TEST_F(TestMessageHandler_test, catchingTestBehavior)
{
    this->catchingTestBehavior();
}

/// performing the regression test on every plugins/projects
TEST_F(TestMessageHandler_test, noEmitTestBehavior)
{
    EXPECT_ATLEAST_ONE_NONFATAL_FAILURE(this->noEmitTestBehavior(), "Message") ;
}

/// performing the regression test on every plugins/projects
TEST_F(TestMessageHandler_test, noEmitIgnoredTestBehavior)
{
    this->noEmitIgnoredTestBehavior();
}

/// performing the regression test on every plugins/projects
TEST_F(TestMessageHandler_test, complexTestBehavior)
{
    EXPECT_ATLEAST_ONE_NONFATAL_FAILURE(this->complexTestBehavior(), "Message") ;
}

