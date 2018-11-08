
#include "Regression_test.h"


namespace sofa {


/// performing the regression test on every plugins/projects

INSTANTIATE_TEST_CASE_P(regression,
    Regression_test,
    ::testing::ValuesIn(regression_tests.listScenes),
    Regression_test::getTestName);

TEST_P(Regression_test, all_tests) { runRegressionScene(GetParam()); }


} // namespace sofa
