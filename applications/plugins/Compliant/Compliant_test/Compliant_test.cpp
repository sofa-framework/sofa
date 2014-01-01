#include "Assembly_test.h"

namespace sofa {

/** \page Page_CompliantTestSuite Compliant plugin test suite
  This test suite uses the google test Framework

  The test suite is run from file Compliant_test.cpp .
  The main() function is actually in an external library.

 Currently all the tests are based on the unit test methods defined in class Assembly_test.

  */

TEST_F( Assembly_test, testHardString )
{
    unsigned numParticles=3;
    ::testing::Message() << "Assembly_test: hard string of " << numParticles << " particles";
    testHardString(numParticles);
    ASSERT_TRUE(matricesAreEqual( expected.M, complianceSolver->M() ));
    ASSERT_TRUE(matricesAreEqual( expected.P, complianceSolver->P() ));
    ASSERT_TRUE(matricesAreEqual( expected.J, complianceSolver->J() ));
    ASSERT_TRUE(matricesAreEqual( expected.C, complianceSolver->C() ));
    ASSERT_TRUE(vectorsAreEqual( expected.dv, complianceSolver->getDv() ));
    ASSERT_TRUE(vectorsAreEqual( expected.lambda, complianceSolver->getLambda() ));
    //    cout<<"testHardString results compared"<< endl;
}
TEST_F( Assembly_test, testAttachedHardString )
{
    unsigned numParticles=3;
    ::testing::Message() << "Assembly_test: hard string of " << numParticles << " particles attached using a projective constraint (FixedConstraint)";
    testAttachedHardString(numParticles);
    ASSERT_TRUE(matricesAreEqual( expected.M, complianceSolver->M() ));
    ASSERT_TRUE(matricesAreEqual( expected.P, complianceSolver->P() ));
    ASSERT_TRUE(matricesAreEqual( expected.J, complianceSolver->J() ));
    ASSERT_TRUE(matricesAreEqual( expected.C, complianceSolver->C() ));
    ASSERT_TRUE(vectorsAreEqual( expected.dv, complianceSolver->getDv() ));
    ASSERT_TRUE(vectorsAreEqual( expected.lambda, complianceSolver->getLambda() ));
    //    cout<<"testAttachedHardString results compared"<< endl;
}
TEST_F( Assembly_test, testConstrainedHardString )
{
    unsigned numParticles=4;
    ::testing::Message() << "Assembly_test: hard string of " << numParticles << " particles attached using a distance constraint";
    testConstrainedHardString(numParticles);
    ASSERT_TRUE(matricesAreEqual( expected.M, complianceSolver->M() ));
    ASSERT_TRUE(matricesAreEqual( expected.P, complianceSolver->P() ));
    ASSERT_TRUE(matricesAreEqual( expected.J, complianceSolver->J() ));
    ASSERT_TRUE(matricesAreEqual( expected.C, complianceSolver->C() ));
    ASSERT_TRUE(vectorsAreEqual( expected.dv, complianceSolver->getDv() ));
    ASSERT_TRUE(vectorsAreEqual( expected.lambda, complianceSolver->getLambda() ));
    //    cout<<"testConstrainedHardString results compared"<< endl;
}
TEST_F( Assembly_test, testExternallyConstrainedHardString )
{
    unsigned numParticles=2;
    ::testing::Message() << "Assembly_test: hard string of " << numParticles << " particles attached using a constraint with an out-of-scope particle";
    testExternallyConstrainedHardString(numParticles);
    ASSERT_TRUE(matricesAreEqual( expected.M, complianceSolver->M() ));
    ASSERT_TRUE(matricesAreEqual( expected.P, complianceSolver->P() ));
    ASSERT_TRUE(matricesAreEqual( expected.J, complianceSolver->J() ));
    ASSERT_TRUE(matricesAreEqual( expected.C, complianceSolver->C() ));
    ASSERT_TRUE(vectorsAreEqual( expected.dv, complianceSolver->getDv() ));
    ASSERT_TRUE(vectorsAreEqual( expected.lambda, complianceSolver->getLambda() ));
    //    //    cout<<"testExternallyConstrainedHardString results compared"<< endl;
}
TEST_F( Assembly_test, testAttachedConnectedHardStrings )
{
    unsigned numParticles=2;
    ::testing::Message() << "Assembly_test: hard strings of " << numParticles << " particles connected using a MultiMapping";
    testAttachedConnectedHardStrings(numParticles);
    ASSERT_TRUE(matricesAreEqual( expected.M, complianceSolver->M() ));
    ASSERT_TRUE(matricesAreEqual( expected.P, complianceSolver->P() ));
    ASSERT_TRUE(matricesAreEqual( expected.J, complianceSolver->J() ));
    ASSERT_TRUE(matricesAreEqual( expected.C, complianceSolver->C() ));
    ASSERT_TRUE(vectorsAreEqual( expected.dv, complianceSolver->getDv() ));
    ASSERT_TRUE(vectorsAreEqual( expected.lambda, complianceSolver->getLambda() ));
    //    cout<<"testAttachedConnectedHardString results compared"<< endl;
}
TEST_F( Assembly_test, testRigidConnectedToString )
{
    unsigned numParticles=2;
    ::testing::Message() << "Assembly_test: hard string of " << numParticles << " particles connected to a rigid";
    testRigidConnectedToString(numParticles);
//    cerr<<"expected.M = " << endl << expected.M << endl;
//    cerr<<"complianceSolver->M() = " << endl << complianceSolver->M() << endl;
    ASSERT_TRUE(matricesAreEqual( expected.M, complianceSolver->M() ));
    ASSERT_TRUE(matricesAreEqual( expected.P, complianceSolver->P() ));
    ASSERT_TRUE(matricesAreEqual( expected.J, complianceSolver->J() ));
    ASSERT_TRUE(matricesAreEqual( expected.C, complianceSolver->C() ));
    ASSERT_TRUE(vectorsAreEqual( expected.dv, complianceSolver->getDv() ));
    ASSERT_TRUE(vectorsAreEqual( expected.lambda, complianceSolver->getLambda() ));
    //    cout<<"testRigidConnectedToString results compared"<< endl;

    //    cout<<"all tests done" << endl;
}


} // sofa



