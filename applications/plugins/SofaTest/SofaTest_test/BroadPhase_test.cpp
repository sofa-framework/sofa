#include "stdafx.h"
#include "../BroadPhase_test.h"


typedef BroadPhaseTest<sofa::component::collision::BruteForceDetection> Brut;
TEST_F(Brut, rand_sparse_test ) { ASSERT_TRUE( randSparse()); }
TEST_F(Brut, rand_dense_test ) { ASSERT_TRUE( randDense()); }

typedef BroadPhaseTest<sofa::component::collision::IncrSAP> IncrSAPTest;
TEST_F(IncrSAPTest, rand_sparse_test ) { ASSERT_TRUE( randSparse()); }
TEST_F(IncrSAPTest, rand_dense_test ) { ASSERT_TRUE( randDense()); }

typedef BroadPhaseTest<sofa::component::collision::DirectSAP> DirectSAPTest;
TEST_F(DirectSAPTest, rand_sparse_test ) { ASSERT_TRUE( randSparse()); }
TEST_F(DirectSAPTest, rand_dense_test ) { ASSERT_TRUE( randDense()); }
