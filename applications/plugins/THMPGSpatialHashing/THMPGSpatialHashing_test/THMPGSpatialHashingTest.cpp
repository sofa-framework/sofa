#include <SofaTest/BroadPhase_test.h>
#include "../THMPGSpatialHashing.h"

#include <SofaMiscCollision/initSofaMiscCollision.h>

typedef BroadPhaseTest<sofa::component::collision::THMPGSpatialHashing> Teschner;
TEST_F(Teschner, rand_sparse_test ) 
{
    sofa::component::initSofaMiscCollision();
    ASSERT_TRUE( randSparse());
}

TEST_F(Teschner, rand_dense_test ) { ASSERT_TRUE( randDense()); }
