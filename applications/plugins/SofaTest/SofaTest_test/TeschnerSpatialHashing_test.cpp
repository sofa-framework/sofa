#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <sofa/helper/UnitTest.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/system/PluginManager.h>

//#include <sofa/simulation/tree/TreeSimulation.h>
#ifdef SOFA_HAVE_DAG
#include <sofa/simulation/graph/DAGSimulation.h>
#endif
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/xml/initXml.h>

#include <sofa/gui/GUIManager.h>
#include <sofa/gui/Main.h>
#include <sofa/helper/system/FileRepository.h>

#include <sofa/component/init.h>

//Using double by default, if you have SOFA_FLOAT in use in you sofa-default.cfg, then it will be FLOAT.
#include <sofa/component/typedef/Sofa_typedef.h>
//#include <plugins/SceneCreator/SceneCreator.h>


#include <sofa/simulation/tree/GNode.h>
#include <sofa/component/collision/TeschnerHashTable.h>

#include <PrimitiveCreation.h>
#include "Sofa_test.h"


namespace sofa {

struct TestTeschnerHashTable : public Sofa_test<SReal>{
    bool test1();
    bool test2();

};


bool TestTeschnerHashTable::test1(){
    TeschnerHashTable table;

    table.resize(5);

    sofa::core::CollisionElementIterator e0(0x0,0);
    sofa::core::CollisionElementIterator e1(0x0,1);
    sofa::core::CollisionElementIterator e2(0x0,2);

    table(0,5,-1).add(e0,0);
    table(0,5,-1).add(e1,0);
    table(1,0,-2).add(e2,0);

    std::vector<core::CollisionElementIterator> & collisionelems = table(0,5,-1).getCollisionElems();

    bool e0found = false;
    bool e1found = false;
    for(unsigned int i = 0 ; i < collisionelems.size() ; ++i){
        if(collisionelems[i].getIndex() == 0)
            e0found = true;
        if(collisionelems[i].getIndex() == 1)
            e1found = true;
    }

    return e0found && e1found;
}


bool TestTeschnerHashTable::test2(){
    TeschnerHashTable table;

    table.resize(5);

    sofa::core::CollisionElementIterator e0(0x0,0);
    sofa::core::CollisionElementIterator e1(0x0,1);
    sofa::core::CollisionElementIterator e2(0x0,2);

    table(0,5,-1).add(e0,0);
    table(0,5,-1).add(e1,1);
    table(1,0,-2).add(e2,0);

    std::vector<core::CollisionElementIterator> & collisionelems = table(0,5,-1).getCollisionElems();

    bool e0found = false;
    bool e1found = false;
    for(unsigned int i = 0 ; i < collisionelems.size() ; ++i){
        if(collisionelems[i].getIndex() == 0)
            e0found = true;
        if(collisionelems[i].getIndex() == 1)
            e1found = true;
    }

    return (!e0found) && e1found;
}


//TEST_F(TestTeschnerHashTable, test_1 ) { ASSERT_TRUE( test1()); }
//TEST_F(TestTeschnerHashTable, test_2 ) { ASSERT_TRUE( test2()); }

//    sofa::component::collision::OBB movingOBB;
//    sofa::component::collision::OBB staticOBB;
}



