#include "InitLinearMapping.h"
#include <plugins/Flexible/shapeFunction/VoronoiShapeFunction.h>
#include <plugins/Flexible/deformationMapping/BaseDeformationMapping.h>


namespace sofa
{

    template< typename Real >
        InitLinearMapping_test<Real>::InitLinearMapping_test()
        {
            //Allow to load python scene
            std::string plugin = "SofaPython";
            sofa::helper::system::PluginManager::getInstance().loadPlugin(plugin);
            loader=simulation::SceneLoaderFactory::getInstance()->getEntryFileExtension("py");
            if( !loader ) throw std::logic_error("can't get scene loader, is SofaPython available ?");
        }

    template< typename Real >
        InitLinearMapping_test<Real>::~InitLinearMapping_test(){}

    template< typename Real >
        bool InitLinearMapping_test<Real>::testInitFlexible()
        {
            std::string filepath= std::string(FLEXIBLE_TEST_SCENES_DIR)+"/python/reInitMapping.py";
            std::ifstream file(filepath.c_str());
            bool scriptFound = file.good();
            if(scriptFound==false)
            {
                std::cout << "Reason : Script not found" << std::endl; 
                std::cout << "Path given : " << filepath.c_str() << std::endl;
                return false;
            }

            simulation::Node::SPtr root = loader->load(filepath.c_str());

            root->addObject( new Listener );

            simulation::getSimulation()->init(root.get());

            bool res = false;
            try
            {
                simulation::getSimulation()->animate(root.get(), root->getDt());
            }
            catch( bool testResult )
            {
                res = testResult;   
            }
            return res; 
        }

    typedef testing::Types<float> DataTypes;
    TYPED_TEST_CASE(InitLinearMapping_test, DataTypes);

    TYPED_TEST(InitLinearMapping_test,reInitFlexible)
    {
        EXPECT_TRUE( this->testInitFlexible() );
    }
}//sofa namespace

