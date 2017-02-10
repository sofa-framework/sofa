#include <SofaPython/PythonScriptEvent.h>

#include "Python_test.h"

#include <sofa/helper/system/PluginManager.h>
#include <sofa/simulation/Simulation.h>

#include <sofa/helper/logging/Messaging.h>


namespace sofa {



Python_test::Python_test()
{
    static const std::string plugin = "SofaPython";
    sofa::helper::system::PluginManager::getInstance().loadPlugin(plugin);
}



void Python_test::run( const Python_test_data& data ) {

    msg_info("Python_test") << "running " << data.filepath;

    {
        // Check the file exists
        std::ifstream file(data.filepath.c_str());
        bool scriptFound = file.good();
        ASSERT_TRUE(scriptFound);
    }

    ASSERT_TRUE( loader.loadTestWithArguments(data.filepath.c_str(),data.arguments) );

}



////////////////////////////////////////////////////
////////////////////////////////////////////////////
////////////////////////////////////////////////////


struct Listener : core::objectmodel::BaseObject {

    Listener() {
        f_listening = true;
    }

    virtual void handleEvent(core::objectmodel::Event * event) {
        if (core::objectmodel::PythonScriptEvent::checkEventType(event)
              || core::objectmodel::ScriptEvent::checkEventType(event) )
       {
            core::objectmodel::ScriptEvent* e = static_cast<core::objectmodel::ScriptEvent*>(event);
            std::string name = e->getEventName();
            if( name == "success" ) {
                throw Python_scene_test::result(true);
            } else if (name == "failure") {
                throw Python_scene_test::result(false);
            }
        }
    }

};


static simulation::Node::SPtr instance;

void Python_scene_test::run( const Python_test_data& data ) {

    msg_info("Python_scene_test") << "running "<< data.filepath;

    {
        // Check the file exists
        std::ifstream file(data.filepath.c_str());
        bool scriptFound = file.good();
        ASSERT_TRUE(scriptFound);
    }

    simulation::Node::SPtr root = loader.loadSceneWithArguments(data.filepath.c_str(),data.arguments);
    instance = root;
    
	root->addObject( new Listener );

	simulation::getSimulation()->init(root.get());

	try {
		while(root->isActive()) {
			simulation::getSimulation()->animate(root.get(), root->getDt());
		}
	} catch( const result& test_result ) {
		ASSERT_TRUE(test_result.value);
	}
}



} // namespace sofa


extern "C" {

    void finish() {
	if(sofa::instance) sofa::instance->setActive(false);
    }

    void expect_true(bool test, const char* msg) {
	EXPECT_TRUE(test) << msg;
    }
    
    void assert_true(bool test, const char* msg) {
        auto trigger = [&] {
            ASSERT_TRUE(test) << msg;
        };

        trigger();
        if(!test) finish();
    }
    

}
