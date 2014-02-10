#include <plugins/SofaPython/PythonScriptController.h>
#include <plugins/SofaPython/ScriptEvent.h>

#include "Python_test.h"

#include <sofa/simulation/common/SceneLoaderFactory.h>
#include <sofa/helper/system/PluginManager.h>


#include <sofa/simulation/graph/DAGSimulation.h>

#include "../utils/edit.h"

namespace sofa {

Python_test::Python_test() 
{
	std::string plugin = "SofaPython";
	sofa::helper::system::PluginManager::getInstance().loadPlugin(plugin);

	
	loader = simulation::SceneLoaderFactory::getInstance()->getEntryFileExtension("py");
	
	if( !loader ) throw std::logic_error("can't get scene loader, is SofaPython available ?");
}


struct Listener : core::objectmodel::BaseObject {
	
	Listener() {
		f_listening = true;
	}

	virtual void handleEvent(core::objectmodel::Event * event) {
		typedef core::objectmodel::ScriptEvent event_type;
		
		if (event_type* e = dynamic_cast<event_type *>(event)) {

			std::string name = e->getEventName();
			if( name == "success" ) {
				throw Python_test::result(true);
			} else if (name == "failure") {
				throw Python_test::result(false);
			} 
		}
	}
	
};




std::string Python_test::path() {
	std::string full = sofa_tostring(SOFA_SRC_DIR);

	full += "/applications/plugins/Compliant/Compliant_test/";
	
	return full;
}

void Python_test::run(const char* filename) {

	// adapt filename
	std::string full = path();
	full += filename;
	
	// std::cerr << "running: " << full << std::endl;

	simulation::Node::SPtr root = loader->load(full.c_str());
	
    root->addObject( new Listener );

	simulation::getSimulation()->init(root.get());

	try {
		while(true) {
			simulation::getSimulation()->animate(root.get(), root->getDt());
		}
	} catch( const result& test_result ) {
		ASSERT_TRUE(test_result.value);
	}
}


TEST_P(Python_test, Run) {
	run( GetParam() );
}

Python_test::~Python_test() {

}




}
