#include <plugins/SofaPython/PythonScriptController.h>
#include <plugins/SofaPython/ScriptEvent.h>

#include <plugins/SofaTest/Sofa_test.h>
#include <sofa/simulation/common/SceneLoaderFactory.h>
#include <sofa/helper/system/PluginManager.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <fstream>

#include <sofa/simulation/common/Node.h>

///Load a python scene to perform test
///The python scene contains a scriptController sending event of success or failure
///A Listener is added to the scene to receive the scriptController signal as the test result

///File Test.py from Compliant plugin is used as an helper

namespace sofa
{

    namespace simulation
    {
        class SceneLoader;
    }

    template< typename Real >
        class InitLinearMapping_test : public Sofa_test<Real>
    {
        public :
            InitLinearMapping_test();
            ~InitLinearMapping_test();
        protected:
            simulation::SceneLoader* loader;
        public:
            bool testInitFlexible();
            bool testVoronoiShapeFunction();
            bool testPrintLog();
    };

    struct Listener : core::objectmodel::BaseObject
    {
        Listener(){f_listening=true;}
        virtual void handleEvent(core::objectmodel::Event* event)
        {
            typedef core::objectmodel::ScriptEvent event_type;
            if(event_type* e = dynamic_cast<event_type*>(event))
            {
                std::string name = e->getEventName();
                if(name=="success"){throw true;}
                else if(name=="failure"){throw false;}
            }
        }
    };

}//sofa namespace

