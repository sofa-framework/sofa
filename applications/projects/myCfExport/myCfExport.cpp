/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <iostream>
#include <fstream>
#include <ctime>

#include <sofa/helper/ArgumentParser.h>
#include <sofa/helper/system/PluginManager.h>

#include <SofaComponentBase/initComponentBase.h>
#include <SofaComponentCommon/initComponentCommon.h>
#include <SofaComponentGeneral/initComponentGeneral.h>
#include <SofaComponentAdvanced/initComponentAdvanced.h>
#include <SofaComponentMisc/initComponentMisc.h>

#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
#include <SofaSimulationGraph/init.h>
#include <SofaSimulationGraph/DAGSimulation.h>

#include <sofa/helper/Factory.h>
#include <sofa/helper/BackTrace.h>
#include <SofaExporter/WriteState.h>

#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/objectmodel/Base.h>

using std::cerr;
using std::endl;
using std::cout;



namespace sofa {


class MyCfExportVisitor : public sofa::simulation::Visitor
{
public:
	// type def
	typedef helper::vector<sofa::core::objectmodel::BaseData*> VecData;

    MyCfExportVisitor(const sofa::core::ExecParams* params)
        : Visitor(params)
    {}

    virtual void processObject(simulation::Node* /*node*/, core::objectmodel::BaseObject* o) 
	{
		// Test
        cout <<"object --> " << o->getName() << " || of  type --> " << o->getTypeName() << " traversed" << endl;

		// Parent node
		cout << "Parent node --> " << o->getContext()->getName() << endl;

		// Get the object with its list of attributes
		const VecData data(o->getDataFields());
		cout << "Attributes --> " << endl;
		for(unsigned int i=0; i<data.size(); ++i)
		{
			cout << "dataName --> " <<  data[i]->getName() << endl;
			cout << "dataValue --> " <<  data[i]->getValueString() << endl;
		}
    }

    virtual Result processNodeTopDown( simulation::Node* node)
    {
        cout <<"node " << node->getName() << " traversed ==================================" << endl;
        for_each(this, node, node->object, &MyCfExportVisitor::processObject);
        return RESULT_CONTINUE;
    }


protected:
};

}





void apply(std::string &input)
{
    cout<<"\n****   Processing scene:"<< input<< endl;

    // --- Create simulation graph ---
    sofa::simulation::Node::SPtr groot = sofa::core::objectmodel::SPtr_dynamic_cast<sofa::simulation::Node>( sofa::simulation::getSimulation()->load(input.c_str()));
    if (groot==NULL)
    {
        cerr << "================== Error, unable to read scene ===============  " << std::endl;
        return;
    }

    sofa::simulation::getSimulation()->init(groot.get());

    sofa::MyCfExportVisitor visitor( sofa::core::ExecParams::defaultInstance() );
    groot->executeVisitor( &visitor );


    return;
}


int main(int argc, char** argv)
{
    sofa::simulation::graph::init();
    sofa::component::initComponentBase();
    sofa::component::initComponentCommon();
    sofa::component::initComponentGeneral();
    sofa::component::initComponentAdvanced();
    sofa::component::initComponentMisc();

    // --- Parameter initialisation ---
    std::vector<std::string> files; // filename
    std::string fileName ;
    std::vector<std::string> plugins;

    sofa::helper::parse(&files, "\nThis is a SOFA batch that permits to run and to save simulation states without GUI.\nGive a name file containing actions == list of (input .scn, #simulated time steps, output .simu). See file tasks for an example.\n\nHere are the command line arguments")
    .option(&plugins,'l',"load","load given plugins")
    (argc,argv);


    // --- check input file
    if (!files.empty())
        fileName = files[0];
    else
    {
        fileName = "scenes"; // defaut file for storing the list of scenes
    }

    fileName = std::string(MyCfExport_DIR) + "/" + fileName;  // MyCfExport_DIR defined in CMakeLists.txt
    cout << "Reading scene list in " << fileName << endl;


    // --- Init components ---
    sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());

    // --- plugins ---
    for (unsigned int i=0; i<plugins.size(); i++)
        sofa::helper::system::PluginManager::getInstance().loadPlugin(plugins[i]);
    sofa::helper::system::PluginManager::getInstance().init();


    // --- Perform task list ---
    std::ifstream fileList(fileName.c_str());
    std::string sceneFile;

    while( fileList >> sceneFile   )
    {
        sofa::helper::system::DataRepository.findFile(sceneFile);
        apply(sceneFile);
    }
    fileList.close();

    sofa::simulation::graph::cleanup();
    return 0;
}
