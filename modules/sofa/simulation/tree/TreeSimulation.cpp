/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/simulation/tree/TreeSimulation.h>

#include <sofa/simulation/common/FindByTypeVisitor.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/PipeProcess.h>

#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/FileRepository.h>

#include <fstream>
#include <string.h>
#ifndef WIN32
#include <locale.h>
#endif

namespace sofa
{

namespace simulation
{

namespace tree
{

using namespace sofa::defaulttype;


Simulation* getSimulation()
{
    if ( simulation::Simulation::theSimulation==NULL )
        setSimulation( new TreeSimulation );
    return simulation::getSimulation();
}

/// Load a scene from a file
Node* TreeSimulation::processXML(xml::BaseElement* xml, const char *filename)
{
    if ( xml==NULL )
    {
        return NULL;
    }

    // We go the the current file's directory so that all relative path are correct
    helper::system::SetDirectory chdir ( filename );

#ifndef WIN32
    // Reset local settings to make sure that floating-point values are interpreted correctly
    setlocale(LC_ALL,"C");
    setlocale(LC_NUMERIC,"C");
#endif

// 				std::cout << "Initializing objects"<<std::endl;
    if ( !xml->init() )
    {
        std::cerr << "Objects initialization failed."<<std::endl;
    }

    GNode* root = dynamic_cast<GNode*> ( xml->getObject() );
    if ( root == NULL )
    {
        std::cerr << "Objects initialization failed."<<std::endl;
        delete xml;
        return NULL;
    }

// 				std::cout << "Initializing simulation "<<root->getName() <<std::endl;

    // Find the Simulation component in the scene
    FindByTypeVisitor<Simulation> findSimu;
    findSimu.execute(root);
    if( !findSimu.found.empty() )
        setSimulation( findSimu.found[0] );

    // As mappings might be initialized after visual models, it is necessary to update them
    // BUGFIX (Jeremie A.): disabled as initTexture was not called yet, and the GUI might not even be up yet
    //root->execute<VisualUpdateVisitor>();

    return root;
}

/// Load from a string in memory
Node* TreeSimulation::loadFromMemory ( const char *filename, const char *data, unsigned int size )
{
    //::sofa::simulation::init();
// 				std::cerr << "Loading simulation XML file "<<filename<<std::endl;
    xml::BaseElement* xml = xml::loadFromMemory (filename, data, size );

    Node* root = processXML(xml, filename);

// 				std::cout << "load done."<<std::endl;
    delete xml;

    return root;
}


/// Load a scene from a file
Node* TreeSimulation::loadFromFile ( const char *filename )
{
    //::sofa::simulation::init();
// 				std::cerr << "Loading simulation XML file "<<filename<<std::endl;
    xml::BaseElement* xml = xml::loadFromFile ( filename );

    Node* root = processXML(xml, filename);

// 				std::cout << "load done."<<std::endl;
    delete xml;

    return root;
}

/// Load a scene
Node* TreeSimulation::load ( const char *filename )
{
    std::string ext = sofa::helper::system::SetDirectory::GetExtension(filename);
    if (ext == "php" || ext == "pscn")
    {
        std::string out="",error="";
        std::vector<std::string> args;


        //TODO : replace when PipeProcess will get file as stdin
        //at the moment, the filename is given as an argument
        args.push_back(std::string("-f" + std::string(filename)));
        //args.push_back("-w");
        std::string newFilename="";
        //std::string newFilename=filename;

        helper::system::FileRepository fp("PATH", ".");
#ifdef WIN32
        std::string command = "php.exe";
#else
        std::string command = "php";
#endif
        if (!fp.findFile(command,""))
        {
            std::cerr << "TreeSimulation : Error : php not found in your PATH environment" << std::endl;
            return NULL;
        }

        sofa::helper::system::PipeProcess::executeProcess(command.c_str(), args,  newFilename, out, error);

        if(error != "")
        {
            std::cerr << "TreeSimulation : load : "<< error << std::endl;
            if (out == "")
                return NULL;
        }

        return loadFromMemory(filename, out.c_str(), out.size());
    }

    if (ext == "scn" || ext == "xml")
    {
        return loadFromFile(filename);
    }

    std::cerr << "TreeSimulation : Error : extension not handled" << std::endl;
    return NULL;

}

/// Delete a scene from memory. After this call the pointer is invalid
void TreeSimulation::unload ( Node* root )
{
    Simulation::unload(root);
    delete root;
}

/// Create a new node
Node* TreeSimulation::newNode(const std::string& name)
{
    return new GNode(name);
}

SOFA_DECL_CLASS ( TreeSimulation );
// Register in the Factory
int TreeSimulationClass = core::RegisterObject ( "Main simulation algorithm, based on tree graph" )
        .add< TreeSimulation >()
        ;



} // namespace tree

} // namespace simulation

} // namespace sofa

