/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_SIMULATION_SCENELOADERFACTORY_H
#define SOFA_SIMULATION_SCENELOADERFACTORY_H

#include <sofa/simulation/simulationcore.h>
#include <sofa/simulation/Node.h>
#include <sofa/helper/system/SetDirectory.h>


namespace sofa
{

namespace simulation
{

/**
 *  \brief Main class used to register scene file loaders
 *
 *  It uses the Factory design pattern, where each class is registered in a map,
 *  and dynamically retrieved given the type name.
 *
 */

/// Abstract interface of a scene loader
class SOFA_SIMULATION_CORE_API SceneLoader
{
public:
    typedef std::vector<std::string> ExtensionList;

    /// Pre-loading check
    virtual bool canLoadFileName(const char *filename)
    {
        std::string ext = sofa::helper::system::SetDirectory::GetExtension(filename);
        return canLoadFileExtension(ext.c_str());
    }

    /// Pre-saving check
    virtual bool canWriteFileName(const char *filename)
    {
        std::string ext = sofa::helper::system::SetDirectory::GetExtension(filename);
        return canWriteFileExtension(ext.c_str());
    }

    virtual bool canLoadFileExtension(const char *extension) = 0;

    virtual bool canWriteFileExtension(const char * /*extension*/) { return false; }

    /// load the file
    /// @warning do not forgot to call notifyLoadingScene()
    virtual sofa::simulation::Node::SPtr load(const char *filename) = 0;

    /// write scene graph in the file
    virtual void write(sofa::simulation::Node* /*node*/, const char * /*filename*/) {}

    /// get the file type description
    virtual std::string getFileTypeDesc() = 0;

    /// get the list of file extensions
    virtual void getExtensionList(ExtensionList* list) = 0;



    /// to be able to inform when a scene is loaded
    struct Listener
    {
        virtual void rightBeforeLoadingScene() {} ///< callback called just before loading the scene file
    };

    /// adding a listener
    static void addListener( Listener* l ) { s_listerners.insert(l); }

    /// removing a listener
    static void removeListener( Listener* l ) { s_listerners.erase(l); }

protected:

    /// the list of listerners
    typedef std::set<Listener*> Listeners;
    static Listeners s_listerners;
    static void notifyLoadingScene() { for( auto* l : s_listerners ) l->rightBeforeLoadingScene(); }

};


class SOFA_SIMULATION_CORE_API SceneLoaderFactory
{

public:
    typedef std::vector<SceneLoader*> SceneLoaderList;

    /// Get the ObjectFactory singleton instance
    static SceneLoaderFactory* getInstance();

protected:

    /// Main class registry
    SceneLoaderList registry;

public:
    /// Get an entry given a file extension
    SceneLoader* getEntryFileExtension(std::string extension);

    /// Get an entry given a file name
    SceneLoader* getEntryFileName(std::string filename);

    /// Get an exporter entry given a file extension
    SceneLoader* getExporterEntryFileExtension(std::string extension);

    /// Get an exporter entry given a file name
    SceneLoader* getExporterEntryFileName(std::string filename);

    /// Add a scene loader
    SceneLoader* addEntry(SceneLoader *loader);

    /// Get the list of loaders
    SceneLoaderList* getEntries() {return &registry;}

    /// Get the list of supported extension. It returns a vector of string
    /// but it is RVO optimized so the overhead is reduced.
    std::vector<std::string> extensions() ;

};

} // namespace simulation

} // namespace sofa


#endif // SOFA_SIMULATION_SCENELOADERFACTORY_H

