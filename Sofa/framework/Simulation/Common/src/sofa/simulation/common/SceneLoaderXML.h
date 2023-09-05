/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/simulation/common/config.h>
#include <sofa/simulation/common/xml/BaseElement.h>
#include <sofa/simulation/SceneLoaderFactory.h>
#include <sofa/simulation/fwd.h>

namespace sofa::simulation
{

class SOFA_SIMULATION_COMMON_API SceneLoaderXML : public SceneLoader
{
public:
    /// Pre-loading check
    bool canLoadFileExtension(const char *extension) override;

    /// Pre-saving check
    bool canWriteFileExtension(const char *extension) override;

    /// load the file
    virtual sofa::simulation::NodeSPtr doLoad(const std::string& filename, const std::vector<std::string>& sceneArgs) override;

    /// write the file
    void write(sofa::simulation::Node* node, const char *filename) override;

    /// generic function to process xml tree (after loading the xml structure)
    static NodeSPtr processXML(xml::BaseElement* xml, const char *filename);

    /// load a scene from memory (typically : an xml into a string)
    NodeSPtr doLoadFromMemory(const char* filename, const char* data);

    /// load a scene from memory (typically : an xml into a string)
    static NodeSPtr loadFromMemory(const char* filename, const char* data);

    SOFA_ATTRIBUTE_DISABLED("v22.12 (PR#)", "v23.06", "loadFromMemory with 3 arguments specifying the size has been deprecated. Use loadFromMemory(const char* filename, const char* data).")
    static NodeSPtr loadFromMemory( const char *filename, const char *data, unsigned int size ) = delete;

    /// get the file type description
    virtual std::string getFileTypeDesc() override;

    /// get the list of file extensions
    void getExtensionList(ExtensionList* list) override;

    bool syntaxForAddingRequiredPlugin(const std::string& pluginName,
                                       const std::vector<std::string>& listComponents, std::ostream& ss, sofa::simulation::Node* nodeWhereAdded) override;

    // Test if load succeed
    static bool loadSucceed;
};

} // namespace sofa::simulation
