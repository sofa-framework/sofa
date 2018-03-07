/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_SIMULATION_SCENELOADERXML_H
#define SOFA_SIMULATION_SCENELOADERXML_H

#include <sofa/simulation/SceneLoaderFactory.h>
#include <SofaSimulationCommon/xml/BaseElement.h>

namespace sofa
{

namespace simulation
{

class SOFA_SIMULATION_COMMON_API SceneLoaderXML : public SceneLoader
{
public:
    /// Pre-loading check
    virtual bool canLoadFileExtension(const char *extension);

    /// Pre-saving check
    virtual bool canWriteFileExtension(const char *extension);

    /// load the file
    virtual sofa::simulation::Node::SPtr load(const char *filename);

    /// write the file
    virtual void write(sofa::simulation::Node* node, const char *filename);

    /// generic function to process xml tree (after loading the xml structure)
    static Node::SPtr processXML(xml::BaseElement* xml, const char *filename);

    /// load a scene from memory (typically : an xml into a string)
    static Node::SPtr loadFromMemory ( const char *filename, const char *data, unsigned int size );

    /// get the file type description
    virtual std::string getFileTypeDesc();

    /// get the list of file extensions
    virtual void getExtensionList(ExtensionList* list);

    // Test if load succeed
    static bool loadSucceed;
};

} // namespace simulation

} // namespace sofa



#endif // SOFA_SIMULATION_SCENELOADERXML_H
