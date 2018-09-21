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
#ifndef SOFAPYTHON3_SCENELOADERPY3_H
#define SOFAPYTHON3_SCENELOADERPY3_H

#include <string>
#include <map>

#include <sofa/simulation/SceneLoaderFactory.h>
#include <sofa/simulation/Visitor.h>
#include <sofa/simulation/Node.h>

#include "config.h"

namespace sofapython3
{
using sofa::simulation::Node;
using sofa::simulation::SceneLoader;

/// The scene loader for python3 scene files
class SOFAPYTHON3_API SceneLoaderPY3 : public SceneLoader
{
public:
    /// Pre-loading check
    virtual bool canLoadFileExtension(const char *extension) override;

    /// Pre-saving check
    virtual bool canWriteFileExtension(const char *extension) override;

    /// load the file
    virtual Node::SPtr load(const char *filename) override;

    void loadSceneWithArguments(const char *filename,
                                const std::vector<std::string>& arguments=std::vector<std::string>(0),
                                Node::SPtr* root_out = 0);

    /// get the file type description
    virtual std::string getFileTypeDesc() override;

    /// get the list of file extensions
    virtual void getExtensionList(ExtensionList* list) override;
};

} // namespace sofapython3

#endif // SOFAPYTHON3_SCENELOADERPY3_H
