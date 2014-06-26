/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SCENELOADERPY_H
#define SCENELOADERPY_H

#include "SofaPython.h"
#include <sofa/simulation/common/SceneLoaderFactory.h>

namespace sofa
{

namespace simulation
{

class SceneLoaderPY : public SceneLoader
{
public:
    /// Pre-loading check
    SOFA_SOFAPYTHON_API virtual bool canLoadFileExtension(const char *extension);

    /// load the file
    SOFA_SOFAPYTHON_API virtual Node::SPtr load(const char *filename);
    SOFA_SOFAPYTHON_API Node::SPtr loadSceneWithArguments(const char *filename, const std::vector<std::string>& arguments=std::vector<std::string>(0));
    SOFA_SOFAPYTHON_API bool loadTestWithArguments(const char *filename, const std::vector<std::string>& arguments=std::vector<std::string>(0));

    /// get the file type description
    SOFA_SOFAPYTHON_API virtual std::string getFileTypeDesc();

    /// get the list of file extensions
    SOFA_SOFAPYTHON_API virtual void getExtensionList(ExtensionList* list);
};

} // namespace simulation

} // namespace sofa



#endif // SCENELOADERPY_H
