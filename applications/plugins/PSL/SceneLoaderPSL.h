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
#ifndef SCENELOADERPYSON_H
#define SCENELOADERPYSON_H

#include <PSL/config.h>
#include <sofa/simulation/SceneLoaderFactory.h>


#include <sofa/simulation/Visitor.h>
#include <string>
#include <map>

extern "C" {
    struct PyMethodDef;
}

namespace sofa
{

namespace simulation
{

namespace _sceneloaderpsl_
{

/// The scene loader/exporter for python scene files
class SOFA_PSL_API SceneLoaderPSL : public SceneLoader
{
public:
    /// Pre-loading check
    virtual bool canLoadFileExtension(const char *extension) override ;

    /// Pre-saving check
    virtual bool canWriteFileExtension(const char *extension) override ;

    /// load the file
    virtual Node::SPtr load(const char *filename) override ;

    /// write the file
    virtual void write(Node* node, const char *filename) override ;

    /// get the file type description
    virtual std::string getFileTypeDesc() override ;

    /// get the list of file extensions
    virtual void getExtensionList(ExtensionList* list) override;
};



} // namespace _sceneloaderpyson_

using _sceneloaderpsl_::SceneLoaderPSL ;

} // namespace simulation

} // namespace sofa



#endif // SCENELOADERPY_H
