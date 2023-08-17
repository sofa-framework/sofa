/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <runSofaValidation.h>
#include <sofa/helper/system/FileSystem.h>

#include <sofa/helper/system/SetDirectory.h>
using sofa::helper::system::SetDirectory;
using sofa::core::execparams::defaultInstance; 

#include <sofa/component/playback/CompareState.h>
using sofa::component::playback::CompareStateCreator;
#include <sofa/component/playback/ReadState.h>
using sofa::component::playback::ReadStateActivator;

using sofa::core::ExecParams;

namespace runSofa
{

void Validation::execute(const std::string& directory, const std::string& filename, sofa::simulation::Node* node)
{
    msg_info("runSofa::Validation") << "load verification data from " << directory << " and file " << filename;

    std::string refFile = sofa::helper::system::FileSystem::append(directory, SetDirectory::GetFileName(filename.c_str()));

    msg_info("runSofa::Validation") << "reference file: " << refFile;

    CompareStateCreator compareVisitor(sofa::core::execparams::defaultInstance());
    compareVisitor.setCreateInMapping(true);
    compareVisitor.setSceneName(refFile);
    compareVisitor.execute(node);

    ReadStateActivator v_read(sofa::core::execparams::defaultInstance(), true);
    v_read.execute(node);
}

} // namespace runSofa
