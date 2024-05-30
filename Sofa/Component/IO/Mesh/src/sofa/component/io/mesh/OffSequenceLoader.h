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
#include <sofa/component/io/mesh/config.h>

#include <sofa/component/io/mesh/MeshOffLoader.h>

namespace sofa::component::io::mesh
{

/** This class load a sequence of .off mesh files, ordered by index in their name
*/
class SOFA_COMPONENT_IO_MESH_API OffSequenceLoader : public MeshOffLoader
{
public:
    SOFA_CLASS(OffSequenceLoader, MeshOffLoader);
protected:
    OffSequenceLoader();
public:
    void init() override;

    void reset() override;

    void handleEvent(sofa::core::objectmodel::Event* event) override;

    using MeshOffLoader::load;
    virtual bool load(const char * filename);

    void clear();

private:
    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_IO_MESH()
    Data<int> nbFiles;
    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_IO_MESH()
    Data<double> stepDuration;


    /// the number of files in the sequences
    Data<int> d_nbFiles;
    /// duration each file must be loaded
    Data<double> d_stepDuration;

    /// index of the first file
    int firstIndex;
    /// index of the current read file
    int currentIndex;

    ///parsed file name
    std::string m_filenameAndNb;

};

} // namespace sofa::component::io::mesh
