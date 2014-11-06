/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_LOADER_OFFSEQUENCELOADER_H
#define SOFA_COMPONENT_LOADER_OFFSEQUENCELOADER_H

#include <SofaLoader/MeshOffLoader.h>
#include <sofa/SofaCommon.h>

namespace sofa
{

namespace component
{

namespace loader
{

/** This class load a sequence of .off mesh files, ordered by index in their name
*/
class SOFA_LOADER_API OffSequenceLoader : public sofa::component::loader::MeshOffLoader
{
public:
    SOFA_CLASS(OffSequenceLoader,sofa::component::loader::MeshOffLoader);
protected:
    OffSequenceLoader();
public:
    virtual void init();

    virtual void reset();

    virtual void handleEvent(sofa::core::objectmodel::Event* event);

    virtual bool load(const char * filename);

    void clear();

private:
    /// the number of files in the sequences
    Data<int> nbFiles;
    /// duration each file must be loaded
    Data<double> stepDuration;

    /// index of the first file
    int firstIndex;
    /// index of the current read file
    int currentIndex;

    ///parsed file name
    std::string m_filenameAndNb;

};

}

}

}

#endif
