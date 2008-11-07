/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_SPARSEGRIDMULTIPLETOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_SPARSEGRIDMULTIPLETOPOLOGY_H

#include <string>


#include <sofa/component/topology/SparseGridTopology.h>


namespace sofa
{

namespace component
{

namespace topology
{

/**
Build a SparseGridTopology for several given Triangular meshes.
A stiffness coefficient has to be assigned for each mesh. The last found stiffness coefficient is used for an element shared by several meshes => The mesh ordering is important, and so, more specific stiffness informations must appear in last.
*/
class SparseGridMultipleTopology : public SparseGridTopology
{
public :

    SparseGridMultipleTopology( bool _isVirtual=false ) : SparseGridTopology(_isVirtual),
        _fileTopologies(initData(&_fileTopologies, helper::vector< std::string >() , "fileTopologies", "All topology filenames")),
        _dataStiffnessCoefs(initData(&_dataStiffnessCoefs, helper::vector< float >() , "stiffnessCoefs", "A stiffness coefficient for each topology filename"))
    {
    }

    virtual void buildAsFinest();
    virtual void buildVirtualFinerLevels();
    virtual void buildFromFiner();

    float getStiffnessCoef(int elementIdx);

protected :


    Data< helper::vector< std::string > > _fileTopologies;
    Data< helper::vector< float > > _dataStiffnessCoefs;


    helper::vector< float > _stiffnessCoefs;

    void buildFromTriangleMesh(helper::io::Mesh*, unsigned fileIdx);
    helper::vector< RegularGridTopology > _regularGrids;
    helper::vector< helper::vector<Type> > _regularGridTypes;
// 		helper::vector< float > _regularStiffnessCoefs;
    void assembleRegularGrids(helper::vector<Type>& regularGridTypes,helper::vector< float >& regularStiffnessCoefs);
};



}
}
}

#endif

