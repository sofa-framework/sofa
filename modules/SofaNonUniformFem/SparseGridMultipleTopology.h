/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_SPARSEGRIDMULTIPLETOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_SPARSEGRIDMULTIPLETOPOLOGY_H
#include "config.h"

#include <string>


#include <SofaNonUniformFem/SparseGridRamificationTopology.h>


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
class SOFA_NON_UNIFORM_FEM_API SparseGridMultipleTopology : public SparseGridRamificationTopology
{
public :
    SOFA_CLASS(SparseGridMultipleTopology,SparseGridRamificationTopology);
protected:
    SparseGridMultipleTopology( bool _isVirtual=false );
public:
    virtual void init()
    {
        if(_computeRamifications.getValue())
            SparseGridRamificationTopology::init(  );
        else
            SparseGridTopology::init(  );
    }

    virtual void buildAsFinest();
    virtual void buildFromFiner()
    {
        if(_computeRamifications.getValue())
            SparseGridRamificationTopology::buildFromFiner(  );
        else
            SparseGridTopology::buildFromFiner(  );
    }
    virtual void buildVirtualFinerLevels();


    virtual int findCube(const Vector3 &pos, SReal &fx, SReal &fy, SReal &fz)
    {
        if(_computeRamifications.getValue())
            return SparseGridRamificationTopology::findCube( pos,fx,fy,fz  );
        else
            return SparseGridTopology::findCube( pos,fx,fy,fz );
    }

    virtual int findNearestCube(const Vector3& pos, SReal& fx, SReal &fy, SReal &fz)
    {
        if(_computeRamifications.getValue())
            return SparseGridRamificationTopology::findNearestCube( pos,fx,fy,fz );
        else
            return SparseGridTopology::findNearestCube( pos,fx,fy,fz );
    }



protected :


    Data< helper::vector< std::string > > _fileTopologies;
    Data< helper::vector< float > > _dataStiffnessCoefs;
    Data< helper::vector< float > > _dataMassCoefs;
    Data<bool> _computeRamifications;
    Data<bool> _erasePreviousCoef;




    void buildFromTriangleMesh(helper::io::Mesh*, unsigned fileIdx);
    helper::vector< RegularGridTopology::SPtr > _regularGrids;
    helper::vector< helper::vector<Type> > _regularGridTypes;
    void assembleRegularGrids(helper::vector<Type>& regularGridTypes,helper::vector< float >& regularStiffnessCoefs,helper::vector< float >& regularMassCoefs);
};



}
}
}

#endif

