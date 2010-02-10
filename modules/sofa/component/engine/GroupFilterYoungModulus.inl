/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#define SOFA_COMPONENT_ENGINE_GROUPFILTERYOUNGMODULUS_CPP
#include <sofa/component/engine/GroupFilterYoungModulus.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace sofa::core::componentmodel::loader;

template <class DataTypes>
GroupFilterYoungModulus<DataTypes>::GroupFilterYoungModulus()
    : f_groups( initData (&f_groups, "groups", "Groups") )
    , f_primitives( initData (&f_primitives, "primitives", "Vector of primitives (indices)") )
    , f_youngModulus( initData (&f_youngModulus, "youngModulus", "Vector of young modulus for each primitive") )
    , p_mapGroupModulus( initData (&p_mapGroupModulus, "mapGroupModulus", "Mapping between groups and modulus") )
    , p_defaultModulus( initData (&p_defaultModulus, (Real) 10000.0, "defaultYoungModulus", "Default value if the primitive is not in a group") )
{
}

template <class DataTypes>
void GroupFilterYoungModulus<DataTypes>::init()
{
    addInput(&f_groups);
    addInput(&f_primitives);
    addOutput(&f_youngModulus);
    setDirtyValue();
}

template <class DataTypes>
void GroupFilterYoungModulus<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void GroupFilterYoungModulus<DataTypes>::update()
{
    cleanDirty();

    //Input
    const std::string& strMap = p_mapGroupModulus.getValue();
    const helper::vector<unsigned int>& primitives = f_primitives.getValue();
    const helper::vector<PrimitiveGroup >& groups = f_groups.getValue();
    const Real& defaultModulus =  p_defaultModulus.getValue();
    //Output
    helper::vector<Real>& youngModulusVector = *f_youngModulus.beginEdit();

    std::map<PrimitiveGroup, Real> mapMG;

    //does not matter if primitives is empty
    int maxSize = primitives.size();

    size_t begin = 0, end = strMap.find(";");
    std::string groupName;
    Real youngModulus;

    //read string and tokenize
    while(end != std::string::npos )
    {
        std::string tempStr = strMap.substr(begin, end-1);
        std::istringstream iss(tempStr);
        iss >> groupName >> youngModulus ;
        begin = end+1;
        end = strMap.find(";", begin);

        if (!groupName.empty() && youngModulus > 0)
        {
            //find group according to name
            bool found = false;
            unsigned int gid = 0;
            for (unsigned int i=0 ; i<groups.size() && !found; i++)
            {
                if (groups[i].groupName.compare(groupName) == 0)
                {
                    found = true;
                    gid = i;
                }
            }

            if (!found)
                serr << "Group " << groupName << " not found" << sendl;
            else
            {
                mapMG[groups[gid]] = youngModulus;

                if (maxSize < groups[gid].p0 + groups[gid].nbp)
                    maxSize = groups[gid].p0+ groups[gid].nbp;
            }
        }
        else serr << "Error while parsing mapping" << sendl;
    }

    //build YM vector
    youngModulusVector.clear();
    youngModulusVector.resize(maxSize);
    std::fill(youngModulusVector.begin(), youngModulusVector.end(), defaultModulus);

    typename std::map<PrimitiveGroup, Real>::const_iterator itMapMG;
    for (itMapMG = mapMG.begin() ; itMapMG != mapMG.end() ; itMapMG++)
    {
        PrimitiveGroup pg = (*itMapMG).first;
        Real ym = (*itMapMG).second;

        for (int i=pg.p0 ; i<pg.p0+pg.nbp ; i++)
            youngModulusVector[i] = ym;
    }

    //std::cout << youngModulusVector.size() << std::endl;
    //std::cout << youngModulusVector << std::endl;

    f_youngModulus.endEdit();
}

} // namespace engine

} // namespace component

} // namespace sofa

