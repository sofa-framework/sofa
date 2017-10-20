/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#define SOFA_COMPONENT_ENGINE_GROUPFILTERYOUNGMODULUS_CPP
#include "GroupFilterYoungModulus.h"
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
GroupFilterYoungModulus<DataTypes>::GroupFilterYoungModulus()
    : f_groups( initData (&f_groups, "groups", "Groups") )
    , f_primitives( initData (&f_primitives, "primitives", "Vector of primitives (indices)") )
    , f_elementsGroup( initData (&f_elementsGroup, "elementsGroup", "Vector of groups (each element gives its group") )
    , f_youngModulus( initData (&f_youngModulus, "youngModulus", "Vector of young modulus for each primitive") )
    , p_mapGroupModulus( initData (&p_mapGroupModulus, "mapGroupModulus", "Mapping between groups and modulus") )
    , p_defaultModulus( initData (&p_defaultModulus, (Real) 10000.0, "defaultYoungModulus", "Default value if the primitive is not in a group") )
    , p_groupMod( initData (&p_groupMod, "groupModulus", "list of young modulus for each group") )
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
    //Input
    const std::string& strMap = p_mapGroupModulus.getValue();
    const helper::vector<unsigned int>& primitives = f_primitives.getValue();
    const helper::vector<sofa::core::loader::PrimitiveGroup >& groups = f_groups.getValue();
    const helper::vector<int >& elementsGroup = f_elementsGroup.getValue();

    const Real& defaultModulus =  p_defaultModulus.getValue();

    cleanDirty();

    //Output
    helper::vector<Real>& youngModulusVector = *f_youngModulus.beginWriteOnly();

    //1st method : with Group definition
    if (!groups.empty())
    {
        if (p_groupMod.getValue().empty()) {
            size_t begin = 0, end = strMap.find(";");
            std::string groupName;
            Real youngModulus;

            //does not matter if primitives is empty
            int maxSize = primitives.size();
            std::map<sofa::core::loader::PrimitiveGroup, Real> mapMG;

            //read string and tokenize
            while(end != std::string::npos )
            {
                std::string tempStr = strMap.substr(begin, end);
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

            typename std::map<sofa::core::loader::PrimitiveGroup, Real>::const_iterator itMapMG;
            for (itMapMG = mapMG.begin() ; itMapMG != mapMG.end() ; itMapMG++)
            {
                sofa::core::loader::PrimitiveGroup pg = (*itMapMG).first;
                Real ym = (*itMapMG).second;

                for (int i=pg.p0 ; i<pg.p0+pg.nbp ; i++)
                    youngModulusVector[i] = ym;
            }
        } else {
            youngModulusVector.clear();
            unsigned sz = p_groupMod.getValue().size();
            for (unsigned i=0;i<groups.size();i++) {
                Real ym;
                if (i<sz) ym = p_groupMod.getValue()[i];
                else ym = defaultModulus;

                for (int j=0;j<groups[i].nbp;j++) youngModulusVector.push_back(ym);
            }
        }
    }
    ////2n method : each element gives its "group"
    else if (!elementsGroup.empty())
    {
        size_t begin = 0, end = strMap.find(";");
        int gid;
        Real youngModulus;

        helper::vector<Real> mapMG;
        youngModulusVector.resize(1);

        //read string and tokenize
        while(end != std::string::npos )
        {
            std::string tempStr = strMap.substr(begin, end);

            std::istringstream iss(tempStr);
            iss >> gid >> youngModulus ;
            begin = end+1;
            end = strMap.find(";", begin);

            if ((int) mapMG.size() <= gid)
                mapMG.resize(gid+1);
            mapMG[gid] = youngModulus;
        }

        youngModulusVector.resize(elementsGroup.size());
        for (unsigned int i=0 ; i<elementsGroup.size() ; i++)
        {

            int elem = elementsGroup[i];
            youngModulusVector[i] = mapMG[elem];
        }
    }
    f_youngModulus.endEdit();
}

} // namespace engine

} // namespace component

} // namespace sofa

