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
#ifndef LMLFORCE_INL
#define LMLFORCE_INL

#include "LMLForce.h"
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/helper/gl/template.h>


namespace sofa
{

namespace filemanager
{

namespace pml
{

using namespace sofa::helper::gl;

template<class DataTypes>
LMLForce<DataTypes>::LMLForce(Loads* loadsList, const map<unsigned int, unsigned int> &atomIndexToDOFIndex, MechanicalState<DataTypes> *mm)
    : ForceField<DataTypes>(mm), atomToDOFIndexes(atomIndexToDOFIndex)
{
    mmodel = mm;
    loads = new Loads();
    Load * load;
    SReal dirX, dirY, dirZ;
    Deriv ve;
    this->setName("loads");

    //for each load, we search which ones are forces applied on the body nodes
    for (unsigned int i=0 ; i<loadsList->numberOfLoads() ; i++)
    {
        load = loadsList->getLoad(i);
        if (load->getType() == "Force")
        {
            //set the direction
            if (load->getDirection().isToward())
            {
                std::map<unsigned int, unsigned int>::const_iterator titi = atomIndexToDOFIndex.find(load->getDirection().getToward());
                if (titi != atomIndexToDOFIndex.end())
                {
                    unsigned int dofInd = titi->second;
                    dirX = mm->read(core::ConstVecCoordId::position())->getValue()[dofInd].x();
                    dirY = mm->read(core::ConstVecCoordId::position())->getValue()[dofInd].y();
                    dirZ = mm->read(core::ConstVecCoordId::position())->getValue()[dofInd].z();
                }
            }
            else
                load->getDirection(dirX, dirY, dirZ);

            //set the target
            unsigned int cpt=0;
            for (unsigned int j=0 ; j<load->numberOfTargets(); j++)
            {
                std::map<unsigned int, unsigned int>::const_iterator result = atomIndexToDOFIndex.find(load->getTarget(j));
                if (result != atomIndexToDOFIndex.end())
                {
                    cpt++;
                    if (load->getDirection().isToward())
                        ve = Deriv(dirX - mm->read(core::ConstVecCoordId::position())->getValue()[result->second].x(),
                                   dirY - mm->read(core::ConstVecCoordId::position())->getValue()[result->second].y(),
                                   dirZ - mm->read(core::ConstVecCoordId::position())->getValue()[result->second].z());
                    else
                        ve = Deriv(dirX, dirY, dirZ);
                    ve.normalize();
                    forces.push_back(ve);
                    targets.push_back(result->second);
                }
            }

            if (cpt > 0)
                loads->addLoad(load);
        }
    }
}

template<class DataTypes>
void LMLForce<DataTypes>::addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& /*v*/)
{
    //for each points, update the force vector f
    SReal time = this->getContext()->getTime();
    f.resize(x.size());
    Load * load;
    std::vector<unsigned int>::iterator it1 = targets.begin();
    VecDerivIterator it2 = forces.begin();
    SReal dirX, dirY, dirZ;

    for (unsigned int i=0 ; i<loads->numberOfLoads() ; i++)
    {
        load = loads->getLoad(i);
        SReal val = load->getValue(time);
        for (unsigned int j=0 ; j<load->numberOfTargets() ; j++)
        {
            if ( atomToDOFIndexes.find(load->getTarget(j)) != atomToDOFIndexes.end() )
            {
                // if force value == 0, then there's no force !
                if (val == 0)
                    (*it2) = Deriv(0,0,0);
                else
                {
                    //if force direction is defined by a toward index, then it must be updated each step time
                    if (load->getDirection().isToward())
                    {
                        std::map<unsigned int, unsigned int>::const_iterator titi = atomToDOFIndexes.find(load->getDirection().getToward());
                        if (titi != atomToDOFIndexes.end())
                        {
                            unsigned int dofInd = titi->second;
                            (*it2) = mmodel->read(core::ConstVecCoordId::position())->getValue()[dofInd] - mmodel->read(core::ConstVecCoordId::position())->getValue()[*it1];
                            (*it2).normalize();
                        }
                    }
                    else
                    {
                        // if there is new force, then update the direction
                        if( (*it2)[0]==0 && (*it2)[1]==0 && (*it2)[2]==0 )
                        {
                            load->getDirection(dirX, dirY, dirZ);
                            (*it2) = Deriv(dirX, dirY, dirZ);
                            (*it2).normalize();
                        }
                    }

                    //put the force in f
                    f[*it1] += (*it2) * val;
                }
                it1++;
                it2++;
            }
        }
    }
}


// -- VisualModel interface
template<class DataTypes>
void LMLForce<DataTypes>::draw()
{
    //display a little green segment with force direction
    // if (!vparams->displayFlags().getShowForceFields()) return;
    const VecCoord& x = mmodel->read(core::ConstVecCoordId::position())->getValue();
    glDisable (GL_LIGHTING);
    glPointSize(10);
    glColor4f (0.5,1,0.5,1);
    glBegin (GL_LINES);
    VecDerivIterator it2 = forces.begin();
    for (std::vector<unsigned int>::const_iterator it = this->targets.begin(); it != this->targets.end(); ++it)
    {
        glVertexT(x[*it]);
        glVertexT(x[*it]+(*it2));
        it2++;
    }
    glEnd();
}

}
}
}

#endif
