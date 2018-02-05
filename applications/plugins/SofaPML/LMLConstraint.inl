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
#ifndef LMLCONSTRAINT_INL
#define LMLCONSTRAINT_INL

#include "sofa/core/behavior/Constraint.inl"
#include "LMLConstraint.h"
#include "sofa/helper/gl/template.h"

namespace sofa
{

namespace filemanager
{

namespace pml
{


template<class DataTypes>
LMLConstraint<DataTypes>::LMLConstraint(Loads* loadsList, const std::map<unsigned int, unsigned int> &atomIndexToDOFIndex, sofa::core::behavior::MechanicalState<DataTypes> *mm)
    : sofa::core::behavior::Constraint<DataTypes>(mm), atomToDOFIndexes(atomIndexToDOFIndex)
{
    mmodel = mm;
    loads = new Loads();
    Load * load;
    SReal dirX, dirY, dirZ;
    this->setName("loads");

    //for each load, we search which ones are translations applied on the body nodes
    for (unsigned int i=0 ; i<loadsList->numberOfLoads() ; i++)
    {
        load = loadsList->getLoad(i);
        if (load->getType() == "Translation")
        {
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
            unsigned int cpt=0;
            for (unsigned int j=0 ; j<load->numberOfTargets(); j++)
            {
                std::map<unsigned int, unsigned int>::const_iterator result = atomIndexToDOFIndex.find(load->getTarget(j));
                if (result != atomIndexToDOFIndex.end())
                {
                    cpt++;
                    if (load->getDirection().isToward())
                        addConstraint(result->second,
                                      Deriv(dirX - mm->read(core::ConstVecCoordId::position())->getValue()[result->second].x(),
                                            dirY - mm->read(core::ConstVecCoordId::position())->getValue()[result->second].y(),
                                            dirZ - mm->read(core::ConstVecCoordId::position())->getValue()[result->second].z()));
                    else
                        addConstraint(result->second, Deriv(dirX,dirY,dirZ) );
                    // fix targets on the X axe
                    if (load->getDirection().isXNull() && load->getValue(0) != 0)
                        fixDOF(result->second, 0);
                    if (load->getDirection().isYNull() && load->getValue(0) != 0) // fix targets on the Y axe
                        fixDOF(result->second, 1);
                    if (load->getDirection().isZNull() && load->getValue(0) != 0) // fix targets on the Z axe
                        fixDOF(result->second, 2);
                }
            }
            if (cpt > 0)
                loads->addLoad(load);
        }
    }
}


template<class DataTypes>
LMLConstraint<DataTypes>*  LMLConstraint<DataTypes>::addConstraint(unsigned int index, Deriv trans)
{
    std::cout << "IPMAC Adding constraint " << index << std::endl;
    this->targets.push_back(index);
    trans.normalize();
    this->translations.push_back(trans);
    this->directionsNULLs.push_back(Deriv(1,1,1));
    this->initPos.push_back(Deriv(0,0,0));
    return this;
}

template<class DataTypes>
LMLConstraint<DataTypes>*  LMLConstraint<DataTypes>::removeConstraint(int index)
{
    std::vector<unsigned int>::iterator it1=targets.begin();
    VecDerivIterator it2=translations.begin();
    VecDerivIterator it3=directionsNULLs.begin();
    while(it1 != targets.end() && *it1!=(unsigned)index)
    {
        it1++;
        it2++;
        it3++;
    }

    targets.erase(it1);
    translations.erase(it2);
    directionsNULLs.erase(it3);

    return this;
}


template<class DataTypes>
void LMLConstraint<DataTypes>::fixDOF(int index, int axe)
{
    //set the value to 1 on the corrects vector component
    std::vector<unsigned int>::iterator it1=targets.begin();
    VecDerivIterator it2 = directionsNULLs.begin();
    while(it1 != targets.end() && *it1!=(unsigned)index)
    {
        it1++;
        it2++;
    }

    (*it2)[axe] = 0;
}


template<class DataTypes>
void LMLConstraint<DataTypes>::projectResponse(VecDeriv& dx)
{
    //VecCoord& x = *this->mmodel->getX();
    //dx.resize(x.size());
    std::cout << "VecDeriv before = " << dx << std::endl;
    SReal time = this->getContext()->getTime();
    SReal prevTime = time - this->getContext()->getDt();

    std::vector<unsigned int>::iterator it1=targets.begin();
    VecDerivIterator it2=translations.begin();
    VecDerivIterator it3=directionsNULLs.begin();
    Load * load;
    SReal valTime, prevValTime;

    for (unsigned int i=0 ; i<loads->numberOfLoads() ; i++)
    {
        load = loads->getLoad(i);
        valTime = load->getValue(time);
        prevValTime = load->getValue(prevTime);
        for(unsigned int j=0 ; j<load->numberOfTargets(); j++)
        {
            std::cout << "load = " << j << std::endl;
            if ( atomToDOFIndexes.find(load->getTarget(j)) != atomToDOFIndexes.end() )
            {
                std::cout << "applying" << std::endl;
                //Deriv dirVec(0,0,0);
                if (load->getDirection().isXNull() && valTime != 0)
                    (*it3)[0]=0;	// fix targets on the X axe
                if (load->getDirection().isYNull() && valTime != 0)
                    (*it3)[1]=0;	// fix targets on the Y axe
                if (load->getDirection().isZNull() && valTime != 0)
                    (*it3)[2]=0;	// fix targets on the Z axe

                std::cout << "valTime = " << valTime << std::endl;
                if (valTime == 0)
                    (*it3) = Deriv(1,1,1);
                else
                {
                    std::cout << "ELSE" << std::endl;
                    if (load->getDirection().isToward())
                    {
                        std::map<unsigned int, unsigned int>::const_iterator titi = atomToDOFIndexes.find(load->getDirection().getToward());
                        if (titi != atomToDOFIndexes.end())
                        {
                            (*it2) = mmodel->read(core::ConstVecCoordId::position())->getValue()[titi->second] - mmodel->read(core::ConstVecCoordId::position())->getValue()[*it1];
                            it2->normalize();
                        }
                    }
                    //cancel the dx value on the axes fixed (where directionNULLs value is equal to 1)
                    //Or apply the translation vector (where directionNULLs value is equal to 0)
                    dx[*it1][0] = ((*it2)[0]*valTime)-((*it2)[0]*prevValTime)*(*it3)[0];
                    dx[*it1][1] = ((*it2)[1]*valTime)-((*it2)[1]*prevValTime)*(*it3)[1];
                    dx[*it1][2] = ((*it2)[2]*valTime)-((*it2)[2]*prevValTime)*(*it3)[2];
                }

                it1++;
                it2++;
                it3++;
            }
        }
    }
    std::cout << "VecDeriv after = " << dx << std::endl;
}

template<class DataTypes>
void LMLConstraint<DataTypes>::projectPosition(VecCoord& /*x*/)
{
    /*SReal time = this->getContext()->getTime();

    std::vector<unsigned int>::iterator it1=targets.begin();
    VecDerivIterator it2=translations.begin();
    VecDerivIterator it3=initPos.begin();
    Load * load;

    for (unsigned int i=0 ; i<loads->numberOfLoads() ; i++)
    {
        load = loads->getLoad(i);
        for(unsigned int j=0 ; j<load->numberOfTargets();j++)
        {
            if ( atomToDOFIndexes.find(load->getTarget(j)) != atomToDOFIndexes.end() ) {
                if ( (*it3)[0]==0.0 && (*it3)[1]==0.0 && (*it3)[2]==0.0)
                    *it3 = x[*it1];
                if(  load->getDirection().isXNull() && load->getDirection().isYNull() && load->getDirection().isZNull() )
                    *it3 = x[*it1];

                if (load->getValue(time) != 0.0){
                    if (load->getDirection().isToward()) {
                        std::map<unsigned int, unsigned int>::const_iterator titi = atomToDOFIndexes.find(load->getDirection().getToward());
                        if (titi != atomToDOFIndexes.end()){
                            (*it2) = (*mmodel->getX())[titi->second] - (*mmodel->getX())[*it1];
                            it2->normalize();
                        }
                    }
                    x[*it1] = (*it3) + (*it2)*load->getValue(time);
                }

                it1++;
                it2++;
                it3++;
            }
        }
    }*/
}



// -- VisualModel interface
template<class DataTypes>
void LMLConstraint<DataTypes>::draw()
{

    // if (!vparams->displayFlags().getShowBehaviorModels()) return;

    const VecCoord& x = mmodel->read(core::ConstVecCoordId::position())->getValue();
    glDisable (GL_LIGHTING);
    glColor4f (1,0.5,0.5,1);

    glPointSize(20);

    //for Fixed points, display a big red point
    glBegin (GL_POINTS);
    VecDerivIterator it2 = directionsNULLs.begin();
    for (std::vector<unsigned int>::const_iterator it = this->targets.begin(); it != this->targets.end(); ++it)
    {
        //std::cout << "IP MAC its = " << (*it2)[0] << " " << (*it2)[1] << " " <<(*it2)[2] << " " <<std::endl;
        if ((*it2)[0]==0 && (*it2)[1]==0 && (*it2)[2]==0 )
        {
            //std::cout << "IPMAC draw constraint" << x[*it] << std::endl;
            helper::gl::glVertexT(x[*it]);
        }
        it2++;
    }
    glEnd();

    //for translated points, display a little red segment with translation direction
    glPointSize(10);
    glBegin( GL_POINTS );
    VecDerivIterator it3 = translations.begin();
    it2 = directionsNULLs.begin();
    for (std::vector<unsigned int>::const_iterator it = this->targets.begin(); it != this->targets.end(); ++it)
    {
        if ((*it2)[0]==1 || (*it2)[1]==1 || (*it2)[2]==1 )
        {
            //std::cout << "IPMAC draw constraint " << x[*it] << std::endl;
            helper::gl::glVertexT(x[*it]);
            //helper::gl::glVertexT(x[*it]+*it3);
        }
        it3++;
        it2++;
    }
    glEnd();


}

}
}
}

#endif //LMLCONSTRAINT_INL
