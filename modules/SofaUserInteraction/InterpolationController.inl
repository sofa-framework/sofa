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
#ifndef SOFA_COMPONENT_FORCEFIELD_INTERPOLATIONCONTROLLER_INL
#define SOFA_COMPONENT_FORCEFIELD_INTERPOLATIONCONTROLLER_INL

#include "InterpolationController.h"
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/Simulation.h>

#include <sofa/core/visual/VisualParams.h>

namespace sofa
{

namespace component
{

namespace controller
{
  
using core::behavior::MechanicalState;
using sofa::defaulttype::Vector3;

template<class DataTypes>
InterpolationController<DataTypes>::InterpolationController()
  : f_evolution( initData(&f_evolution, (int)STABLE , "evolution", "O for fixity, 1 for inflation, 2 for deflation"))
  , f_period( initData(&f_period, double(1.0), "period", "time to cover all the interpolation positions between original mesh and alpha*(objective mesh), in seconds "))
  , f_alphaMax( initData(&f_alphaMax, float(1.0), "alphaMax", "bound defining the max interpolation between the origina (alpha=0) and the objectiv (alpha=1) meshes"))
  , f_alpha0( initData(&f_alpha0, float(0.0), "alpha0", "alpha value at t=0. (0 < alpha0 < 1)"))
  , f_interpValues(initData(&f_interpValues, "interpValues", "values or the interpolation"))
  , fromModel(initLink("original", "Original mesh"))
  , toModel(initLink("objective", "Objective mesh"))
{
}

template<class DataTypes>
void InterpolationController<DataTypes>::bwdInit() {
    if (!fromModel || !toModel ) //|| !interpModel)
    {
        serr << "One or more MechanicalStates are missing";
        return;
    }

    fromXs = &fromModel->read(core::ConstVecCoordId::position())->getValue();
    toXs = &toModel->read(core::ConstVecCoordId::position())->getValue();

    if (fromXs->size() != toXs->size())
    {
       serr << "<InterpolationController> Different number of nodes between the two meshes (original and objective)";
    }

    if (f_alpha0.getValue()>=0.0 && f_alpha0.getValue()<=f_alphaMax.getValue())
    {
        alpha = f_alpha0.getValue();
    }
    else
    {
        serr << "<InterpolationController> Wrong value for alpha0";
        alpha=0;
    }

    sofa::helper::WriteAccessor< DataVecCoord > interpValues = f_interpValues;
    interpValues.resize(fromXs[0].size());
    interpolation(); //interpXs);



}

template<class DataTypes>
void InterpolationController<DataTypes>::interpolation() { //VecCoord &interpXs) {
    sofa::helper::WriteAccessor< DataVecCoord > interpValues = f_interpValues;
//    interpValues.resize(fromXs[0].size());

    for (size_t ptIter=0; ptIter < fromXs[0].size(); ptIter++)
    {
        for (size_t i=0; i< interpValues[0].size(); i++) //interpXs[0].size(); i++)
        {
            //interpXs[ptIter][i] = (fromXs[0][ptIter][i] + alpha*(toXs[0][ptIter][i] - fromXs[0][ptIter][i]) );
            interpValues[ptIter][i] = (fromXs[0][ptIter][i] + alpha*(toXs[0][ptIter][i] - fromXs[0][ptIter][i]) );
        }
    }
}

template<class DataTypes>
void InterpolationController<DataTypes>::handleEvent(core::objectmodel::Event *event) {
    if (sofa::simulation::AnimateBeginEvent::checkEventType(event))
    {
        if (f_evolution.getValue() != STABLE)
        {
            //helper::WriteAccessor<Data<VecCoord> > interpXData = *interpModel->write(sofa::core::VecCoordId::position());
            //VecCoord& interpXs = interpXData.wref();

            //dAlpha computation(period,dt)
            dAlpha = (float)(1.0 / (f_period.getValue() / this->getContext()->getDt()));

            //alpha computation(evolution,alpha,alphaMax,dAlpha)
            switch (static_cast<Evolution_Type>(f_evolution.getValue()))
            {
            case INFLATING:
                alpha = (alpha <= f_alphaMax.getValue()-dAlpha)? alpha+dAlpha : alpha;
                break;

            case DEFLATING:
                alpha = (alpha >= dAlpha)? alpha-dAlpha : alpha;
                break;

            default:
                break;
            }

            //interpolation computation(alpha)
            interpolation();//interpXs);

            if(alpha<dAlpha || alpha>(f_alphaMax.getValue()-dAlpha))
            {
                f_evolution.setValue(STABLE);
            }
        }
    }
}

template<class DataTypes>
void InterpolationController<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if ((!vparams->displayFlags().getShowVisualModels()) || f_evolution.getValue()==STABLE) return;
    
    sofa::helper::ReadAccessor< DataVecCoord > interpValues = f_interpValues;
    if(interpValues.size() != this->fromXs[0].size()) return;
    //const VecCoordinterpXs = interpModel->read(core::ConstVecCoordId::position())->getValue();

    defaulttype::Vec<4,float> color;
    switch (static_cast<Evolution_Type>(f_evolution.getValue()))
    {
    case INFLATING:
        color = defaulttype::Vec<4,float>(1.0f,0.4f,0.4f,1.0f);
        break;

    case DEFLATING:
        color = defaulttype::Vec<4,float>(0.4f,0.4f,1.0f,1.0f);
        break;

    default:
        break;
    }

    const Vector3 fromXs_to_toXs(this->toXs[0][0][0] - this->fromXs[0][0][0],
                                 this->toXs[0][0][1] - this->fromXs[0][0][1],
                                 this->toXs[0][0][2] - this->fromXs[0][0][2]);
    const float norm = (float)(fromXs_to_toXs.norm()/10.0);

    for (unsigned ptIter = 0; ptIter < fromXs[0].size(); ptIter += fromXs[0].size()/80)
    {
        Vector3 p1(interpValues[ptIter][0],interpValues[ptIter][1],interpValues[ptIter][2]), p2;
        //Vector3 p1(interpXs[0][ptIter][0],interpXs[0][ptIter][1],interpXs[0][ptIter][2]), p2;

        switch (static_cast<Evolution_Type>(f_evolution.getValue()))
        {
        case INFLATING:
            p2 = Vector3(this->toXs[0][ptIter][0],this->toXs[0][ptIter][1],this->toXs[0][ptIter][2]) ;
            break;

        case DEFLATING:
            p2 = Vector3(this->fromXs[0][ptIter][0],this->fromXs[0][ptIter][1],this->fromXs[0][ptIter][2]);
            break;

        default:
            break;
        }

        vparams->drawTool()->drawArrow(p1,p2, norm, color);
    }
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_INTERPOLATIONCONTROLLER_INL



