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
#include <SofaValidation/config.h>

#include <sofa/type/vector.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/core/DevBaseMonitor.h>

namespace sofa::component::misc
{

template <class TDataTypes>
class SOFA_SOFAVALIDATION_API DevMonitor: public virtual core::DevBaseMonitor
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(DevMonitor,TDataTypes), core::DevBaseMonitor);

    typedef TDataTypes DataTypes;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;
    typedef typename std::pair< Coord,Real > TData;

    Data < double > f_period; ///< period between outputs
    Data< sofa::type::vector< unsigned int > > f_indices; ///< Indices of the points which will be monitored

    DevMonitor():
        f_period( initData(&f_period, 1.0, "period", "period between outputs"))
        , f_indices( initData(&f_indices,"indices","Indices of the points which will be monitored") )
        , lastTime(0)
    {
    }

    sofa::type::vector<TData> getData()
    {
        sofa::type::vector<TData> copy;
        copy = data;
        data.clear();
        return copy;
    }

    void handleEvent(sofa::core::objectmodel::Event* event) override
    {
        if (simulation::AnimateEndEvent::checkEventType(event))
        {
            timestamp = getContext()->getTime();
            // write the state using a period
            if (timestamp+getContext()->getDt()/2 >= (lastTime + f_period.getValue()))
            {
                eval();
                lastTime += f_period.getValue();
            }
        }
    }

protected:
    double lastTime;
    double timestamp;
    sofa::type::vector<TData> data;
};

} // namespace sofa::component::misc

