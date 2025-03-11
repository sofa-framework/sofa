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
#include <sofa/component/odesolver/backward/NewtonStatus.h>

namespace sofa::component::odesolver::backward
{

struct NewtonRaphsonConvergenceMeasure
{
    virtual ~NewtonRaphsonConvergenceMeasure() = default;
    virtual bool isMeasured() const = 0;
    virtual bool hasConverged() const = 0;
    virtual NewtonStatus status() const = 0;
    virtual std::string writeWhenConverged() const = 0;
    virtual std::string writeWhenNotConverged() const { return {}; };
    virtual std::string_view measureName() const = 0;

    unsigned int newtonIterationCount = 0;
};

struct NewtonRaphsonConvergenceMeasureWithSquaredParameter : NewtonRaphsonConvergenceMeasure
{
    SReal param;
    SReal squaredParam;

    explicit NewtonRaphsonConvergenceMeasureWithSquaredParameter(SReal p)
    {
        setParam(p);
    }

    void setParam(const SReal p)
    {
        param = p;
        squaredParam = param * param;
    }

    [[nodiscard]] bool isMeasured() const override
    {
        return param > 0;
    }
};

}
