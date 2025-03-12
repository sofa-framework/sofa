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

#include <sofa/component/odesolver/backward/convergence/NewtonRaphsonConvergenceMeasure.h>

namespace sofa::component::odesolver::backward
{


class RelativeEstimateDifferenceMeasure : public NewtonRaphsonConvergenceMeasureWithSquaredParameter
{
public:
    explicit RelativeEstimateDifferenceMeasure(SReal relativeEstimateDifferenceThreshold)
        : NewtonRaphsonConvergenceMeasureWithSquaredParameter(relativeEstimateDifferenceThreshold)
    {}
    
    bool hasConverged() const override
    {
        return squaredPreviousEvaluation > 0
            && squaredAbsoluteDifference < squaredPreviousEvaluation * squaredParam;
    }
    
    NewtonStatus status() const override
    {
        static constexpr auto convergedRelativeEstimateDifference = NewtonStatus("ConvergedRelativeEstimateDifference");
        return convergedRelativeEstimateDifference;
    }

    std::string writeWhenConverged() const override
    {
        std::stringstream ss;
        ss << "relative successive estimate difference (" <<
            std::sqrt(squaredAbsoluteDifference / squaredPreviousEvaluation)
            << ") is smaller than the threshold ("
            << param << ") after "
            << (newtonIterationCount+1) << " Newton iterations.";
        return ss.str();
    }
    
    std::string_view measureName() const override
    {
        return "Relative estimate difference";
    }

    SReal squaredAbsoluteDifference = 0;
    SReal squaredPreviousEvaluation = 0;
};

}
