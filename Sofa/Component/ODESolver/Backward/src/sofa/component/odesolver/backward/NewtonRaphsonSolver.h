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
#include <sofa/component/odesolver/backward/NonLinearFunction.h>
#include <sofa/component/odesolver/backward/config.h>
#include <sofa/component/odesolver/backward/convergence/NewtonRaphsonConvergenceMeasure.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/helper/map.h>

namespace sofa::component::odesolver::backward
{

class SOFA_COMPONENT_ODESOLVER_BACKWARD_API NewtonRaphsonSolver : public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(NewtonRaphsonSolver, core::objectmodel::BaseObject);

    Data<unsigned int> d_maxNbIterationsNewton;
    Data<SReal> d_relativeSuccessiveStoppingThreshold;
    Data<SReal> d_relativeInitialStoppingThreshold;
    Data<SReal> d_absoluteResidualStoppingThreshold;
    Data<SReal> d_relativeEstimateDifferenceThreshold;
    Data<SReal> d_absoluteEstimateDifferenceThreshold;
    Data<unsigned int> d_maxNbIterationsLineSearch;
    Data<SReal> d_lineSearchCoefficient;
    Data<bool> d_updateStateWhenDiverged;
    Data<NewtonStatus> d_status;
    Data<std::map < std::string, sofa::type::vector<SReal> > > d_residualGraph;
    Data<bool> d_warnWhenLineSearchFails;
    Data<bool> d_warnWhenDiverge;

    void init() override;
    void reset() override;

    /**
     * Main function to call to solve a nonlinear function
     * @param function The nonlinear function to solve
     */
    void solve(newton_raphson::BaseNonLinearFunction& function);

protected:
    NewtonRaphsonSolver();

    void start();

    void initialConvergence(SReal squaredResidualNorm, SReal squaredAbsoluteStoppingThreshold);
    bool measureConvergence(const NewtonRaphsonConvergenceMeasure& measure, std::stringstream& os);

    static void lineSearchIteration(newton_raphson::BaseNonLinearFunction& function,
        SReal& squaredResidualNorm, const SReal lineSearchCoefficient);
};

}
