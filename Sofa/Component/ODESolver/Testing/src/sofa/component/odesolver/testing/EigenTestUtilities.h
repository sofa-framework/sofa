/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include <sofa/linearalgebra/FullVector.h>
#include <sofa/simulation/Visitor.h>
#include <sofa/core/VecId.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <Eigen/Dense>

namespace sofa::component::odesolver::testing
{

class GetVectorVisitor : public sofa::simulation::Visitor
{
public:
    GetVectorVisitor(const sofa::core::ExecParams* params, linearalgebra::BaseVector* vec, core::ConstVecId src)
        : sofa::simulation::Visitor(params), vec(vec), src(src) {}

    ~GetVectorVisitor() override = default;

    Result processNodeTopDown(simulation::Node* gnode) override
    {
        if (gnode->mechanicalState != nullptr && (gnode->mechanicalMapping == nullptr || independentOnly == false))
        {
            gnode->mechanicalState->copyToBaseVector(vec, src, offset);
        }
        return Visitor::RESULT_CONTINUE;
    }
    const char* getClassName() const override { return "GetVectorVisitor"; }

    /// If true, process the independent nodes only
    void setIndependentOnly(bool b) { independentOnly = b; }

protected:
    linearalgebra::BaseVector* vec;
    core::ConstVecId src;
    unsigned offset{0};
    bool independentOnly{false};

};

class GetAssembledSizeVisitor : public sofa::simulation::Visitor
{
public:
    GetAssembledSizeVisitor(const sofa::core::ExecParams* params = sofa::core::mechanicalparams::castToExecParams(core::mechanicalparams::defaultInstance())) 
        : sofa::simulation::Visitor(params)
    {};
    ~GetAssembledSizeVisitor() override {}

    Result processNodeTopDown(simulation::Node* gnode) override
    {
        if (gnode->mechanicalState != nullptr && (gnode->mechanicalMapping == nullptr || independentOnly == false))
        {
            xsize += gnode->mechanicalState->getSize() * gnode->mechanicalState->getCoordDimension();
            vsize += gnode->mechanicalState->getMatrixSize();
        }
        return Visitor::RESULT_CONTINUE;
    }
    const char* getClassName() const override { return "GetAssembledSizeVisitor"; }

    unsigned positionSize() const { return xsize; }
    unsigned velocitySize() const { return vsize; }
    void setIndependentOnly(bool b) { independentOnly = b; }

protected:
    std::size_t xsize{ 0 };
    std::size_t vsize{ 0 };
    bool independentOnly{ false };
};

inline Eigen::VectorXd getVector(simulation::Node::SPtr root, core::ConstVecId id, bool indep = true )
{
    GetAssembledSizeVisitor getSizeVisitor;
    getSizeVisitor.setIndependentOnly(indep);
    root->execute(getSizeVisitor);
    unsigned size;
    if (id.type == sofa::core::V_COORD)
        size =  getSizeVisitor.positionSize();
    else
        size = getSizeVisitor.velocitySize();
    linearalgebra::FullVector<SReal> v(size);
    GetVectorVisitor getVec( sofa::core::mechanicalparams::castToExecParams(core::mechanicalparams::defaultInstance()), &v, id);
    getVec.setIndependentOnly(indep);
    root->execute(getVec);

    Eigen::VectorXd ve(size);
    for(size_t i=0; i<size; i++)
        ve(i)=v[i];
    return ve;
}

} /// sofa
