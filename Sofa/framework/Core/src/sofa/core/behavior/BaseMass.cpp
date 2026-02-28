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
#include <sofa/core/BaseMatrixAccumulatorComponent.h>
#include <sofa/core/behavior/BaseMass.h>
#include <sofa/core/objectmodel/BaseNode.h>
#include <sofa/core/behavior/BaseLocalMassMatrix.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/behavior/MatrixAPICompatibility.h>

namespace sofa::core::behavior
{

BaseMass::BaseMass()
    : m_separateGravity (initData(&m_separateGravity , false, "separateGravity", "add separately gravity to velocity computation"))
    , rayleighMass (initData(&rayleighMass , 0_sreal, "rayleighMass", "Rayleigh damping - mass matrix coefficient"))
{
}

void BaseMass::addMDx(const MechanicalParams* mparams, MultiVecDerivId fid, SReal factor)
{
    //TODO (SPRINT SED 2025): Component state mechanism
    doAddMDx(mparams, fid, factor);
}

void BaseMass::accFromF(const MechanicalParams* mparams, MultiVecDerivId aid)
{
    //TODO (SPRINT SED 2025): Component state mechanism
    doAccFromF(mparams, aid);
}

void BaseMass::addGravityToV(const MechanicalParams* mparams, MultiVecDerivId vid)
{
    //TODO (SPRINT SED 2025): Component state mechanism
    doAddGravityToV(mparams, vid);
}

SReal BaseMass::getKineticEnergy(const MechanicalParams* mparams) const
{
    //TODO (SPRINT SED 2025): Component state mechanism
    return doGetKineticEnergy(mparams);
}

SReal BaseMass::getPotentialEnergy(const MechanicalParams* mparams) const
{
    //TODO (SPRINT SED 2025): Component state mechanism
    return doGetPotentialEnergy(mparams);
}

type::Vec6 BaseMass::getMomentum(const MechanicalParams* mparams) const
{
    //TODO (SPRINT SED 2025): Component state mechanism
    return doGetMomentum(mparams);
}

void BaseMass::addMToMatrix(const MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    //TODO (SPRINT SED 2025): Component state mechanism
    doAddMToMatrix(mparams, matrix);
}

void BaseMass::buildMassMatrix(sofa::core::behavior::MassMatrixAccumulator* matrices)
{
    //TODO (SPRINT SED 2025): Component state mechanism
    doBuildMassMatrix(matrices);
}

void BaseMass::initGnuplot(const std::string path)
{
    //TODO (SPRINT SED 2025): Component state mechanism
    doInitGnuplot(path);
}

void BaseMass::exportGnuplot(const MechanicalParams* mparams, SReal time)
{
    //TODO (SPRINT SED 2025): Component state mechanism
    doExportGnuplot(mparams, time);
}

SReal BaseMass::getElementMass(sofa::Index index) const
{
    //TODO (SPRINT SED 2025): Component state mechanism
    return doGetElementMass(index);
}

void BaseMass::getElementMass(sofa::Index index, linearalgebra::BaseMatrix *m) const
{
    //TODO (SPRINT SED 2025): Component state mechanism
    doGetElementMass(index, m);
}

bool BaseMass::insertInNode( objectmodel::BaseNode* node )
{
    node->addMass(this);
    Inherit1::insertInNode(node);
    return true;
}

bool BaseMass::removeInNode( objectmodel::BaseNode* node )
{
    node->removeMass(this);
    Inherit1::removeInNode(node);
    return true;
}

void BaseMass::doBuildMassMatrix(sofa::core::behavior::MassMatrixAccumulator* matrices)
{
    static std::set<BaseMass*> hasEmittedWarning;
    if (hasEmittedWarning.insert(this).second)
    {
        dmsg_warning() << "buildMassMatrix not implemented: for compatibility reason, the "
            "deprecated API (addMToMatrix) will be used. This compatibility will disappear in the "
            "future, and will cause issues in simulations. Please update the code of " <<
            this->getClassName() << " to ensure right behavior: the function addMToMatrix "
            "has been replaced by buildMassMatrix";
    }

    MatrixAccessorCompat accessor;
    accessor.setDoPrintInfo(true);

    const auto& mstates = this->getMechanicalStates();
    std::set<BaseMechanicalState*> uniqueMstates(mstates.begin(), mstates.end());

    for(const auto& mstate1 : uniqueMstates)
    {
        if (mstate1)
        {
            for(const auto& mstate2 : uniqueMstates)
            {
                if (mstate2)
                {
                    const auto mat = std::make_shared<AddToMatrixCompatMatrix<matrixaccumulator::Contribution::MASS> >();
                    mat->component = this;
                    mat->matrices = matrices;
                    mat->mstate1 = mstate1;
                    mat->mstate2 = mstate2;

                    accessor.setMatrix(mstate1, mstate2, mat);
                }
            }
        }
    }

    MechanicalParams params;
    params.setKFactor(1.);
    params.setMFactor(1.);

    addMToMatrix(&params, &accessor);
}


} //  namespace sofa::core::behavior
