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

#include <MultiThreading/component/solidmechanics/fem/elastic/ParallelTetrahedronFEMForceField.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/task/TaskScheduler.h>
#include <sofa/simulation/task/MainTaskSchedulerFactory.h>
#include <sofa/simulation/task/ParallelForEach.h>

namespace multithreading::component::forcefield::solidmechanics::fem::elastic
{

template<class DataTypes>
void ParallelTetrahedronFEMForceField<DataTypes>::init()
{
    Inherit1::init();
    initTaskScheduler();
}

template <class DataTypes>
void ParallelTetrahedronFEMForceField<DataTypes>::drawTrianglesFromTetrahedra(
    const sofa::core::visual::VisualParams* vparams, bool showVonMisesStressPerElement,
    bool drawVonMisesStress, const VecCoord& x, const VecReal& youngModulus, bool heterogeneous,
    Real minVM, Real maxVM, sofa::helper::ReadAccessor<sofa::Data<sofa::type::vector<Real>>> vM)
{
    this->m_renderedPoints.resize(this->_indexedElements->size() * 3 * 4);
    this->m_renderedColors.resize(this->_indexedElements->size() * 3 * 4);

    const auto showWireFrame = vparams->displayFlags().getShowWireFrame();

    sofa::simulation::parallelForEachRange(*m_taskScheduler, this->_indexedElements->begin(), this->_indexedElements->end(),
        [&](const sofa::simulation::Range<VecElement::const_iterator>& range)
        {
            this->drawTrianglesFromRangeOfTetrahedra(range, vparams, showVonMisesStressPerElement, drawVonMisesStress, showWireFrame, x, youngModulus, heterogeneous, minVM, maxVM, vM);
        });
    vparams->drawTool()->drawTriangles(this->m_renderedPoints, this->m_renderedColors);
}

template<class DataTypes>
void ParallelTetrahedronFEMForceField<DataTypes>::addForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& d_f,
              const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    Inherit1::addForce(mparams, d_f, d_x, d_v);
}

template <class DataTypes>
template <class Function>
void ParallelTetrahedronFEMForceField<DataTypes>::addDForceGeneric(VecDeriv& df, const VecDeriv& dx,
    Real kFactor, const VecElement& indexedElements, Function f)
{
    std::mutex mutex;
    sofa::simulation::parallelForEachRange(*m_taskScheduler, indexedElements.begin(), indexedElements.end(),
           [&indexedElements, this, kFactor, &dx, &df, &f, &mutex](const auto& range)
           {
               auto elementId = std::distance(indexedElements.begin(), range.start);

               VecDeriv& threadLocal_df = m_threadLocal_df[std::this_thread::get_id()];
               threadLocal_df.clear();
               threadLocal_df.resize(df.size());

               for (auto it = range.start; it != range.end; ++it, ++elementId)
               {
                   sofa::Index a = (*it)[0];
                   sofa::Index b = (*it)[1];
                   sofa::Index c = (*it)[2];
                   sofa::Index d = (*it)[3];

                   f( threadLocal_df, dx, elementId, a,b,c,d, kFactor );
               }

               std::lock_guard guard(mutex);

               auto it = df.begin();
               for (const auto& d : threadLocal_df)
               {
                   *it++ += d;
               }
           });
}

template <class DataTypes>
void ParallelTetrahedronFEMForceField<DataTypes>::addDForceSmall(VecDeriv& df, const VecDeriv& dx, const Real kFactor, const VecElement& indexedElements)
{
    addDForceGeneric(df, dx, kFactor, indexedElements,
        [this](VecDeriv& f, const VecDeriv& x, sofa::Index i, sofa::Index a, sofa::Index b, sofa::Index c, sofa::Index d, SReal fact)
        {
            this->applyStiffnessSmall(f, x, i, a, b, c, d, fact);
        });
}

template <class DataTypes>
void ParallelTetrahedronFEMForceField<DataTypes>::addDForceCorotational(VecDeriv& df, const VecDeriv& dx, const Real kFactor, const VecElement& indexedElements)
{
    addDForceGeneric(df, dx, kFactor, indexedElements,
        [this](VecDeriv& f, const VecDeriv& x, sofa::Index i, sofa::Index a, sofa::Index b, sofa::Index c, sofa::Index d, SReal fact)
        {
            this->applyStiffnessCorotational(f, x, i, a, b, c, d, fact);
        });
}

template<class DataTypes>
void ParallelTetrahedronFEMForceField<DataTypes>::addDForce (const sofa::core::MechanicalParams *mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    auto dfAccessor = sofa::helper::getWriteAccessor(d_df);
    VecDeriv& df = dfAccessor.wref();

    const VecDeriv& dx = d_dx.getValue();
    df.resize(dx.size());

    const Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    const auto& indexedElements = *this->_indexedElements;

    if( this->method == Inherit1::SMALL )
    {
        addDForceSmall(df, dx, kFactor, indexedElements);
    }
    else
    {
        addDForceCorotational(df, dx, kFactor, indexedElements);
    }
}

template <class DataTypes>
void ParallelTetrahedronFEMForceField<DataTypes>::addKToMatrix(sofa::linearalgebra::BaseMatrix* mat,
    SReal kFactor, unsigned& offset)
{
    Transformation Rot;
    Rot.identity(); //set the transformation to identity

    const auto& indexedElements = *this->_indexedElements;

    const auto m = this->method;

    std::mutex mutex;

    static constexpr auto S = DataTypes::deriv_total_size; // size of node blocks
    static constexpr auto N = Element::size();
    using Block = sofa::type::fixed_array<sofa::type::fixed_array<sofa::type::Mat<S, S, double>, 4>, 4>;

    sofa::simulation::parallelForEachRange(*m_taskScheduler, indexedElements.begin(), indexedElements.end(),
        [&indexedElements, m, &Rot, this, &offset, mat, &mutex, kFactor](const auto& range)
        {


            StiffnessMatrix JKJt,tmp;

            auto elementId = std::distance(indexedElements.begin(), range.start);

            sofa::type::vector<Block> blocks;
            blocks.reserve(std::distance(range.start, range.end));

            for (auto it = range.start; it != range.end; ++it, ++elementId)
            {
                if (m == Inherit1::SMALL)
                    this->computeStiffnessMatrix(JKJt,tmp, this->materialsStiffnesses[elementId], this->strainDisplacements[elementId],Rot);
                else
                    this->computeStiffnessMatrix(JKJt,tmp, this->materialsStiffnesses[elementId], this->strainDisplacements[elementId], this->rotations[elementId]);

                Block tmpBlock;
                for (sofa::Index n1=0; n1 < N; n1++)
                {
                    for(sofa::Index i=0; i < S; i++)
                    {
                        for (sofa::Index n2=0; n2 < N; n2++)
                        {
                            for (sofa::Index j=0; j < S; j++)
                            {
                                tmpBlock[n1][n2](i,j) = - tmp(n1*S+i,n2*S+j)* kFactor;
                            }
                        }
                    }
                }

                blocks.emplace_back(tmpBlock);
            }

            std::lock_guard guard(mutex);

            auto blockIt = blocks.begin();
            for (auto it = range.start; it != range.end; ++it, ++blockIt)
            {
                const auto& block = *blockIt;
                for (sofa::Index n1=0; n1 < N; n1++)
                {
                    for (sofa::Index n2=0; n2 < N; n2++)
                    {
                        mat->add(offset + (*it)[n1] * S, offset + (*it)[n2] * S, block[n1][n2]);
                    }
                }
            }
        });

}

} //namespace sofa::component::forcefield
