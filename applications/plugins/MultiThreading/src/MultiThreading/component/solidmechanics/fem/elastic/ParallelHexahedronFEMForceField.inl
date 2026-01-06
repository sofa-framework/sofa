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

#include <MultiThreading/component/solidmechanics/fem/elastic/ParallelHexahedronFEMForceField.h>
#include <sofa/simulation/task/TaskScheduler.h>
#include <sofa/simulation/task/MainTaskSchedulerFactory.h>
#include <sofa/simulation/task/ParallelForEach.h>

namespace multithreading::component::forcefield::solidmechanics::fem::elastic
{

template<class DataTypes>
void ParallelHexahedronFEMForceField<DataTypes>::init()
{
    Inherit1::init();
    initTaskScheduler();

    const auto indexedElements = *this->getIndexedElements();

    m_vertexIdInAdjacentHexahedra.resize(this->l_topology->getNbPoints());
    m_around.clear();
    for (sofa::Size i = 0; i < this->l_topology->getNbPoints(); ++i)
    {
        const auto& around = this->l_topology->getHexahedraAroundVertex(i);
        m_around.push_back(around);

        sofa::Size j {};
        for (const auto hexaId : around)
        {
            const auto element = indexedElements[hexaId];

            sofa::Size indexInElement {};
            for (const auto v : element)
            {
                if (v == i)
                {
                    m_vertexIdInAdjacentHexahedra[i][j] = indexInElement;
                }
                ++indexInElement;
            }
            ++j;
        }

    }
}

template<class DataTypes>
void ParallelHexahedronFEMForceField<DataTypes>::addForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& f,
              const DataVecCoord& p, const DataVecDeriv& v)
{
    using namespace sofa::component::solidmechanics::fem::elastic;
    if (this->method != HexahedronFEMForceField<DataTypes>::LARGE)
    {
        static bool firstTime = true;
        msg_warning_when(firstTime) << "Multithreading is only partially supported for other methods than 'large'";
        firstTime = false;
        HexahedronFEMForceField<DataTypes>::addForce(mparams, f, p, v);
        return;
    }

    WDataRefVecDeriv _f = f;
    RDataRefVecCoord _p = p;

    _f.resize(_p.size());

    if (this->needUpdateTopology)
    {
        this->reinit();
        this->needUpdateTopology = false;
    }

    const auto* indexedElements = this->getIndexedElements();
    this->m_potentialEnergy = 0;

    const auto& elementStiffnesses = this->d_elementStiffnesses.getValue();
    updateStiffnessMatrices = this->d_updateStiffnessMatrix.getValue();
    if (updateStiffnessMatrices)
    {
        static bool first = true;
        msg_warning_when(first) << "Updating the stiffness matrix is not fully supported in a multithreaded context. "
                                   "It may lead to a crash. "
                                   "Set 'updateStiffnessMatrix' to false to remove this warning.";
        first = false;
    }

    std::mutex mutex;

    sofa::simulation::parallelForEachRange(*m_taskScheduler,
        indexedElements->begin(), indexedElements->end(),
        [this, &_p, &elementStiffnesses, &mutex, &_f](const auto& range)
        {
            auto elementId = std::distance(this->getIndexedElements()->begin(), range.start);

            std::vector<sofa::type::Vec<8, Deriv>> fElements;
            fElements.reserve(std::distance(range.start, range.end));

            SReal potentialEnergy { 0_sreal };
            for (auto it = range.start; it != range.end; ++it, ++elementId)
            {
                sofa::type::Vec<8, Deriv> forceInElement;
                this->computeTaskForceLarge(_p, elementId, *it, elementStiffnesses, potentialEnergy, forceInElement);
                fElements.emplace_back(forceInElement);
            }

            std::lock_guard guard(mutex);

            this->m_potentialEnergy += potentialEnergy;

            auto it = range.start;
            for (const auto& forceInElement : fElements)
            {
                for (int w = 0; w < 8; ++w)
                {
                    _f[(*it)[w]] += forceInElement[w];
                }
                ++it;
            }
        });

    this->m_potentialEnergy/=-2.0;
}

template<class DataTypes>
void ParallelHexahedronFEMForceField<DataTypes>::computeTaskForceLarge(RDataRefVecCoord &p,
                                                                      sofa::Index elementId,
                                                                      const Element& elem,
                                                                      const VecElementStiffness& elementStiffnesses,
                                                                      SReal& OutPotentialEnery,
                                                                      sofa::type::Vec<8, Deriv>& OutF)
{
    sofa::type::Vec<8,Coord> nodes;
    for(int w=0; w<8; ++w)
        nodes[w] = p[elem[w]];

    Coord horizontal;
    horizontal = (nodes[1]-nodes[0] + nodes[2]-nodes[3] + nodes[5]-nodes[4] + nodes[6]-nodes[7])*.25;
    Coord vertical;
    vertical = (nodes[3]-nodes[0] + nodes[2]-nodes[1] + nodes[7]-nodes[4] + nodes[6]-nodes[5])*.25;

    // 	Transformation R_0_2; // Rotation matrix (deformed and displaced Tetrahedron/world)
    this->computeRotationLarge(this->_rotations[elementId], horizontal, vertical);

    // positions of the deformed and displaced Tetrahedron in its frame
    sofa::type::Vec<8,Coord> deformed;
    for(int w=0; w<8; ++w)
        deformed[w] = this->_rotations[elementId] * nodes[w];


    // displacement
    sofa::type::Vec<24, Real> D;
    for(int k=0 ; k<8 ; ++k )
    {
        const int index = k*3;
        for(int j=0 ; j<3 ; ++j )
            D[index+j] = this->_rotatedInitialElements[elementId][k][j] - deformed[k][j];
    }

    if(updateStiffnessMatrices)
    {
        //Not thread safe: a message warned the user earlier that updating the stiffness matrices is not fully supported
        this->computeElementStiffness((*this->d_elementStiffnesses.beginEdit())[elementId], this->_materialsStiffnesses[elementId], deformed, elementId, this->_sparseGrid ? this->_sparseGrid->getStiffnessCoef(elementId) : 1.0 );
    }

    sofa::type::Vec<24, Real> F; //forces
    this->computeForce( F, D, elementStiffnesses[elementId] ); // compute force on element

    for(int w=0; w<8; ++w)
        OutF[w] += this->_rotations[elementId].multTranspose(Deriv(F[w * 3], F[w * 3 + 1], F[w * 3 + 2]  ) );

    OutPotentialEnery += dot(Deriv( F[0], F[1], F[2] ) ,-Deriv( D[0], D[1], D[2]));
    OutPotentialEnery += dot(Deriv( F[3], F[4], F[5] ) ,-Deriv( D[3], D[4], D[5] ));
    OutPotentialEnery += dot(Deriv( F[6], F[7], F[8] ) ,-Deriv( D[6], D[7], D[8] ));
    OutPotentialEnery += dot(Deriv( F[9], F[10], F[11]),-Deriv( D[9], D[10], D[11] ));
    OutPotentialEnery += dot(Deriv( F[12], F[13], F[14]),-Deriv( D[12], D[13], D[14] ));
    OutPotentialEnery += dot(Deriv( F[15], F[16], F[17]),-Deriv( D[15], D[16], D[17] ));
    OutPotentialEnery += dot(Deriv( F[18], F[19], F[20]),-Deriv( D[18], D[19], D[20] ));
    OutPotentialEnery += dot(Deriv( F[21], F[22], F[23]),-Deriv( D[21], D[22], D[23] ));
}


template<class DataTypes>
void ParallelHexahedronFEMForceField<DataTypes>::addDForce (const sofa::core::MechanicalParams *mparams, DataVecDeriv& v, const DataVecDeriv& x)
{
    WDataRefVecDeriv _df = v;
    RDataRefVecCoord _dx = x;
    const Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    if (_df.size() != _dx.size())
        _df.resize(_dx.size());

    const auto& elementStiffnesses = this->d_elementStiffnesses.getValue();
    const auto& indexedElements = *this->getIndexedElements();

    m_elementsDf.resize(indexedElements.size());

    sofa::simulation::parallelForEachRange(*m_taskScheduler,
         indexedElements.begin(), indexedElements.end(),
         [this, &_dx, &elementStiffnesses, kFactor, &indexedElements](const auto& range)
         {
             auto elementId = std::distance(indexedElements.begin(), range.start);
             auto elementsDfIt = m_elementsDf.begin() + elementId;
             auto elementStiffnessesIt = elementStiffnesses.begin() + elementId;
             auto rotationIt = this->_rotations.begin() + elementId;

             for (auto it = range.start; it != range.end; ++it, ++elementId)
             {
                 sofa::type::Vec<24, Real> X(sofa::type::NOINIT); //displacement
                 sofa::type::Vec<24, Real> F(sofa::type::NOINIT); //force

                 const auto& r = *rotationIt++;

                 const auto& element = *it;
                 for (sofa::Size w = 0; w < 8; ++w)
                 {
                     const Coord x_2 = r * _dx[element[w]];

                     X[w * 3] = x_2[0];
                     X[w * 3 + 1] = x_2[1];
                     X[w * 3 + 2] = x_2[2];
                 }

                 // F = K * X
                 this->computeForce(F, X, *elementStiffnessesIt++);

                 sofa::type::Vec<8, Deriv>& df = *elementsDfIt++;
                 for (sofa::Size w = 0; w < 8; ++w)
                 {
                     df[w] = -r.multTranspose(Deriv(F[w * 3], F[w * 3 + 1], F[w * 3 + 2])) * kFactor;
                 }
             }
         });

    sofa::simulation::parallelForEachRange(*m_taskScheduler,
        static_cast<std::size_t>(0), _df.size(), [&_df, this](const auto& range)
        {
            for (auto vertexId = range.start; vertexId < range.end; ++vertexId)
            {
                const auto& around = m_around[vertexId];

                sofa::Size hexaAroundId {};
                for (const auto hexaId : around)
                {
                    _df[vertexId] += m_elementsDf[hexaId][m_vertexIdInAdjacentHexahedra[vertexId][hexaAroundId]];
                    hexaAroundId++;
                }
            }
        });
}

} //namespace sofa::component::forcefield
