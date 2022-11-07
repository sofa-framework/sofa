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

#include <MultiThreading/ParallelHexahedronFEMForceField.h>
#include <sofa/simulation/TaskScheduler.h>

namespace sofa::component::forcefield
{

template<class DataTypes>
void ParallelHexahedronFEMForceField<DataTypes>::init()
{
    Inherit1::init();
    initTaskScheduler();
}

template<class DataTypes>
void ParallelHexahedronFEMForceField<DataTypes>::initTaskScheduler()
{
    auto* taskScheduler = sofa::simulation::TaskScheduler::getInstance();
    assert(taskScheduler != nullptr);
    if (taskScheduler->getThreadCount() < 1)
    {
        taskScheduler->init(0);
        msg_info() << "Task scheduler initialized on " << taskScheduler->getThreadCount() << " threads";
    }
    else
    {
        msg_info() << "Task scheduler already initialized on " << taskScheduler->getThreadCount() << " threads";
    }
}

template<class DataTypes>
void ParallelHexahedronFEMForceField<DataTypes>::addForce(const core::MechanicalParams* mparams, DataVecDeriv& f,
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

    auto *taskScheduler = sofa::simulation::TaskScheduler::getInstance();
    assert(taskScheduler != nullptr);
    if (taskScheduler->getThreadCount() == 0)
    {
        dmsg_error() << "Task scheduler not correctly initialized";
        return;
    }

    //status that will be used to check if all tasks are over
    sofa::simulation::CpuTask::Status status;

    //total number of unit tasks to perform
    const unsigned int nbElements = this->getIndexedElements()->size();

    //number of threads is based on the number of threads in the thread pools
    const auto nbThreads = std::min(taskScheduler->getThreadCount(), nbElements);

    //how many unit tasks are performed per thread
    const auto nbElementsPerThread = nbElements / nbThreads;

    //iterators used in each thread to access elements
    auto first = this->getIndexedElements()->begin();
    auto last = first + nbElementsPerThread;

    this->m_potentialEnergy = 0;

    //the number of tasks given to the task scheduler will be nbThreads
    m_accumulateForceLargeTasks.reserve(nbThreads);

    const auto& elementStiffnesses = this->_elementStiffnesses.getValue();
    updateStiffnessMatrices = this->f_updateStiffnessMatrix.getValue();
    if (updateStiffnessMatrices)
    {
        static bool first = true;
        msg_warning_when(first) << "Updating the stiffness matrix is not fully supported in a multithreaded context. "
                                   "It may lead to a crash. "
                                   "Set 'updateStiffnessMatrix' to false to remove this warning.";
        first = false;
    }

    // Create and add the tasks to the task scheduler
    for (unsigned int i = 0; i < nbThreads; ++i)
    {
        if (i == nbThreads - 1)
        {
            last = this->getIndexedElements()->end();
        }
        m_accumulateForceLargeTasks.emplace_back(&status, this, elementStiffnesses, first, last, _p, i * nbElementsPerThread);
        taskScheduler->addTask(&m_accumulateForceLargeTasks.back());
        first += nbElementsPerThread;
        last += nbElementsPerThread;
    }

    // Wait that all tasks are over
    taskScheduler->workUntilDone(&status);

    // Gather the result of each task
    for (const auto& task : m_accumulateForceLargeTasks)
    {
        this->m_potentialEnergy += task.getPotentialEnergyOutput();
        auto elementIt = task.m_first; //first element processed by this task
        for (const auto& outF : task.getFOutput())
        {
            for (int w = 0; w < 8; ++w)
            {
                _f[(*elementIt)[w]] += outF[w];
            }
            ++elementIt;
        }
    }

    this->m_potentialEnergy/=-2.0;
    m_accumulateForceLargeTasks.clear();

}

template<class DataTypes>
void ParallelHexahedronFEMForceField<DataTypes>::computeTaskForceLarge(RDataRefVecCoord &p,
                                                                      sofa::Index elementId,
                                                                      const Element& elem,
                                                                      const VecElementStiffness& elementStiffnesses,
                                                                      SReal& OutPotentialEnery,
                                                                      type::Vec<8, Deriv>& OutF)
{
    type::Vec<8,Coord> nodes;
    for(int w=0; w<8; ++w)
        nodes[w] = p[elem[w]];

    Coord horizontal;
    horizontal = (nodes[1]-nodes[0] + nodes[2]-nodes[3] + nodes[5]-nodes[4] + nodes[6]-nodes[7])*.25;
    Coord vertical;
    vertical = (nodes[3]-nodes[0] + nodes[2]-nodes[1] + nodes[7]-nodes[4] + nodes[6]-nodes[5])*.25;

    // 	Transformation R_0_2; // Rotation matrix (deformed and displaced Tetrahedron/world)
    this->computeRotationLarge(this->_rotations[elementId], horizontal, vertical);

    // positions of the deformed and displaced Tetrahedron in its frame
    type::Vec<8,Coord> deformed;
    for(int w=0; w<8; ++w)
        deformed[w] = this->_rotations[elementId] * nodes[w];


    // displacement
    type::Vec<24, Real> D;
    for(int k=0 ; k<8 ; ++k )
    {
        int indice = k*3;
        for(int j=0 ; j<3 ; ++j )
            D[indice+j] = this->_rotatedInitialElements[elementId][k][j] - deformed[k][j];
    }

    if(updateStiffnessMatrices)
    {
        //Not thread safe: a message warned the user earlier that updating the stiffness matrices is not fully supported
        this->computeElementStiffness((*this->_elementStiffnesses.beginEdit())[elementId], this->_materialsStiffnesses[elementId], deformed, elementId, this->_sparseGrid ? this->_sparseGrid->getStiffnessCoef(elementId) : 1.0 );
    }

    type::Vec<24, Real> F; //forces
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
void ParallelHexahedronFEMForceField<DataTypes>::addDForce (const core::MechanicalParams *mparams, DataVecDeriv& v, const DataVecDeriv& x)
{
    WDataRefVecDeriv _df = v;
    RDataRefVecCoord _dx = x;
    const Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    if (_df.size() != _dx.size())
        _df.resize(_dx.size());

    auto *taskScheduler = sofa::simulation::TaskScheduler::getInstance();
    assert(taskScheduler != nullptr);
    if (taskScheduler->getThreadCount() == 0)
    {
        msg_error() << "Task scheduler not correctly initialized";
        return;
    }

    //total number of unit tasks to perform
    const unsigned int nbElements = this->getIndexedElements()->size();

    //number of threads is based on the number of threads in the thread pools
    const auto nbThreads = std::min(taskScheduler->getThreadCount(), nbElements);

    //the number of tasks given to the task scheduler will be nbThreads
    m_addDForceTasks.reserve(nbThreads);

    //how many unit tasks are performed per thread
    const auto nbElementsPerThread = nbElements / nbThreads;

    //iterators used in each thread to access elements
    auto first = this->getIndexedElements()->begin();
    auto last = first + nbElementsPerThread;

    //status that will be used to check if all tasks are over
    sofa::simulation::CpuTask::Status status;

    const auto& elementStiffnesses = this->_elementStiffnesses.getValue();

    // Create and add the tasks to the task scheduler
    for (unsigned int i = 0; i < nbThreads; ++i)
    {
        if (i == nbThreads - 1)
        {
            last = this->getIndexedElements()->end();
        }
        m_addDForceTasks.emplace_back(&status, this, elementStiffnesses, first, last, kFactor, _dx, i * nbElementsPerThread);
        taskScheduler->addTask(&m_addDForceTasks.back());
        first += nbElementsPerThread;
        last += nbElementsPerThread;
    }

    // Wait that all tasks are over
    taskScheduler->workUntilDone(&status);

    // Gather the result of each task
    for (const auto& task : m_addDForceTasks)
    {
        auto it = task.m_first;
        for (const auto& outDf : task.getDfOutput())
        {
            for (int w = 0 ; w < 8; ++w)
            {
                _df[(*it)[w]] += outDf[w];
            }
            ++it;
        }
    }
    m_addDForceTasks.clear();
}

template<class DataTypes>
sofa::simulation::Task::MemoryAlloc AccumulateForceLargeTasks<DataTypes>::run()
{
    auto elementId = m_startingElementId;
    auto it = m_first;
    auto outFIt = m_outF.begin();

    while (it != m_last)
    {
        m_ff->computeTaskForceLarge(m_p, elementId++, *it++, m_elementStiffnesses, m_potentialEnergy, *outFIt++);
    }
    return simulation::Task::Stack;
}

template<class DataTypes>
AccumulateForceLargeTasks<DataTypes>::AccumulateForceLargeTasks(sofa::simulation::CpuTask::Status* status,
                                                                ParallelHexahedronFEMForceField<DataTypes>* ff,
                                                                const typename ParallelHexahedronFEMForceField<DataTypes>::VecElementStiffness& elementStiffnesses,
                                                                VecElement::const_iterator first, VecElement::const_iterator last,
                                                                RDataRefVecCoord& p,
                                                                sofa::Index startingElementId)
    : sofa::simulation::CpuTask(status)
    , m_first(first)
    , m_last(last)
    , m_outF()
    , m_ff(ff)
    , m_elementStiffnesses(elementStiffnesses)
    , m_p(p)
    , m_startingElementId(startingElementId)
{
    m_outF.resize(std::distance(first, last));
}

template<class DataTypes>
sofa::simulation::Task::MemoryAlloc AddDForceTask<DataTypes>::run()
{
    auto elementId = m_startingElementId;
    auto it = m_first;

    type::Vec<24, Real> X; //displacement
    type::Vec<24, Real> F; //force

    auto outDfIt = m_outDf.begin();

    while (it != m_last)
    {
        const auto& element = *it;
        for (int w = 0; w < 8; ++w)
        {
            Coord x_2 = m_ff->_rotations[elementId] * m_dx[element[w]];

            X[w * 3] = x_2[0];
            X[w * 3 + 1] = x_2[1];
            X[w * 3 + 2] = x_2[2];
        }

        // F = K * X
        m_ff->computeForce(F, X, m_elementStiffnesses[elementId]);

        for (int w = 0; w < 8; ++w)
        {
            (*outDfIt)[w] -= m_ff->_rotations[elementId].multTranspose(Deriv(F[w * 3], F[w * 3 + 1], F[w * 3 + 2])) * m_kFactor;
        }

        ++elementId;
        ++it;
        ++outDfIt;
    }

    return simulation::Task::Stack;
}

template<class DataTypes>
AddDForceTask<DataTypes>::AddDForceTask(sofa::simulation::CpuTask::Status* status,
                                        ParallelHexahedronFEMForceField<DataTypes>* ff,
                                        const typename ParallelHexahedronFEMForceField<DataTypes>::VecElementStiffness& elementStiffnesses,
                                        VecElement::const_iterator first, VecElement::const_iterator last, Real kFactor,
                                        RDataRefVecCoord& dx, sofa::Index startingElementId)
        : CpuTask(status)
        , m_first(first)
        , m_last(last)
        , m_outDf()
        , m_ff(ff)
        , m_elementStiffnesses(elementStiffnesses)
        , m_startingElementId(startingElementId)
        , m_dx(dx)
        , m_kFactor(kFactor)
{
    m_outDf.resize(std::distance(first, last));
}

} //namespace sofa::component::forcefield
