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

#include <SofaMiscMapping/SubsetMultiMapping.h>

#include <sofa/linearalgebra/EigenSparseMatrix.h>
#include <sofa/core/MappingHelper.h>
#include <sofa/component/topology/container/dynamic/PointSetTopologyModifier.h>

namespace sofa::component::mapping
{

template <class TIn, class TOut>
SubsetMultiMapping<TIn, TOut>::SubsetMultiMapping()
    : Inherit()
    , indexPairs( initData( &indexPairs, type::vector<unsigned>(), "indexPairs", "list of couples (parent index + index in the parent)"))
    , l_inputTopologies( initLink("inputTopologies", "Optional list of topologies matching the list of input states"))
    , l_outputTopologies( initLink("outputTopologies", "Optional list of topologies matching the list of output states"))
    {}

template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::init()
{
    assert( indexPairs.getValue().size()%2==0 );
    const unsigned indexPairSize = indexPairs.getValue().size()/2;

    checkInputOutput();

    this->toModels[0]->resize( indexPairSize );
    initializeOutputTopologies();

    Inherit::init();

    initializeMappingMatrices();
}

template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::checkInputOutput()
{
    msg_error_when(!l_inputTopologies.empty() && this->fromModels.size() != l_inputTopologies.size())<< "Inconsistency "
        "between " << l_inputTopologies.getName() << " (" << l_inputTopologies.size() << " elements) and " <<
        this->fromModels.getName() << " (" << this->fromModels.size() << " elements)";

    msg_warning_when(this->toModels.size() > 1) << "Found " << this->toModels.size() << " mapping output: only the first one will be considered.";
    msg_error_when(!l_outputTopologies.empty() && this->toModels.size() != l_outputTopologies.size()) << "Inconsistency "
        "between " << l_outputTopologies.getName() << " (" << l_outputTopologies.size() << " elements) and " <<
        this->toModels.getName() << " (" << this->toModels.size() << " elements)";

    if (f_printLog.getValue())
    {
        const auto inputSize = std::min(this->fromModels.size(), l_inputTopologies.size());
        for (std::size_t i = 0; i < inputSize; ++i)
        {
            if (l_inputTopologies[i] && this->fromModels[i])
            {
                msg_info() << "[Input " << i+1 << "/" << inputSize << "] Topology " << l_inputTopologies[i]->getPathName() << " is associated to state " << this->fromModels[i]->getPathName();
            }
            else
            {
                msg_info_when(!l_inputTopologies[i]) << "[Input " << i+1 << "/" << inputSize << "] Topology is invalid";
                msg_error_when(!this->fromModels[i]) << "[Input " << i+1 << "/" << inputSize << "] State is invalid";
            }
        }
        const auto outputSize = std::min(this->toModels.size(), l_outputTopologies.size());
        for (std::size_t i = 0; i < outputSize; ++i)
        {
            if (l_outputTopologies[i] && this->toModels[i])
            {
                msg_info() << "[Output " << i+1 << "/" << outputSize << "] Topology " << l_outputTopologies[i]->getPathName() << " is associated to state " << this->toModels[i]->getPathName();
            }
            else
            {
                msg_info_when(!l_outputTopologies[i]) << "[Output " << i+1 << "/" << outputSize << "] Topology is invalid";
                msg_error_when(!this->toModels[i]) << "[Output " << i+1 << "/" << outputSize << "] State is invalid";
            }
        }
    }
}

template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::initializeOutputTopologies()
{
    const unsigned indexPairSize = indexPairs.getValue().size()/2;
    for (auto topology = l_outputTopologies.begin(); topology != l_outputTopologies.end(); ++topology)
    {
        if (*topology)
        {
            msg_info() << "Initialization of output topology " << (*topology)->getPathName() << " with " << indexPairSize << " points";
            (*topology)->setNbPoints(indexPairSize);
        }
        else
        {
            msg_error() << "Found an invalid output topology in links " << l_outputTopologies.getName();
        }
    }
}

template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::initializeMappingMatrices()
{
    const unsigned indexPairSize = indexPairs.getValue().size()/2;
    unsigned Nin = TIn::deriv_total_size, Nout = TOut::deriv_total_size;


    for( unsigned i=0; i<baseMatrices.size(); i++ )
        delete baseMatrices[i];

    typedef linearalgebra::EigenSparseMatrix<TIn,TOut> Jacobian;
    baseMatrices.resize( this->getFrom().size() );
    type::vector<Jacobian*> jacobians( this->getFrom().size() );
    for(unsigned i=0; i<baseMatrices.size(); i++ )
    {
        baseMatrices[i] = jacobians[i] = new linearalgebra::EigenSparseMatrix<TIn,TOut>;
        jacobians[i]->resize(Nout*indexPairSize,Nin*this->fromModels[i]->getSize() ); // each jacobian has the same number of rows
    }

    // fill the Jacobians
    for(unsigned i=0; i<indexPairSize; i++)
    {
        unsigned parent = indexPairs.getValue()[i*2];
        Jacobian* jacobian = jacobians[parent];
        unsigned bcol = indexPairs.getValue()[i*2+1];  // parent particle
        for(unsigned k=0; k<Nout; k++ )
        {
            unsigned row = i*Nout + k;

            // the Jacobian could be filled in order, but empty rows should have to be managed
            jacobian->add( row, Nin*bcol +k, (SReal)1. );
        }
    }
    // finalize the Jacobians
    for(unsigned i=0; i<baseMatrices.size(); i++ )
        baseMatrices[i]->compress();
}

template <class TIn, class TOut>
SubsetMultiMapping<TIn, TOut>::~SubsetMultiMapping()
{
    for(unsigned i=0; i<baseMatrices.size(); i++ )
    {
        delete baseMatrices[i];
    }
}

template <class TIn, class TOut>
const type::vector<sofa::linearalgebra::BaseMatrix*>* SubsetMultiMapping<TIn, TOut>::getJs()
{
    return &baseMatrices;
}

template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::propagateRemovedPoints(const sofa::type::vector<core::topology::Topology::PointID>& removedPointsInOutput)
{
    unsigned int nbValidTopologies = 0;
    for (auto topology = l_outputTopologies.begin(); topology != l_outputTopologies.end(); ++topology)
    {
        if (*topology)
        {
            ++nbValidTopologies;
            sofa::component::topology::container::dynamic::PointSetTopologyModifier *modifier;
            (*topology)->getContext()->get(modifier);

            if (modifier)
            {
                modifier->removeItems(removedPointsInOutput);
            }
            else
            {
                msg_error() << "A topological change occurs, but no topology modifier has been found in the context of "
                    "the topology " << (*topology)->getPathName() << ": the topological change will not be properly "
                    "propagated";
            }
        }
    }
    msg_error_when(nbValidTopologies == 0) << "A topological change occurs, but no output topology has been provided: the "
        "topological change will not be properly propagated";
}

template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::propagateEndingEvent()
{
    for (auto topology = l_outputTopologies.begin(); topology != l_outputTopologies.end(); ++topology)
    {
        if (*topology)
        {
            sofa::component::topology::container::dynamic::PointSetTopologyModifier *modifier;
            (*topology)->getContext()->get(modifier);

            if (modifier)
            {
                modifier->notifyEndingEvent();
            }
        }
    }
}

template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::applyRemovePoints(const core::topology::TopologyChange* topoChange, core::topology::BaseMeshTopology* inputTopology)
{
    const auto *pRem = static_cast< const core::topology::PointsRemoved * >( topoChange );
    const sofa::type::vector<Index> tab = pRem->getArray();

    auto& indexPairsValue = *helper::getWriteAccessor(indexPairs);

    const auto inputId = std::distance(l_inputTopologies.begin(),
        std::find_if(l_inputTopologies.begin(), l_inputTopologies.end(), [inputTopology](const auto t) { return t.get() == inputTopology;}));

    //identify points to remove in the output object
    sofa::type::vector<core::topology::Topology::PointID> removedPointsInOutput;
    for(std::size_t i = 0; i < indexPairsValue.size() / 2; i++)
    {
        for (const auto removed : tab)
        {
            if (indexPairsValue[i * 2] == inputId && indexPairsValue[i * 2 + 1] == removed)
            {
                removedPointsInOutput.push_back(i);
            }
        }
    }
    msg_info(this) << "Request to remove points: [" << removedPointsInOutput << "] in target";

    //renumber indexPairs
    auto nbPoints = inputTopology->getNbPoints();
    for (const auto removed : tab)
    {
        --nbPoints;

        sofa::type::vector<core::topology::Topology::PointID> indicesToDelete;
        for(std::size_t i = 0; i < indexPairsValue.size() / 2; i++)
        {
            if (indexPairsValue[i * 2] == inputId && indexPairsValue[i * 2 + 1] == removed)
            {
                indicesToDelete.push_back(i);
            }
        }

        for (auto it = indicesToDelete.rbegin(); it != indicesToDelete.rend(); ++it)
        {
            indexPairsValue.erase(indexPairsValue.begin() + 2 * (*it) + 1);
            indexPairsValue.erase(indexPairsValue.begin() + 2 * (*it));
        }

        if (removed == nbPoints)
            continue;

        for(std::size_t i = 0; i < indexPairsValue.size() / 2; i++)
        {
            if (indexPairsValue[i * 2] == inputId && indexPairsValue[i * 2 + 1] == nbPoints)
            {
                indexPairsValue[i * 2 + 1] = removed;
            }
        }
    }

    //propagate the topology changes to the current context through the topology modifier of the Node
    propagateRemovedPoints(removedPointsInOutput);

    //brutal re-init based on the new indexPairs
    Inherit::init();
    initializeMappingMatrices();
}

template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::updateTopologicalMappingTopDown()
{
    for (auto topology = l_inputTopologies.begin(); topology != l_inputTopologies.end(); ++topology)
    {
        if (*topology)
        {
            auto itBegin = (*topology)->beginChange();
            auto itEnd = (*topology)->endChange();

            if (itBegin != itEnd)
            {
                msg_info() << "Topological change from " << (*topology)->getPathName();

                while( itBegin != itEnd )
                {
                    const core::topology::TopologyChange* topoChange = *itBegin++;

                    switch(topoChange->getChangeType())
                    {
                    case core::topology::ENDING_EVENT:
                    {
                        propagateEndingEvent();
                        break;
                    }
                    case core::topology::POINTSREMOVED:
                    {
                        applyRemovePoints(topoChange, *topology);
                        break;
                    }
                    default:
                        break;
                    }
                }
            }
        }
    }
}

template <class TIn, class TOut>
bool SubsetMultiMapping<TIn, TOut>::isTopologyAnInput(core::topology::Topology* topology)
{
    return std::any_of(l_inputTopologies.begin(), l_inputTopologies.end(), [topology](const auto t) { return t.get() == topology;});
}

template <class TIn, class TOut>
bool SubsetMultiMapping<TIn, TOut>::isTopologyAnOutput(core::topology::Topology* topology)
{
    return std::any_of(l_outputTopologies.begin(), l_outputTopologies.end(), [topology](const auto t) { return t.get() == topology;});
}

template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::addPoint( const core::BaseState* from, int index)
{

    // find the index of the parent state
    unsigned i;
    for(i=0; i<this->fromModels.size(); i++)
        if(this->fromModels.get(i)==from )
            break;
    if(i==this->fromModels.size())
    {
        msg_error() << "SubsetMultiMapping<TIn, TOut>::addPoint, parent " << from->getName() << " not found !";
        assert(0);
    }

    addPoint(i, index);
}

template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::addPoint( int from, int index)
{
    assert((size_t)from < this->fromModels.size());
    type::vector<unsigned>& indexPairsVector = *indexPairs.beginEdit();
    indexPairsVector.push_back(from);
    indexPairsVector.push_back(index);
    indexPairs.endEdit();
}


template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::apply(const core::MechanicalParams* mparams, const type::vector<OutDataVecCoord*>& dataVecOutPos, const type::vector<const InDataVecCoord*>& dataVecInPos)
{
    SOFA_UNUSED(mparams);

    OutVecCoord& out = *(dataVecOutPos[0]->beginEdit());

    for(unsigned i=0; i<out.size(); i++)
    {
        const InDataVecCoord* inPosPtr = dataVecInPos[indexPairs.getValue()[i*2]];
        const InVecCoord& inPos = (*inPosPtr).getValue();

        core::eq( out[i], inPos[indexPairs.getValue()[i*2+1]] );
    }

    dataVecOutPos[0]->endEdit();

}

template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::applyJ(const core::MechanicalParams* mparams, const type::vector<OutDataVecDeriv*>& dataVecOutVel, const type::vector<const InDataVecDeriv*>& dataVecInVel)
{
    SOFA_UNUSED(mparams);

    OutVecDeriv& out = *(dataVecOutVel[0]->beginEdit());

    for(unsigned i=0; i<out.size(); i++)
    {
        const InDataVecDeriv* inDerivPtr = dataVecInVel[indexPairs.getValue()[i*2]];
        const InVecDeriv& inDeriv = (*inDerivPtr).getValue();
        core::eq( out[i], inDeriv[indexPairs.getValue()[i*2+1]] );
    }

    dataVecOutVel[0]->endEdit();
}

template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::applyJT( const core::ConstraintParams* /*cparams*/, const type::vector< InDataMatrixDeriv* >& dOut, const type::vector< const OutDataMatrixDeriv* >& dIn)
{
    type::vector<unsigned>  indexP = indexPairs.getValue();

    // hypothesis: one child only:
    const OutMatrixDeriv& in = dIn[0]->getValue();

    if (dOut.size() != this->fromModels.size())
    {
        msg_error() << "Problem with number of output constraint matrices";
        return;
    }

    typename OutMatrixDeriv::RowConstIterator rowItEnd = in.end();
    // loop on the constraints defined on the child of the mapping
    for (typename OutMatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {

        typename OutMatrixDeriv::ColConstIterator colIt = rowIt.begin();
        typename OutMatrixDeriv::ColConstIterator colItEnd = rowIt.end();


        // A constraint can be shared by several nodes,
        // these nodes can be linked to 2 different parents. // TODO handle more parents
        // we need to add a line to each parent that is concerned by the constraint


        while (colIt != colItEnd)
        {
            unsigned int index_parent = indexP[colIt.index()*2]; // 0 or 1 (for now...)
            // writeLine provide an iterator on the line... if this line does not exist, the line is created:
            typename InMatrixDeriv::RowIterator o = dOut[index_parent]->beginEdit()->writeLine(rowIt.index());
            dOut[index_parent]->endEdit();

            // for each col of the constraint direction, it adds a col in the corresponding parent's constraint direction
            if(indexPairs.getValue()[colIt.index()*2+1] < (unsigned int)this->fromModels[index_parent]->getSize())
            {
                InDeriv tmp; core::eq( tmp, colIt.val() );
                o.addCol(indexP[colIt.index()*2+1], tmp);
            }
            ++colIt;
        }



    }
}


template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::applyJT(const core::MechanicalParams* mparams, const type::vector<InDataVecDeriv*>& dataVecOutForce, const type::vector<const OutDataVecDeriv*>& dataVecInForce)
{
    SOFA_UNUSED(mparams);

    const OutDataVecDeriv* cderData = dataVecInForce[0];
    const OutVecDeriv& cder = cderData->getValue();

    for(unsigned i=0; i<cder.size(); i++)
    {
        InDataVecDeriv* inDerivPtr = dataVecOutForce[indexPairs.getValue()[i*2]];
        InVecDeriv& inDeriv = *(*inDerivPtr).beginEdit();
        core::peq( inDeriv[indexPairs.getValue()[i*2+1]], cder[i] );
        (*inDerivPtr).endEdit();
    }
}

} // namespace sofa::component::mapping
