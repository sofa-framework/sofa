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
#include <SofaBaseMechanics/MechanicalObject.h>

#include "sofa/simulation/Node.h"
#include "SofaBaseTopology/PointSetTopologyModifier.h"

namespace sofa::component::mapping
{

template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::initializeTopologyIndices()
{
    auto& indexPairsAccessor = *helper::getWriteAccessor(indexPairs);
    const auto indexPairSize = indexPairsAccessor.size() / 2;

    for (const auto &[parentId, data] : m_topologyIndices)
    {
        if (data)
        {
            sofa::helper::getWriteAccessor(*data)->clear();
        }
    }

    std::map<unsigned int, sofa::type::vector<unsigned int> > parents;
    for(std::size_t i=0; i<indexPairSize; i++)
    {
        parents[indexPairsAccessor[i * 2]].push_back(indexPairsAccessor[i * 2 + 1]);
    }

    for (const auto& [parentId, indices] : parents)
    {
        auto* d = new sofa::core::topology::TopologySubsetIndices("Parent #" + std::to_string(parentId) + " indices", true, true);
        d->setName("parent"+std::to_string(parentId));
        this->addData(d);
        m_topologyIndices.emplace(parentId, d);

        auto& indicesData = *sofa::helper::getWriteAccessor(*d);
        indicesData.reserve(indices.size());
        for (const auto id : indices)
            indicesData.push_back(id);

        if (parentId < this->fromModels.size())
        {
            if (const auto from = this->fromModels[parentId])
            {
                if (auto* topology = from->getContext()->getMeshTopology())
                {
                    d->createTopologyHandler(topology);
                    d->addTopologyEventCallBack(core::topology::TopologyChangeType::POINTSREMOVED,
                        [this, topology, parentId](const core::topology::TopologyChange* change)
                        {
                            const auto* pointsRemoved = static_cast<const core::topology::PointsRemoved*>(change);
                            msg_info(this) << "Removed points: [" << pointsRemoved->getArray() << "] in parent topology " << topology->getPathName();
                            applyRemovedPoints(pointsRemoved, parentId, topology);
                        });
                }
            }
        }
        else
        {
            msg_error() << "Parent with id " << parentId << " cannot be found: only " << this->fromModels.size() << " parents are available";
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
void SubsetMultiMapping<TIn, TOut>::init()
{
    assert( indexPairs.getValue().size()%2==0 );
    const unsigned indexPairSize = indexPairs.getValue().size()/2;
    this->toModels[0]->resize( indexPairSize );
    if (auto* mobject = dynamic_cast<sofa::component::container::MechanicalObject<Out> *>(this->toModels[0]))
    {
        if (auto* topology = mobject->getTopology())
        {
            topology->setNbPoints(indexPairSize);
        }
    }


    initializeTopologyIndices();
    Inherit::init();

    initializeMappingMatrices();
}

template <class TIn, class TOut>
SubsetMultiMapping<TIn, TOut>::SubsetMultiMapping()
    : Inherit()
    , indexPairs( initData( &indexPairs, type::vector<unsigned>(), "indexPairs", "list of couples (parent index + index in the parent)"))
{}

template <class TIn, class TOut>
SubsetMultiMapping<TIn, TOut>::~SubsetMultiMapping()
{
    for(unsigned i=0; i<baseMatrices.size(); i++ )
    {
        delete baseMatrices[i];
    }
}

template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::applyRemovedPoints(const core::topology::PointsRemoved* pointsRemoved, const unsigned int parentId, core::topology::BaseMeshTopology* parentTopology)
{
    auto& indexPairsValue = *helper::getWriteAccessor(indexPairs);

    //identify points to remove in the target object
    sofa::type::vector<core::topology::Topology::PointID> removedPointsInTarget;
    for(std::size_t i = 0; i < indexPairsValue.size() / 2; i++)
    {
        for (const auto removed : pointsRemoved->getArray())
        {
            if (indexPairsValue[i * 2] == parentId && indexPairsValue[i * 2 + 1] == removed)
            {
                removedPointsInTarget.push_back(i);
            }
        }
    }
    msg_info(this) << "Request to remove points: [" << removedPointsInTarget << "] in target";

    //renumber indexPairs
    auto nbPoints = parentTopology->getNbPoints();
    for (const auto removed : pointsRemoved->getArray())
    {
        --nbPoints;

        sofa::type::vector<core::topology::Topology::PointID> indicesToDelete;
        for(std::size_t i = 0; i < indexPairsValue.size() / 2; i++)
        {
            if (indexPairsValue[i * 2] == parentId && indexPairsValue[i * 2 + 1] == removed)
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
            if (indexPairsValue[i * 2] == parentId && indexPairsValue[i * 2 + 1] == nbPoints)
            {
                indexPairsValue[i * 2 + 1] = removed;
            }
        }
    }

    //propagate the topology changes to the current context through the topology modifier of the Node
    sofa::component::topology::PointSetTopologyModifier* topoMod { nullptr };
    this->getContext()->get(topoMod, core::objectmodel::BaseContext::SearchDirection::Local);
    if (topoMod)
    {
        topoMod->removeItems(removedPointsInTarget);
    }
    else
    {
        msg_error() << "No topology modifier has been found in the context of the target state. Consider adding one in order to support propagation of topological changes through the mapping.";
    }

    //brutal re-init based on the new indexPairs
    Inherit::init();
    initializeMappingMatrices();
}

template <class TIn, class TOut>
const type::vector<sofa::linearalgebra::BaseMatrix*>* SubsetMultiMapping<TIn, TOut>::getJs()
{
    return &baseMatrices;
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
