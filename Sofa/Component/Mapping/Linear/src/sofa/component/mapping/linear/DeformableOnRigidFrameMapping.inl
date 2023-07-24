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

#include <sofa/component/mapping/linear/DeformableOnRigidFrameMapping.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/topology/TopologyDataHandler.h>

namespace sofa::component::mapping::linear
{

template <class TIn, class TInRoot, class TOut>
DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::DeformableOnRigidFrameMapping()
    : index ( initData ( &index, ( unsigned ) 0,"index","input DOF index" ) )
    , indexFromEnd( initData ( &indexFromEnd,false,"indexFromEnd","input DOF index starts from the end of input DOFs vector") )
    , repartition ( initData ( &repartition,"repartition","number of dest dofs per entry dof" ) )
    , globalToLocalCoords ( initData ( &globalToLocalCoords,"globalToLocalCoords","are the output DOFs initially expressed in global coordinates" ) )
    , m_rootAngularForceScaleFactor(initData(&m_rootAngularForceScaleFactor, (Real)1.0, "rootAngularForceScaleFactor", "Scale factor applied on the angular force accumulated on the rigid model"))
    , m_rootLinearForceScaleFactor(initData(&m_rootLinearForceScaleFactor, (Real)1.0, "rootLinearForceScaleFactor", "Scale factor applied on the linear force accumulated on the rigid model"))
    , m_fromModel(nullptr)
    , m_toModel(nullptr)
    , m_fromRootModel(nullptr)
{
}

template <class TIn, class TInRoot, class TOut>
int DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::addPoint(const OutCoord& /*c*/)
{
    msg_info() << "addPoint should be supressed" ;
    return 0;
}

template <class TIn, class TInRoot, class TOut>
int DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::addPoint(const OutCoord& /*c*/, int /*indexFrom*/)
{
    msg_info() << "addPoint should be supressed" ;
    return 0;
}

template <class TIn, class TInRoot, class TOut>
void DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::init()
{

    if(this->getFromModels1().empty())
    {
        msg_error() << "Error while initializing ; input Model not found" ;
        return;
    }

    if(this->getToModels().empty())
    {
        msg_error() << "Error while initializing ; output Model not found" ;
        return;
    }

    m_fromModel = this->getFromModels1()[0];
    m_toModel = this->getToModels()[0];
    m_toModel->resize(m_fromModel->getSize());

    //Root
    if(!this->getFromModels2().empty())
    {
        m_fromRootModel = this->getFromModels2()[0];
        msg_info() << "Root Model found : Name = " << m_fromRootModel->getName() ;
    }

    Inherit::init();
}

template <class TIn, class TInRoot, class TOut>
void DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::clear(int /*reserve*/)
{

}

template <class TIn, class TInRoot, class TOut>
void DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::setRepartition(unsigned int value)
{
    type::vector<unsigned int>& rep = *this->repartition.beginEdit();
    rep.clear();
    rep.push_back(value);
    this->repartition.endEdit();
}

template <class TIn, class TInRoot, class TOut>
void DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::setRepartition(sofa::type::vector<unsigned int> /*values*/)
{

}

template<class DataTypes>
const typename DataTypes::VecCoord& M_getX0(core::behavior::MechanicalState<DataTypes>* model)
{
    return model->read(core::ConstVecCoordId::restPosition())->getValue();
}

template<class DataTypes>
const typename DataTypes::VecCoord& M_getX0(core::State<DataTypes>* /*model*/)
{
    return nullptr;
}

template <class TIn, class TInRoot, class TOut>
void DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::apply( typename Out::VecCoord& out, const typename In::VecCoord& inDeformed, const typename InRoot::VecCoord* inRigid  )
{
    //Find the rigid center[s] and its displacement
    //Apply the displacement to all the located points

    //Root
    if(!m_fromRootModel && !this->getFromModels2().empty())
    {
        m_fromRootModel = this->getFromModels2()[0];
    }

    if (m_fromRootModel)
    {
        unsigned int cptOut;
        unsigned int val;
        Coord translation;
        Mat rotation;

        rotatedPoints.resize(inDeformed.size());
        out.resize(inDeformed.size());
        switch (repartition.getValue().size())
        {
        case 0 :
            if (indexFromEnd.getValue())
            {
                translation = (*inRigid)[(*inRigid).size() - 1 - index.getValue()].getCenter();
                (*inRigid)[(*inRigid).size() - 1 - index.getValue()].writeRotationMatrix(rotation);
                rootX = (*inRigid)[(*inRigid).size() - 1 - index.getValue()];
            }
            else
            {
                translation = (*inRigid)[index.getValue()].getCenter();
                (*inRigid)[index.getValue()].writeRotationMatrix(rotation);
                rootX = (*inRigid)[index.getValue()];
            }

            for(unsigned int i=0; i<inDeformed.size(); i++)
            {
                rotatedPoints[i] = rotation*inDeformed[i];
                out[i] = rotatedPoints[i];
                out[i] += translation;
            }
            break;

        case 1 ://one value specified : uniform repartition mapping on the input dofs
            val = repartition.getValue()[0];
            cptOut=0;
            for (unsigned int ifrom=0 ; ifrom<(*inRigid).size() ; ifrom++)
            {
                translation = (*inRigid)[ifrom].getCenter();
                (*inRigid)[ifrom].writeRotationMatrix(rotation);
                rootX = (*inRigid)[ifrom];

                for(unsigned int ito=0; ito<val; ito++)
                {
                    rotatedPoints[cptOut] = rotation*inDeformed[cptOut];
                    out[cptOut] = rotatedPoints[cptOut];
                    out[cptOut] += translation;
                    cptOut++;
                }
            }
            break;

        default :
            if (repartition.getValue().size() != (*inRigid).size())
            {
                msg_error()<<"Error : mapping dofs repartition is not correct";
                return;
            }
            cptOut=0;

            for (unsigned int ifrom=0 ; ifrom<(*inRigid).size() ; ifrom++)
            {
                translation = (*inRigid)[ifrom].getCenter();
                (*inRigid)[ifrom].writeRotationMatrix(rotation);
                rootX = (*inRigid)[ifrom];

                for(unsigned int ito=0; ito<repartition.getValue()[ifrom]; ito++)
                {
                    rotatedPoints[cptOut] = rotation*inDeformed[cptOut];
                    out[cptOut] = rotatedPoints[cptOut];
                    out[cptOut] += translation;
                    cptOut++;
                }
            }
            break;
        }
    }
    else  // no m_fromRootModel found => mapping is identity !
    {
        rootX = InRootCoord();
        out.resize(inDeformed.size()); rotatedPoints.resize(inDeformed.size());
        for(unsigned int i=0; i<inDeformed.size(); i++)
        {
            rotatedPoints[i] = inDeformed[i];
            out[i] = rotatedPoints[i];
        }
    }
}
template <class TIn, class TInRoot, class TOut>
void DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::apply(
    const core::MechanicalParams* /* mparams */, const type::vector<OutDataVecCoord*>& dataVecOutPos,
    const type::vector<const InDataVecCoord*>& dataVecInPos ,
    const type::vector<const InRootDataVecCoord*>& dataVecInRootPos)
{
    if(dataVecOutPos.empty() || dataVecInPos.empty())
        return;

    const InRootVecCoord* inroot = nullptr;

    //We need only one input In model and input Root model (if present)
    OutVecCoord& out = *dataVecOutPos[0]->beginEdit();
    const InVecCoord& in = dataVecInPos[0]->getValue();

    if (!dataVecInRootPos.empty())
        inroot = &dataVecInRootPos[0]->getValue();

    apply(out, in, inroot);

    dataVecOutPos[0]->endEdit();
}

template <class TIn, class TInRoot, class TOut>
void DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::applyJ( typename Out::VecDeriv&  out , const typename In::VecDeriv& inDeformed , const typename InRoot::VecDeriv* inRigid)
{
    if (m_fromRootModel)
    {
        Deriv v,omega;//Vec3d
        out.resize(inDeformed.size());
        if (indexFromEnd.getValue())
        {
            v = getVCenter((*inRigid)[(*inRigid).size() - 1 - index.getValue()]);
            omega = getVOrientation((*inRigid)[(*inRigid).size() - 1 - index.getValue()]);
        }
        else
        {
            v = getVCenter((*inRigid)[index.getValue()]);
            omega = getVOrientation((*inRigid)[index.getValue()]);
        }


        for(unsigned int i=0; i<inDeformed.size(); i++)
        {
            out[i] = cross(omega,rotatedPoints[i]);
            out[i] += rootX.getOrientation().rotate(inDeformed[i]); //velocity on the local system : (Vrigid + Vdeform)
            out[i]+= v; //center velocity
        }
    }
    else // no root model!
    {
        msg_error()<<"NO ROOT MODEL";
        out.resize(inDeformed.size());
        for(unsigned int i=0; i<inDeformed.size(); i++)
        {
            out[i] = inDeformed[i];
        }
    }
}

template <class TIn, class TInRoot, class TOut>
void DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::applyJ(
    const core::MechanicalParams* /* mparams */, const type::vector< OutDataVecDeriv*>& dataVecOutVel,
    const type::vector<const InDataVecDeriv*>& dataVecInVel,
    const type::vector<const InRootDataVecDeriv*>& dataVecInRootVel)
{
    if(dataVecOutVel.empty() || dataVecInVel.empty())
        return;

    const InRootVecDeriv* inroot = nullptr;

    //We need only one input In model and input Root model (if present)
    OutVecDeriv& out = *dataVecOutVel[0]->beginEdit();
    const InVecDeriv& in = dataVecInVel[0]->getValue();

    if (!dataVecInRootVel.empty())
        inroot = &dataVecInRootVel[0]->getValue();

    applyJ(out,in, inroot);

    dataVecOutVel[0]->endEdit();
}

template <class TIn, class TInRoot, class TOut>
void DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in, typename InRoot::VecDeriv* outRoot)
{

    if (m_fromRootModel)
    {
        Deriv v,omega;
        if (in.size() > rotatedPoints.size())
        {
            bool log = this->notMuted();

            msg_warning()<<" applyJT was called before any apply ("<<in.size() << "!="<<rotatedPoints.size()<<")";

            const InDataVecCoord* xfromData = m_toModel->read(core::ConstVecCoordId::position());
            const InVecCoord xfrom = xfromData->getValue();
            OutDataVecCoord* xtoData = m_toModel->write(core::VecCoordId::position());
            OutVecCoord &xto = *xtoData->beginEdit();
            apply(xto, xfrom, (m_fromRootModel==nullptr ? nullptr : &m_fromRootModel->read(core::ConstVecCoordId::position())->getValue()));
            this->f_printLog.setValue(log);
            xtoData->endEdit();
        }

        for(unsigned int i=0; i<in.size(); i++)
        {
            Deriv f = in[i];
            v += f;
            omega += cross(rotatedPoints[i],f);
        }

        if (indexFromEnd.getValue())
        {

            getVCenter((*outRoot)[(*outRoot).size() - 1 - index.getValue()]) += v;
            getVOrientation((*outRoot)[(*outRoot).size() - 1 - index.getValue()]) += omega;
            for(unsigned int i=0; i<in.size(); i++)
                out[i]+=rootX.getOrientation().inverseRotate(in[i]);
        }
        else
        {

            getVCenter((*outRoot)[index.getValue()]) += v;
            getVOrientation((*outRoot)[index.getValue()]) += omega;
            for(unsigned int i=0; i<in.size(); i++)
                out[i]+=rootX.getOrientation().inverseRotate(in[i]);
        }
    }
    else
    {
        out.resize(in.size());
        for(unsigned int i=0; i<in.size(); i++)
        {
            out[i] = in[i];
        }
    }
}

template <class TIn, class TInRoot, class TOut>
void DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::applyJT(
    const core::MechanicalParams* /* mparams */, const type::vector< InDataVecDeriv*>& dataVecOutForce,
    const type::vector< InRootDataVecDeriv*>& dataVecOutRootForce,
    const type::vector<const OutDataVecDeriv*>& dataVecInForce)
{
    if(dataVecOutForce.empty() || dataVecInForce.empty())
        return;

    InRootVecDeriv* outroot = nullptr;

    //We need only one input In model and input Root model (if present)
    InVecDeriv& out = *dataVecOutForce[0]->beginEdit();
    const OutVecDeriv& in = dataVecInForce[0]->getValue();

    if (!dataVecOutRootForce.empty())
        outroot = dataVecOutRootForce[0]->beginEdit();

    applyJT(out,in, outroot);

    dataVecOutForce[0]->endEdit();
    if (outroot != nullptr)
        dataVecOutRootForce[0]->endEdit();
}

template <class TIn, class TInRoot, class TOut>
void DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId inForce, core::ConstMultiVecDerivId outForce)
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(inForce);
    SOFA_UNUSED(outForce);
}

template <class TIn, class TInRoot, class TOut>
void DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::applyJT( typename In::MatrixDeriv&  out , const typename Out::MatrixDeriv&  in , typename InRoot::MatrixDeriv*  outroot)
{
    if (m_fromRootModel)
    {
        typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

        for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
        {
            typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();
            typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();

            // Creates a constraint if the input constraint is not empty.
            if (colIt != colItEnd)
            {
                Vector v, omega;

                typename In::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());
                typename InRoot::MatrixDeriv::RowIterator oRoot = outroot->writeLine(rowIt.index());

                while (colIt != colItEnd)
                {
                    const unsigned int node_index = colIt.index();

                    const Deriv f = colIt.val();
                    v += f;
                    omega += cross(rotatedPoints[node_index], f);
                    InDeriv f_deform = rootX.getOrientation().inverseRotate(f);

                    o.addCol(node_index, f_deform);

                    ++colIt;
                }

                const InRootDeriv result(m_rootLinearForceScaleFactor.getValue() * v, m_rootAngularForceScaleFactor.getValue() * omega);

                if (!indexFromEnd.getValue())
                {
                    oRoot.addCol(index.getValue(), result);
                }
                else
                {
                    const unsigned int numDofs = m_fromModel->getSize();
                    oRoot.addCol(numDofs - 1 - index.getValue(), result);
                }
            }
        }
    }
}

template <class TIn, class TInRoot, class TOut>
void DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::applyJT(
    const core::ConstraintParams* /* cparams */, const type::vector< InDataMatrixDeriv*>& dataMatOutConst ,
    const type::vector< InRootDataMatrixDeriv*>&  dataMatOutRootConst ,
    const type::vector<const OutDataMatrixDeriv*>& dataMatInConst)
{
    if(dataMatOutConst.empty() || dataMatInConst.empty())
        return;

    InRootMatrixDeriv* outroot = nullptr;

    //We need only one input In model and input Root model (if present)
    InMatrixDeriv& out = *dataMatOutConst[0]->beginEdit();
    const OutMatrixDeriv& in = dataMatInConst[0]->getValue();

    if (!dataMatOutRootConst.empty())
        outroot = dataMatOutRootConst[0]->beginEdit();

    applyJT(out,in, outroot);

    dataMatOutConst[0]->endEdit();
    if (outroot != nullptr)
        dataMatOutRootConst[0]->endEdit();
}

template <class TIn, class TInRoot, class TOut>
void DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::handleTopologyChange(core::topology::Topology* t)
{
    const core::topology::BaseMeshTopology* from = t->toBaseMeshTopology();
    if(from == nullptr ) {
        msg_error() << __FUNCTION__ << ": could not cast topology to BaseMeshTopology";
        return;
    }
    const std::list<const core::topology::TopologyChange *>::const_iterator itBegin = from->beginChange();
    const std::list<const core::topology::TopologyChange *>::const_iterator itEnd = from->endChange();

    for ( std::list<const core::topology::TopologyChange *>::const_iterator changeIt = itBegin;
            changeIt != itEnd; ++changeIt )
    {
        const core::topology::TopologyChangeType changeType = ( *changeIt )->getChangeType();
        switch ( changeType )
        {
            case core::topology::TRIANGLESADDED:       ///< To notify the end for the current sequence of topological change events
            {
                core::Multi2Mapping<TIn, TInRoot, TOut>::apply(core::mechanicalparams::defaultInstance(), core::VecCoordId::restPosition(), core::ConstVecCoordId::restPosition());
                if(this->f_applyRestPosition.getValue() )
                    core::Multi2Mapping<TIn, TInRoot, TOut>::apply(core::mechanicalparams::defaultInstance(), core::VecCoordId::position(), core::ConstVecCoordId::position());
                break;
            }
            default:
                break;

        }

    }

}

template <class TIn, class TInRoot, class TOut>
void DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::recomputeRigidMass()
{
    // (30-11-2018, Olivier Goury): Not implemented yet. Someone want to do it?
}

template <class TIn, class TInRoot, class TOut>
void DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowMappings()) return;
    std::vector< sofa::type::Vec3 > points;
    sofa::type::Vec3 point;

    const typename Out::VecCoord& x = m_toModel->read(core::ConstVecCoordId::position())->getValue();
    for (unsigned int i=0; i<x.size(); i++)
    {
        point = Out::getCPos(x[i]);
        points.push_back(point);
    }
    vparams->drawTool()->drawPoints(points, 7, sofa::type::RGBAColor::yellow());
}

} // namespace sofa::component::mapping::linear
