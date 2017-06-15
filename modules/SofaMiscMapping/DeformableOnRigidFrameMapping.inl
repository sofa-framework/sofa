/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MAPPING_DEFORMABLEONRIGIDFRAME_INL
#define SOFA_COMPONENT_MAPPING_DEFORMABLEONRIGIDFRAME_INL

#include <SofaMiscMapping/DeformableOnRigidFrameMapping.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/MechanicalState.h>

#include <sofa/core/VecId.h>

#include <sofa/simulation/Simulation.h>

#include <string.h>
#include <iostream>


/*!
*   This mapping is derived from the RigidMapping. The difference is :
*   In the RigidMapping, the rigid is considered like a perfect rigid (non-deformable)
*   In this one, the rigid allow a low deformation of the rigid.
*
*   Principale difference with the RigidMapping is in the fonctions apply, applyJ and applyJT
*/

namespace sofa
{

namespace component
{

namespace mapping
{

template <class TIn, class TInRoot, class TOut>
DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::DeformableOnRigidFrameMapping()
    : index ( initData ( &index, ( unsigned ) 0,"index","input DOF index" ) )
    , indexFromEnd( initData ( &indexFromEnd,false,"indexFromEnd","input DOF index starts from the end of input DOFs vector") )
    , repartition ( initData ( &repartition,"repartition","number of dest dofs per entry dof" ) )
    , globalToLocalCoords ( initData ( &globalToLocalCoords,"globalToLocalCoords","are the output DOFs initially expressed in global coordinates" ) )
    , m_rootAngularForceScaleFactor(initData(&m_rootAngularForceScaleFactor, (Real)1.0, "rootAngularForceScaleFactor", "Scale factor applied on the angular force accumulated on the rigid model"))
    , m_rootLinearForceScaleFactor(initData(&m_rootLinearForceScaleFactor, (Real)1.0, "rootLinearForceScaleFactor", "Scale factor applied on the linear force accumulated on the rigid model"))
    , m_fromModel(NULL)
    , m_toModel(NULL)
    , m_fromRootModel(NULL)
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
        serr << "Error while initializing ; input Model not found" << sendl;
        return;
    }

    if(this->getToModels().empty())
    {
        serr << "Error while initializing ; output Model not found" << sendl;
        return;
    }

    m_fromModel = this->getFromModels1()[0];
    m_toModel = this->getToModels()[0];
    m_toModel->resize(m_fromModel->getSize());

    //Root
    if(!this->getFromModels2().empty())
    {
        m_fromRootModel = this->getFromModels2()[0];
        sout << "Root Model found : Name = " << m_fromRootModel->getName() << sendl;
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
    helper::vector<unsigned int>& rep = *this->repartition.beginEdit();
    rep.clear();
    rep.push_back(value);
    this->repartition.endEdit();
}

template <class TIn, class TInRoot, class TOut>
void DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::setRepartition(sofa::helper::vector<unsigned int> /*values*/)
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
    return NULL;
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
                serr<<"Error : mapping dofs repartition is not correct"<<sendl;
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
void DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::applyJ( typename Out::VecDeriv&  out , const typename In::VecDeriv& inDeformed , const typename InRoot::VecDeriv* inRigid)
{
    if (m_fromRootModel)
    {
        Deriv v,omega;//Vec3d
        out.resize(inDeformed.size());
        //unsigned int cptOut;
        //unsigned int val;


        //switch (repartition.getValue().size())
        //  {
        //  case 0:
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
        //         break;
        /* case 1://one value specified : uniform repartition mapping on the input dofs
        val = repartition.getValue()[0];
        cptOut=0;
        for (unsigned int ifrom=0 ; ifrom<(*inRigid).size() ; ifrom++){
        v = (*inRigid)[ifrom].getVCenter();
        omega = (*inRigid)[ifrom].getVOrientation();

        for(unsigned int ito=0; ito<val; ito++){
        out[cptOut] = -cross(rotatedPoints[ito],omega)+ rootX.getOrientation().rotate(inDeformed[ito]);
        out[cptOut] += v;
        cptOut++;
        }
        }
        break;
        default:
        if (repartition.getValue().size() != (*inRigid).size()){
        serr<<"Error : mapping dofs repartition is not correct"<<sendl;
        return;
        }
        cptOut=0;
        for (unsigned int ifrom=0 ; ifrom<(*inRigid).size() ; ifrom++){
        v = (*inRigid)[ifrom].getVCenter();
        omega = (*inRigid)[ifrom].getVOrientation();

        for(unsigned int ito=0; ito<repartition.getValue()[ifrom]; ito++){
        out[cptOut] = -cross(rotatedPoints[cptOut],omega) + rootX.getOrientation().rotate(inDeformed[cptOut]);
        out[cptOut] += v;
        cptOut++;
        }
        }
        break;
        }
        */
    }



    else // no root model!
    {
        serr<<"NO ROOT MODEL"<<sendl;
        out.resize(inDeformed.size());
        for(unsigned int i=0; i<inDeformed.size(); i++)
        {
            out[i] = inDeformed[i];
        }
    }
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
            apply(xto, xfrom, (m_fromRootModel==NULL ? NULL : &m_fromRootModel->read(core::ConstVecCoordId::position())->getValue()));
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
void DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::applyJT( typename In::MatrixDeriv&  out , const typename Out::MatrixDeriv&  in , typename InRoot::MatrixDeriv*  outroot)
{
    if (m_fromRootModel)
    {
        typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

        for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
        {
            typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();
            typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();

            // Creates a constraints if the input constraint is not empty.
            if (colIt != colItEnd)
            {
                Vector v, omega;

                typename In::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());
                typename InRoot::MatrixDeriv::RowIterator oRoot = outroot->writeLine(rowIt.index());

                while (colIt != colItEnd)
                {
                    const unsigned int node_index = colIt.index();
                    // out = Jt in
                    // Jt = [ I     ]
                    //      [ -OM^t ]
                    // -OM^t = OM^

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
                    // Commented by PJ. Bug??
                    // todo(dmarchal 2017-05-03) so what ?
                    // oRoot.addCol(out.size() - 1 - index.getValue(), result);

                    const unsigned int numDofs = m_fromModel->getSize();
                    oRoot.addCol(numDofs - 1 - index.getValue(), result);
                }
            }
        }
    }
}



template <class TIn, class TInRoot, class TOut>
void DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::handleTopologyChange(core::topology::Topology* t)
{
    core::topology::BaseMeshTopology* from = t->toBaseMeshTopology();
    if(from == NULL ) {
        this->serr << __FUNCTION__ << ": could not cast topology to BaseMeshTopology" << this->sendl;
        return;
    }
    std::list<const core::topology::TopologyChange *>::const_iterator itBegin = from->beginChange();
    std::list<const core::topology::TopologyChange *>::const_iterator itEnd = from->endChange();

    for ( std::list<const core::topology::TopologyChange *>::const_iterator changeIt = itBegin;
            changeIt != itEnd; ++changeIt )
    {
        const core::topology::TopologyChangeType changeType = ( *changeIt )->getChangeType();
        switch ( changeType )
        {
            case core::topology::TRIANGLESADDED:       ///< To notify the end for the current sequence of topological change events
            {
                core::Multi2Mapping<TIn, TInRoot, TOut>::apply(core::MechanicalParams::defaultInstance(), core::VecCoordId::restPosition(), core::ConstVecCoordId::restPosition());
                if(this->f_applyRestPosition.getValue() )
                    core::Multi2Mapping<TIn, TInRoot, TOut>::apply(core::MechanicalParams::defaultInstance(), core::VecCoordId::position(), core::ConstVecCoordId::position());
                break;
            }
            default:
                break;

        }

    }

}


/// Template specialization for 2D rigids
// template<typename real1, typename real2>
// void DeformableOnRigidFrameMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::StdRigidTypes<2, real1> >, core::behavior::MechanicalState< defaulttype::StdVectorTypes<defaulttype::Vec<2, real2>, defaulttype::Vec<2, real2>, real2 > > > >::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
// {
//     Deriv v;
//     Real omega;
//     v = in[index.getValue()].getVCenter();
//     omega = (Real)in[index.getValue()].getVOrientation();
//     out.resize(points.size());
//     for(unsigned int i=0;i<points.size();i++)
//     {
//         out[i] =  v + Deriv(-rotatedPoints[i][1],rotatedPoints[i][0])*omega;
//     }
// }
//#ifndef SOFA_FLOAT
//template<>
//    void DeformableOnRigidFrameMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );
//template<>
//    void DeformableOnRigidFrameMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );
//template<>
//    void DeformableOnRigidFrameMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::MatrixDeriv& out, const Out::MatrixDeriv& in );
//#endif
//#ifndef SOFA_DOUBLE
//template<>
//    void DeformableOnRigidFrameMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );
//template<>
//    void DeformableOnRigidFrameMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );
//template<>
//    void DeformableOnRigidFrameMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::MatrixDeriv& out, const Out::MatrixDeriv& in );
//#endif
//
//#ifndef SOFA_FLOAT
//#ifndef SOFA_DOUBLE
//template<>
//void DeformableOnRigidFrameMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );
//template<>
//void DeformableOnRigidFrameMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );
//template<>
//    void DeformableOnRigidFrameMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );
//template<>
//    void DeformableOnRigidFrameMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );
//template<>
//    void DeformableOnRigidFrameMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::MatrixDeriv& out, const Out::MatrixDeriv& in );
//template<>
//    void DeformableOnRigidFrameMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::MatrixDeriv& out, const Out::MatrixDeriv& in );
//#endif
//#endif
/// Template specialization for 2D rigids
// template<typename real1, typename real2>
// void DeformableOnRigidFrameMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::StdRigidTypes<2, real1> >, core::behavior::MechanicalState< defaulttype::StdVectorTypes<defaulttype::Vec<2, real2>, defaulttype::Vec<2, real2>, real2 > > > >::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
// {
//     Deriv v;
//     Real omega;
//     for(unsigned int i=0;i<points.size();i++)
//     {
//         Deriv f = in[i];
//         v += f;
//         omega += cross(rotatedPoints[i],f);
//     }
//     out[index.getValue()].getVCenter() += v;
//     out[index.getValue()].getVOrientation() += (typename In::Real)omega;
// }


//
//template <class TIn, class TInRoot, class TOut>
//void DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::propagateX()
//{
//	if (m_fromModel!=NULL && m_toModel->read(sofa::core::ConstVecCoordId::position())->getValue()!=NULL && m_fromModel->read(sofa::core::ConstVecCoordId::position())->getValue()!=NULL)
//		apply(*m_toModel->read(sofa::core::ConstVecCoordId::position())->getValue(),m_fromModel->read(sofa::core::ConstVecCoordId::position())->getValue(), (m_fromRootModel==NULL ? NULL : m_fromRootModel->read(core::ConstVecCoordId::position())->getValue()));
//
//
//	if( notMuted())	{
//		serr<<"DeformableOnRigidFrameMapping::propageX processed :"<<sendl;
//		if (m_fromRootModel!=NULL)
//			serr<<"input root: "<<*m_fromRootModel->read(sofa::core::ConstVecCoordId::position())->getValue();
//		serr<<"  - input: "<<*m_fromModel->read(sofa::core::ConstVecCoordId::position())->getValue()<<"  output : "<<*m_toModel->read(sofa::core::ConstVecCoordId::position())->getValue()<<sendl;
//	}
//
//
//}
//
//template <class TIn, class TInRoot, class TOut>
//void DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::propagateXfree()
//{
//	if (m_fromModel!=NULL && m_toModel->read(sofa::core::ConstVecCoordId::freePosition())->getValue()!=NULL && m_fromModel->read(sofa::core::ConstVecCoordId::freePosition())->getValue()!=NULL)
//		apply(*m_toModel->read(sofa::core::ConstVecCoordId::freePosition())->getValue(), *m_fromModel->read(sofa::core::ConstVecCoordId::freePosition())->getValue(), (m_fromRootModel==NULL ? NULL : m_fromRootModel->read(sofa::core::ConstVecCoordId::freePosition())->getValue()));
//
//	if( notMuted()){
//		serr<<"DeformableOnRigidFrameMapping::propageXfree processed"<<sendl;
//		if (m_fromRootModel!=NULL)
//			serr<<"input root: "<<*m_fromRootModel->read(sofa::core::ConstVecCoordId::freePosition())->getValue();
//		serr<<"  - input: "<<*m_fromModel->read(sofa::core::ConstVecCoordId::freePosition())->getValue()<<"  output : "<<*m_toModel->read(sofa::core::ConstVecCoordId::freePosition())->getValue()<<sendl;
//	}
//
//}
//
//
//template <class TIn, class TInRoot, class TOut>
//void DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::propagateV()
//{
//	if (m_fromModel!=NULL && m_toModel->getV()!=NULL && m_fromModel->getV()!=NULL)
//		applyJ(m_toModel->read(core::ConstVecDerivId::velocity())->getValue(), m_fromModel->read(core::ConstVecCoordId::velocity())->getValue(), (m_fromRootModel==NULL ? NULL : m_fromRootModel->getV()));
//
//	if( notMuted()){
//		serr<<"DeformableOnRigidFrameMapping::propagateV processed"<<sendl;
//		if (m_fromRootModel!=NULL)
//			serr<<"V input root: "<<m_fromRootModel->read(core::ConstVecDerivId::velocity())->getValue();
//		serr<<"  - V input: "<<m_fromModel->read(core::ConstVecDerivId::velocity())->getValue()<<"   V output : "<<m_toModel->read(core::ConstVecCoordId::velocity())->getValue()<<sendl;
//	}
//
//}
//
//
//
//template <class TIn, class TInRoot, class TOut>
//void DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::propagateDx()
//{
//	if (m_fromModel!=NULL && m_toModel->getDx()!=NULL && m_fromModel->getDx()!=NULL)
//		applyJ(*m_toModel->getDx(), *m_fromModel->getDx(), (m_fromRootModel==NULL ? NULL : m_fromRootModel->getDx()));
//
//
//	if( notMuted()){
//		serr<<"DeformableOnRigidFrameMapping::propagateDx processed"<<sendl;
//		if (m_fromRootModel!=NULL)
//			serr<<"input root: "<<*m_fromRootModel->getDx();
//		serr<<"  - input: "<<*m_fromModel->getDx()<<"  output : "<<*m_toModel->getDx()<<sendl;
//	}
//
//}
//
//
//
//template <class TIn, class TInRoot, class TOut>
//void DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::accumulateForce()
//{
//	if (m_fromModel!=NULL && m_toModel->getF()!=NULL && m_fromModel->getF()!=NULL)
//		applyJT(*m_fromModel->getF(), *m_toModel->getF(), (m_fromRootModel==NULL ? NULL : m_fromRootModel->getF()));
//
//
//	if( notMuted()){
//		serr<<"DeformableOnRigidFrameMapping::accumulateForce processed"<<sendl;
//		serr<<" input f : "<<*m_toModel->getF();
//		if (m_fromRootModel!=NULL)
//			serr<<"- output root: "<<*m_fromRootModel->getF();
//		serr<<"  - output F: "<<*m_fromModel->getF()<<sendl;
//	}
//
//}
//
//
//
//template <class TIn, class TInRoot, class TOut>
//void DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::accumulateDf()
//{
//	//if (m_fromModel!=NULL && m_toModel->getF()!=NULL && m_fromModel->getF()!=NULL)
//	applyJT(*m_fromModel->getF(), *m_toModel->getF(), (m_fromRootModel==NULL ? NULL : m_fromRootModel->getF()));
//
//
//	if( notMuted()){
//		serr<<"DeformableOnRigidFrameMapping::accumulateDf processed"<<sendl;
//		serr<<" input df : "<<*m_toModel->getF();
//		if (m_fromRootModel!=NULL)
//			serr<<"- output root: "<<*m_fromRootModel->getF();
//		serr<<"  - output: "<<*m_fromModel->getF()<<sendl;
//	}
//
//}
//
//
//
//template <class TIn, class TInRoot, class TOut>
//void DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::accumulateConstraint()
//{
//	if (m_fromModel!=NULL && m_toModel->getC()!=NULL && m_fromModel->getC()!=NULL)
//	{
//		//propagateX();
//		applyJT(*m_fromModel->getC(), *m_toModel->getC(), (m_fromRootModel==NULL ? NULL : m_fromRootModel->getC()));
//
//		//// Accumulate contacts indices through the MechanicalMapping
//		//std::vector<unsigned int>::iterator it = m_toModel->getConstraintId().begin();
//		//std::vector<unsigned int>::iterator itEnd = m_toModel->getConstraintId().end();
//
//		//while (it != itEnd)
//		//{
//		//	m_fromModel->setConstraintId(*it);
//		//	// in case of a "multi-mapping" (the articulation system is placed on a  simulated object)
//		//	// the constraints are transmitted to the rootModle (the <rigidtype> object which is the root of the articulated system)
//		//	if (m_fromRootModel!=NULL)
//		//		m_fromRootModel->setConstraintId(*it);
//		//	it++;
//		//}
//	}
//}

/*
template <class TIn, class TInRoot, class TOut>
void DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::recomputeRigidMass()
{

if (m_fromModel==NULL || m_fromRootModel==NULL)
return;


masses = m_fromModel->getContext()->getMass();
if(!masses)
return;

totalMass = 0.0;
//compute the total mass of the object
for (unsigned int i=0 ; i<m_fromModel->getSize() ; i++)
totalMass += masses->getElementMass(i);







sofa::core::objectmodel::
m_fromModel->getContext()->get(m_fromRootModel, core::objectmodel::BaseContext::SearchUp);


}
*/



template <class TIn, class TInRoot, class TOut>
void DeformableOnRigidFrameMapping<TIn, TInRoot, TOut>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowMappings()) return;
    std::vector< sofa::defaulttype::Vector3 > points;
    sofa::defaulttype::Vector3 point;

    const typename Out::VecCoord& x = m_toModel->read(core::ConstVecCoordId::position())->getValue();
    for (unsigned int i=0; i<x.size(); i++)
    {
        point = Out::getCPos(x[i]);
        points.push_back(point);
    }
    vparams->drawTool()->drawPoints(points, 7, sofa::defaulttype::Vec<4,float>(1,1,0,1));
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MAPPING_DEFORMABLEONRIGIDFRAME_INL
