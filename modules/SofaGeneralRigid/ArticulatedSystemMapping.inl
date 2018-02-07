/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

#ifndef SOFA_COMPONENT_MAPPING_ARTICULATEDSYSTEMMAPPING_INL
#define SOFA_COMPONENT_MAPPING_ARTICULATEDSYSTEMMAPPING_INL

#include <SofaGeneralRigid/ArticulatedSystemMapping.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/simulation/Simulation.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/helper/gl/template.h>

#include <sofa/simulation/Node.h>

namespace sofa
{

namespace component
{

namespace mapping
{

template <class TIn, class TInRoot, class TOut>
ArticulatedSystemMapping<TIn, TInRoot, TOut>::ArticulatedSystemMapping ()
    : ahc(NULL)
    , m_fromModel(NULL), m_toModel(NULL), m_fromRootModel(NULL)
{

}

template <class TIn, class TInRoot, class TOut>
void ArticulatedSystemMapping<TIn, TInRoot, TOut>::init()
{

    if(this->getFromModels1().empty())
    {
        serr << "Error while iniatilizing ; input Model not found" << sendl;
        return;
    }

    if(this->getToModels().empty())
    {
        serr << "Error while iniatilizing ; output Model not found" << sendl;
        return;
    }

    m_fromModel = this->getFromModels1()[0];
    m_toModel = this->getToModels()[0];

    const InVecCoord& xfrom = m_fromModel->read(core::ConstVecCoordId::position())->getValue();

    ArticulationPos.clear();
    ArticulationAxis.clear();
    ArticulationPos.resize(xfrom.size());
    ArticulationAxis.resize(xfrom.size());

    //Root
    if(!this->getFromModels2().empty())
    {
        m_fromRootModel = this->getFromModels2()[0];
        sout << "Root Model found : Name = " << m_fromRootModel->getName() << sendl;
    }

    CoordinateBuf.clear();
    CoordinateBuf.resize(xfrom.size());
    for (unsigned int c=0; c<xfrom.size(); c++)
    {
        CoordinateBuf[c].x() = 0.0;
    }

    helper::WriteAccessor<Data<OutVecCoord> > xtoData = *m_toModel->write(core::VecCoordId::position());
    apply(xtoData.wref(),
            xfrom,
            m_fromRootModel == NULL ? NULL : &m_fromRootModel->read(core::ConstVecCoordId::position())->getValue());
    Inherit::init();
    /*
    OutVecDeriv& vto = m_toModel->read(core::ConstVecDerivId::velocity())->getValue();
    InVecDeriv& vfrom = m_fromModel->read(core::ConstVecDerivId::velocity())->getValue();
    applyJT(vfrom, vto);
    */

}

template <class TIn, class TInRoot, class TOut>
void ArticulatedSystemMapping<TIn, TInRoot, TOut>::bwdInit()
{
    //make sure that articulatedHierarchyContainer has been initialized before
    m_fromModel->getContext()->get(ahc, sofa::core::objectmodel::BaseContext::SearchDown);
    if (!ahc)
    {
        msg_error("ArticulatedSystemMapping::bwdInit") << "ArticulatedSystemMapping needs a ArticulatedHierarchyContainer, but it could not find it.";
        return;
    }
    articulationCenters = ahc->getArticulationCenters();

    helper::vector< sofa::component::container::ArticulationCenter* >::const_iterator ac = articulationCenters.begin();
    helper::vector< sofa::component::container::ArticulationCenter* >::const_iterator acEnd = articulationCenters.end();

    for (; ac != acEnd; ac++)
    {
        (*ac)->OrientationArticulationCenter.clear();
        (*ac)->DisplacementArticulationCenter.clear();
        (*ac)->Disp_Rotation.clear();

        // todo : warning if a (*a)->articulationIndex.getValue() exceed xfrom size !
    }
}


template <class TIn, class TInRoot, class TOut>
void ArticulatedSystemMapping<TIn, TInRoot, TOut>::reset()
{
    init();
}



template <class TIn, class TInRoot, class TOut>
void ArticulatedSystemMapping<TIn, TInRoot, TOut>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in, const typename InRoot::VecCoord* inroot  )
{
    const Data< OutVecCoord > &xtoData = *m_toModel->read(core::VecCoordId::position());
    out.resize(xtoData.getValue().size());

    // Copy the root position if a rigid root model is present
    if (m_fromRootModel && inroot)
    {
        out[0] = (*inroot)[m_fromRootModel->getSize()-1];
    }

    helper::vector< sofa::component::container::ArticulationCenter* >::const_iterator ac = articulationCenters.begin();
    helper::vector< sofa::component::container::ArticulationCenter* >::const_iterator acEnd = articulationCenters.end();

    for (; ac != acEnd; ac++)
    {
        int parent = (*ac)->parentIndex.getValue();
        int child = (*ac)->childIndex.getValue();

        // Before computing the child position, it is placed with the same orientation than its parent
        // and at the position compatible with the definition of the articulation center
        // (see initTranslateChild function for details...)
        sofa::defaulttype::Quat quat_child_buf = out[child].getOrientation();

        // The position of the articulation center can be deduced using the 6D position of the parent:
        // only useful for visualisation of the mapping => NO ! Used in applyJ and applyJT
        (*ac)->globalPosition.setValue(out[parent].getCenter() +
                out[parent].getOrientation().rotate((*ac)->posOnParent.getValue()));

        helper::vector< sofa::component::container::Articulation* > articulations = (*ac)->getArticulations();
        helper::vector< sofa::component::container::Articulation* >::const_iterator a = articulations.begin();
        helper::vector< sofa::component::container::Articulation* >::const_iterator aEnd = articulations.end();

        int process = (*ac)->articulationProcess.getValue();

        switch(process)
        {
        case 0: // 0-(default) articulation are treated one by one, the axis of the second articulation is updated by the potential rotation of the first articulation
            //			   potential problems could arise when rotation exceed 90? (known problem of euler angles)
        {
            // the position of the child is reset to its rest position (based on the postion of the articulation center)
            out[child].getOrientation() = out[parent].getOrientation();
            out[child].getCenter() = out[parent].getCenter() + (*ac)->initTranslateChild(out[parent].getOrientation());

            sofa::defaulttype::Vec<3,OutReal> APos;
            APos = (*ac)->globalPosition.getValue();
            for (; a != aEnd; a++)
            {
                sofa::defaulttype::Vec<3,Real> axis = (*a)->axis.getValue();
                axis.normalize();
                (*a)->axis.setValue(axis);

                int ind = (*a)->articulationIndex.getValue();
                InCoord value = in[ind];
                axis = out[child].getOrientation().rotate((*a)->axis.getValue());
                ArticulationAxis[ind] = axis;

                if ((*a)->rotation.getValue())
                {
                    sofa::defaulttype::Quat dq;
                    dq.axisToQuat(axis, value.x());
                    out[child].getCenter() += (*ac)->translateChild(dq, out[child].getOrientation());
                    out[child].getOrientation() += dq;

                }
                if ((*a)->translation.getValue())
                {
                    out[child].getCenter() += axis*value.x();
                    APos += axis*value.x();
                }

                ArticulationPos[ind]= APos;
            }
            break;
        }
        case 1: // the axis of the articulations are linked to the parent - rotations are treated by successive increases -
        {
            // no reset of the position of the child its position is corrected at the end to respect the articulation center.
            for (; a != aEnd; a++)
            {
                int ind = (*a)->articulationIndex.getValue();
                InCoord value = in[ind];
                InCoord prev_value = CoordinateBuf[ind];
                sofa::defaulttype::Vec<3,Real> axis = out[parent].getOrientation().rotate((*a)->axis.getValue());
                ArticulationAxis[ind]=axis;

                // the increment of rotation and translation are stored in dq and disp
                if ((*a)->rotation.getValue() )
                {
                    sofa::defaulttype::Quat r;
                    r.axisToQuat(axis, value.x() - prev_value.x());
                    // add the contribution into the quaternion that provides the actual orientation of the articulation center
                    (*ac)->OrientationArticulationCenter+=r;

                }
                if ((*a)->translation.getValue())
                {
                    (*ac)->DisplacementArticulationCenter+=axis*(value.x() - prev_value.x());
                }

            }

            //// in case 1: the rotation of the axis of the articulation follows the parent -> translation are treated "before":


            // step 1: compute the new position of the articulation center and the articulation pos
            //         rq: the articulation center folows the translations
            (*ac)->globalPosition.setValue(out[parent].getCenter() + out[parent].getOrientation().rotate((*ac)->posOnParent.getValue()) + (*ac)->DisplacementArticulationCenter);
            helper::vector< sofa::component::container::Articulation* >::const_iterator a = articulations.begin();

            for (; a != aEnd; a++)
            {
                sofa::defaulttype::Vec<3,OutReal> APos;
                APos = (*ac)->globalPosition.getValue();
                ArticulationPos[(*a)->articulationIndex.getValue()]=APos;
            }

            // step 2: compute the position of the child
            out[child].getOrientation() = out[parent].getOrientation() + (*ac)->OrientationArticulationCenter;
            out[child].getCenter() =  (*ac)->globalPosition.getValue() - out[child].getOrientation().rotate( (*ac)->posOnChild.getValue() );

            break;

        }
        case 2: // the axis of the articulations are linked to the child (previous pos) - rotations are treated by successive increases -
        {
            // no reset of the position of the child its position is corrected at the end to respect the articulation center.
            //Quat dq(0,0,0,1);
            sofa::defaulttype::Vec<3,Real> disp(0,0,0);

            for (; a != aEnd; a++)
            {
                int ind = (*a)->articulationIndex.getValue();
                InCoord value = in[ind];
                InCoord prev_value = CoordinateBuf[ind];
                sofa::defaulttype::Vec<3,Real> axis = quat_child_buf.rotate((*a)->axis.getValue());
                ArticulationAxis[ind]=axis;


                // the increment of rotation and translation are stored in dq and disp
                if ((*a)->rotation.getValue() )
                {
                    sofa::defaulttype::Quat r;
                    r.axisToQuat(axis, value.x() - prev_value.x());
                    // add the contribution into the quaternion that provides the actual orientation of the articulation center
                    (*ac)->OrientationArticulationCenter+=r;
                }
                if ((*a)->translation.getValue())
                {
                    disp += axis*(value.x()) ;

                }

                //// in case 2: the rotation of the axis of the articulation follows the child -> translation are treated "after"
                //// ArticulationPos do not move
                sofa::defaulttype::Vec<3,OutReal> APos;
                APos = (*ac)->globalPosition.getValue();
                ArticulationPos[(*a)->articulationIndex.getValue()]=APos;

            }
            (*ac)->DisplacementArticulationCenter=disp;

            out[child].getOrientation() = out[parent].getOrientation() + (*ac)->OrientationArticulationCenter;
            out[child].getCenter() =  (*ac)->globalPosition.getValue() - out[child].getOrientation().rotate((*ac)->posOnChild.getValue());
            out[child].getCenter() += (*ac)->DisplacementArticulationCenter;

            break;

        }
        }
    }

    //////////////////// buf the actual position of the articulations ////////////////////

    CoordinateBuf.clear();
    CoordinateBuf.resize(in.size());
    for (unsigned int c=0; c<in.size(); c++)
    {
        CoordinateBuf[c].x() = in[c].x();
    }
}

template <class TIn, class TInRoot, class TOut>
void ArticulatedSystemMapping<TIn, TInRoot, TOut>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in, const typename InRoot::VecDeriv* inroot )
{
    Data<OutVecCoord>* xtoData = m_toModel->write(core::VecCoordId::position());

    const OutVecCoord& xto = xtoData->getValue();

    out.clear();
    out.resize(xto.size());

    // Copy the root position if a rigid root model is present
    if (m_fromRootModel && inroot)
        out[0] = (*inroot)[m_fromRootModel->getSize()-1];
    else
        out[0] = OutDeriv();

    helper::vector< sofa::component::container::ArticulationCenter* >::const_iterator ac = articulationCenters.begin();
    helper::vector< sofa::component::container::ArticulationCenter* >::const_iterator acEnd = articulationCenters.end();

    int i = 0;

    for (; ac != acEnd; ac++)
    {
        int parent = (*ac)->parentIndex.getValue();
        int child = (*ac)->childIndex.getValue();

        getVOrientation(out[child]) += getVOrientation(out[parent]);
        sofa::defaulttype::Vec<3,OutReal> P = xto[parent].getCenter();
        sofa::defaulttype::Vec<3,OutReal> C = xto[child].getCenter();
        getVCenter(out[child]) = getVCenter(out[parent]) + cross(P-C, getVOrientation(out[parent]));
        //sout<<"P:"<< P  <<"- C: "<< C;

        helper::vector< sofa::component::container::Articulation* > articulations = (*ac)->getArticulations();
        helper::vector< sofa::component::container::Articulation* >::const_iterator a = articulations.begin();
        helper::vector< sofa::component::container::Articulation* >::const_iterator aEnd = articulations.end();

        for (; a != aEnd; a++)
        {
            int ind = (*a)->articulationIndex.getValue();
            InCoord value = in[ind];
            sofa::defaulttype::Vec<3,OutReal> axis = ArticulationAxis[ind];
            sofa::defaulttype::Vec<3,OutReal> A = ArticulationPos[ind];


            if ((*a)->rotation.getValue())
            {
                getVCenter(out[child]) += cross(A-C, axis*value.x());
                getVOrientation(out[child]) += axis*value.x();
            }
            if ((*a)->translation.getValue())
            {
                getVCenter(out[child]) += axis*value.x();
            }
            i++;

        }
    }
}



template <class TIn, class TInRoot, class TOut>
void ArticulatedSystemMapping<TIn, TInRoot, TOut>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in, typename InRoot::VecDeriv* outroot )
{
    const OutVecCoord& xto = m_toModel->read(core::VecCoordId::position())->getValue();

    OutVecDeriv fObjects6DBuf = in;
    InVecDeriv OutBuf = out;

    helper::vector< sofa::component::container::ArticulationCenter* >::const_iterator ac = articulationCenters.end();
    helper::vector< sofa::component::container::ArticulationCenter* >::const_iterator acBegin = articulationCenters.begin();

    int i=ArticulationAxis.size();
    while (ac != acBegin)
    {
        ac--;
        int parent = (*ac)->parentIndex.getValue();
        int child = (*ac)->childIndex.getValue();

        getVCenter(fObjects6DBuf[parent]) += getVCenter(fObjects6DBuf[child]);
        sofa::defaulttype::Vec<3,OutReal> P = xto[parent].getCenter();
        sofa::defaulttype::Vec<3,OutReal> C = xto[child].getCenter();
        getVOrientation(fObjects6DBuf[parent]) += getVOrientation(fObjects6DBuf[child]) + cross(C-P,  getVCenter(fObjects6DBuf[child]));

        helper::vector< sofa::component::container::Articulation* > articulations = (*ac)->getArticulations();

        helper::vector< sofa::component::container::Articulation* >::const_iterator a = articulations.end();
        helper::vector< sofa::component::container::Articulation* >::const_iterator aBegin = articulations.begin();

        while (a != aBegin)
        {
            a--;
            i--;
            int ind = (*a)->articulationIndex.getValue();
            sofa::defaulttype::Vec<3,OutReal> axis = ArticulationAxis[ind];
            sofa::defaulttype::Vec<3,Real> A = ArticulationPos[ind] ;
            OutDeriv T;
            getVCenter(T) = getVCenter(fObjects6DBuf[child]);
            getVOrientation(T) = getVOrientation(fObjects6DBuf[child]) + cross(C-A, getVCenter(fObjects6DBuf[child]));

            if ((*a)->rotation.getValue())
            {
                out[ind].x() += (InReal)dot(axis, getVOrientation(T));
            }
            if ((*a)->translation.getValue())
            {
                out[ind].x() += (InReal)dot(axis, getVCenter(T));
            }
        }
    }

    if (outroot && m_fromRootModel)
    {
        (*outroot)[m_fromRootModel->getSize()-1] += fObjects6DBuf[0];
    }
}


template <class TIn, class TInRoot, class TOut>
void ArticulatedSystemMapping<TIn, TInRoot, TOut>::applyJT( InMatrixDeriv& out, const OutMatrixDeriv& in, InRootMatrixDeriv* outRoot )
{
    const OutVecCoord& xto = m_toModel->read(core::ConstVecCoordId::position())->getValue();

    typename OutMatrixDeriv::RowConstIterator rowItEnd = in.end();

    for (typename OutMatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        typename OutMatrixDeriv::ColConstIterator colIt = rowIt.begin();
        typename OutMatrixDeriv::ColConstIterator colItEnd = rowIt.end();

        // Creates a constraints if the input constraint is not empty.
        if (colIt != colItEnd)
        {
            typename InMatrixDeriv::RowIterator o = out.writeLine(rowIt.index());

            //Hack to get a RowIterator, withtout default constructor
            InRootMatrixDeriv temp;
            typename InRootMatrixDeriv::RowIterator rootRowIt = temp.end();
            typename InRootMatrixDeriv::RowIterator rootRowItEnd = temp.end();

            if(m_fromRootModel && outRoot)
            {
                rootRowIt = outRoot->end();
                rootRowItEnd = outRoot->end();
            }

            while (colIt != colItEnd)
            {
                int childIndex = colIt.index();
                const OutDeriv valueConst = colIt.val();

                sofa::defaulttype::Vec<3,OutReal> C = xto[childIndex].getCenter();
                helper::vector< sofa::component::container::ArticulationCenter* > ACList = ahc->getAcendantList(childIndex);

                helper::vector< sofa::component::container::ArticulationCenter* >::const_iterator ac = ACList.begin();
                helper::vector< sofa::component::container::ArticulationCenter* >::const_iterator acEnd = ACList.end();

                for (; ac != acEnd; ac++)
                {
                    helper::vector< sofa::component::container::Articulation* > articulations = (*ac)->getArticulations();

                    helper::vector< sofa::component::container::Articulation* >::const_iterator a = articulations.begin();
                    helper::vector< sofa::component::container::Articulation* >::const_iterator aEnd = articulations.end();

                    for (; a != aEnd; a++)
                    {
                        int ind = (*a)->articulationIndex.getValue();
                        InDeriv data;

                        sofa::defaulttype::Vec< 3, OutReal > axis = ArticulationAxis[ind]; // xto[parent].getOrientation().rotate((*a)->axis.getValue());
                        sofa::defaulttype::Vec< 3, Real > A = ArticulationPos[ind] ; // Vec<3,OutReal> posAc = (*ac)->globalPosition.getValue();

                        OutDeriv T;
                        getVCenter(T) = getVCenter(valueConst);
                        getVOrientation(T) = getVOrientation(valueConst) + cross(C - A, getVCenter(valueConst));

                        if ((*a)->rotation.getValue())
                        {
                            data = (InReal)dot(axis, getVOrientation(T));
                        }

                        if ((*a)->translation.getValue())
                        {
                            data = (InReal)dot(axis, getVCenter(T));
                        }

                        o.addCol(ind, data);
                    }
                }

                if(m_fromRootModel && outRoot)
                {
                    unsigned int indexT = m_fromRootModel->getSize() - 1; // On applique sur le dernier noeud
                    sofa::defaulttype::Vec<3,OutReal> posRoot = xto[indexT].getCenter();

                    OutDeriv T;
                    getVCenter(T) = getVCenter(valueConst);
                    getVOrientation(T) = getVOrientation(valueConst) + cross(C - posRoot, getVCenter(valueConst));

                    if (rootRowIt == rootRowItEnd)
                        rootRowIt = (*outRoot).writeLine(rowIt.index());

                    rootRowIt.addCol(indexT, T);
                }

                ++colIt;
            }
        }
    }
}

template <class TIn, class TInRoot, class TOut>
void ArticulatedSystemMapping<TIn, TInRoot, TOut>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowMappings()) return;
    std::vector< sofa::defaulttype::Vector3 > points;
    std::vector< sofa::defaulttype::Vector3 > pointsLine;

    helper::vector< sofa::component::container::ArticulationCenter* >::const_iterator ac = articulationCenters.begin();
    helper::vector< sofa::component::container::ArticulationCenter* >::const_iterator acEnd = articulationCenters.end();
    unsigned int i=0;
    for (; ac != acEnd; ac++)
    {
        helper::vector< sofa::component::container::Articulation* > articulations = (*ac)->getArticulations();
        helper::vector< sofa::component::container::Articulation* >::const_iterator a = articulations.begin();
        helper::vector< sofa::component::container::Articulation* >::const_iterator aEnd = articulations.end();
        for (; a != aEnd; a++)
        {

            // Articulation Pos and Axis are based on the configuration of the parent
            int ind= (*a)->articulationIndex.getValue();
            points.push_back(ArticulationPos[ind]);

            pointsLine.push_back(ArticulationPos[ind]);
            sofa::defaulttype::Vec<3,OutReal> Pos_axis = ArticulationPos[ind] + ArticulationAxis[ind];
            pointsLine.push_back(Pos_axis);

            i++;
        }
    }

    vparams->drawTool()->drawPoints(points, 10, sofa::defaulttype::Vec<4,float>(1,0.5,0.5,1));
    vparams->drawTool()->drawLines(pointsLine, 1, sofa::defaulttype::Vec<4,float>(0,0,1,1));
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MAPPING_ARTICULATEDSYSTEMMAPPING_INL
