/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/

#ifndef SOFA_COMPONENT_MAPPING_ARTICULATEDSYSTEMMAPPING_INL
#define SOFA_COMPONENT_MAPPING_ARTICULATEDSYSTEMMAPPING_INL

#include <sofa/component/mapping/ArticulatedSystemMapping.h>
#include <sofa/core/objectmodel/BaseContext.h>

namespace sofa
{

namespace component
{

namespace mapping
{

template <class BasicMapping>
void ArticulatedSystemMapping<BasicMapping>::init()
{
    GNode* context = dynamic_cast<GNode*>(this->fromModel->getContext());
    context->getNodeObject(ahc);
    articulationCenters = ahc->getArticulationCenters();

    OutVecCoord& xto = *this->toModel->getX();
    InVecCoord& xfrom = *this->fromModel->getX();

    context->parent->getNodeObject(rootModel);

    apply(xto, xfrom);
}

template <class BasicMapping>
void ArticulatedSystemMapping<BasicMapping>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    // Copy the root position if a rigid root model is present
    if (rootModel)
        out[0] = (*rootModel->getX())[0];

    vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator ac = articulationCenters.begin();
    vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator acEnd = articulationCenters.end();

    for (; ac != acEnd; ac++)
    {
        int parent = (*ac)->parentIndex.getValue();
        int child = (*ac)->childIndex.getValue();

        // Before computing the child position, it is placed with the same orientation than its parent
        // and at the position compatible with the definition of the articulation center
        // (see initTranslateChild function for details...)

        out[child].getOrientation() = out[parent].getOrientation();
        out[child].getCenter() = out[parent].getCenter() + (*ac)->initTranslateChild(out[parent].getOrientation());

        // The position of the articulation center can be deduced using the 6D position of the parent:
        // only useful for visualisation of the mapping
        (*ac)->globalPosition.setValue(out[parent].getCenter() +
                out[parent].getOrientation().rotate((*ac)->posOnParent.getValue()));

        vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*> articulations = (*ac)->getArticulations();

        vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*>::const_iterator a = articulations.begin();
        vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*>::const_iterator aEnd = articulations.end();

        for (; a != aEnd; a++)
        {
            InCoord value = in[(*a)->articulationIndex.getValue()];
            Vector3 axis = out[child].getOrientation().rotate((*a)->axis.getValue());

            if ((*a)->rotation.getValue())
            {
                Quat dq;
                dq.axisToQuat(axis, value.x());
                out[child].getCenter() += (*ac)->translateChild(dq, out[child].getOrientation());
                out[child].getOrientation() += dq;
            }
            if ((*a)->translation.getValue())
            {
                out[child].getCenter() += axis*value.x();
            }
        }
    }
}

template <class BasicMapping>
void ArticulatedSystemMapping<BasicMapping>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    OutVecCoord& xto = *this->toModel->getX();

    out[0] = OutDeriv();

    out.clear();
    out.resize(xto.size());

    vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator ac = articulationCenters.begin();
    vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator acEnd = articulationCenters.end();

    for (; ac != acEnd; ac++)
    {
        int parent = (*ac)->parentIndex.getValue();
        int child = (*ac)->childIndex.getValue();

        out[child].getVOrientation() += out[parent].getVOrientation();
        Vector3 P = xto[parent].getCenter(); Vector3 C = xto[child].getCenter();
        out[child].getVCenter() = out[parent].getVCenter() + cross(P-C, out[parent].getVOrientation());

        vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*> articulations = (*ac)->getArticulations();

        vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*>::const_iterator a = articulations.begin();
        vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*>::const_iterator aEnd = articulations.end();

        for (; a != aEnd; a++)
        {
            InCoord value = in[(*a)->articulationIndex.getValue()];
            Vector3 axis = xto[parent].getOrientation().rotate((*a)->axis.getValue());

            Vector3 A = (*ac)->globalPosition.getValue();

            if ((*a)->rotation.getValue())
            {
                out[child].getVCenter() += cross(A-C, axis*value.x());
                out[child].getVOrientation() += axis*value.x();
            }
            if ((*a)->translation.getValue())
            {
                out[child].getVCenter() += axis*value.x();
            }

        }
    }
}

template <class BasicMapping>
void ArticulatedSystemMapping<BasicMapping>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    OutVecCoord& xto = *this->toModel->getX();

    OutVecDeriv fObjects6DBuf = in;

    vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator ac = articulationCenters.end();
    vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator acBegin = articulationCenters.begin();

    while (ac != acBegin)
    {
        ac--;
        int parent = (*ac)->parentIndex.getValue();
        int child = (*ac)->childIndex.getValue();

        fObjects6DBuf[parent].getVCenter() += fObjects6DBuf[child].getVCenter();
        Vector3 P = xto[parent].getCenter(); Vector3 C = xto[child].getCenter();
        fObjects6DBuf[parent].getVOrientation() += fObjects6DBuf[child].getVOrientation() + cross(C-P,  fObjects6DBuf[child].getVCenter());

        Vector3 A = (*ac)->globalPosition.getValue();
        OutDeriv T;
        T.getVCenter() = fObjects6DBuf[child].getVCenter();
        T.getVOrientation() = fObjects6DBuf[child].getVOrientation() + cross(C-A, fObjects6DBuf[child].getVCenter());

        vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*> articulations = (*ac)->getArticulations();

        vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*>::const_iterator a = articulations.end();
        vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*>::const_iterator aBegin = articulations.begin();

        while (a != aBegin)
        {
            a--;
            Vector3 axis = xto[parent].getOrientation().rotate((*a)->axis.getValue());

            if ((*a)->rotation.getValue())
            {
                out[(*a)->articulationIndex.getValue()].x() += dot(axis, T.getVOrientation());
            }
            if ((*a)->translation.getValue())
            {
                out[(*a)->articulationIndex.getValue()].x() += dot(axis, T.getVCenter());
            }
        }
    }
}

template <class BasicMapping>
void ArticulatedSystemMapping<BasicMapping>::applyJT( typename In::VecConst& out, const typename Out::VecConst& in )
{
    OutVecCoord& xto = *this->toModel->getX();

    out.resize(in.size());

    for(unsigned int i=0; i<in.size(); i++)
    {
        for (unsigned int j=0; j<in[i].size(); j++)
        {
            const OutSparseDeriv cIn = in[i][j];
            int childIndex = cIn.index;
            const OutDeriv valueConst = (OutDeriv) cIn.data;
            Vector3 posConst = xto[childIndex].getCenter();
            vector<ArticulatedHierarchyContainer::ArticulationCenter*> ACList = ahc->getAcendantList(childIndex);

            vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator ac = ACList.begin();
            vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator acEnd = ACList.end();

            for (; ac != acEnd; ac++)
            {
                Vector3 posAc = (*ac)->globalPosition.getValue();
                OutDeriv T;
                T.getVCenter() = valueConst.getVCenter();
                T.getVOrientation() = valueConst.getVOrientation() + cross(posConst - posAc, valueConst.getVCenter());

                vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*> articulations = (*ac)->getArticulations();

                vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*>::const_iterator a = articulations.begin();
                vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*>::const_iterator aEnd = articulations.end();

                int parent = (*ac)->parentIndex.getValue();

                for (; a != aEnd; a++)
                {
                    Vector3 axis = xto[parent].getOrientation().rotate((*a)->axis.getValue());

                    InSparseDeriv constArt;
                    constArt.index = (*a)->articulationIndex.getValue();
                    if ((*a)->rotation.getValue())
                    {
                        constArt.data = dot(axis, T.getVOrientation());
                    }
                    if ((*a)->translation.getValue())
                    {
                        constArt.data = dot(axis, T.getVCenter());
                        //printf("\n weightedNormalArticulation : %f", constArt.data);
                    }
                    out[i].push_back(constArt);
                }
            }
        }
    }
}

template <class BasicMapping>
void ArticulatedSystemMapping<BasicMapping>::draw()
{
    //if (!this->getShow()) return;
    //OutVecCoord& xto = *this->toModel->getX();
    //glDisable (GL_LIGHTING);
    //glPointSize(2);
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
