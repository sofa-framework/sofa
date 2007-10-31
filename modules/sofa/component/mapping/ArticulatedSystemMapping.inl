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

namespace sofa
{

namespace component
{

namespace mapping
{

template <class BasicMapping>
void ArticulatedSystemMapping<BasicMapping>::init()
{
    ArticulatedHierarchyContainer* ahc;
    GNode* context = dynamic_cast<GNode*>(this->fromModel->getContext());
    context->getNodeObject(ahc);
    articulationCenters = ahc->getArticulationCenters();

    vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator ac = articulationCenters.begin();
    vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator acEnd = articulationCenters.end();

    OutVecCoord& xto = *this->toModel->getX();

    for (; ac != acEnd; ac++)
    {
        int parent = (*ac)->parentIndex.getValue();

        // The position of the articulation center can be deduced using the 6D position of the parent:
        // only useful for visualisation of the mapping
        (*ac)->globalPosition.setValue(xto[parent].getCenter() +
                xto[parent].getOrientation().rotate((*ac)->posOnParent.getValue()));
    }
}

template <class BasicMapping>
void ArticulatedSystemMapping<BasicMapping>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
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

    //OutVecCoord& xto = *this->toModel->getX();
    //InVecCoord& xfrom = *this->fromModel->getX();

    for(unsigned int i=0; i<in.size(); i++)
    {
        for(unsigned int j=0; j<out.size(); j++)
        {
        }
        //	for (unsigned int j=0;j<in[i].size();j++)
        //	{
        //		const OutSparseDeriv cIn = in[i][j];
        //		int childIndex = cIn.index;
        //		const OutDeriv valueConst = (OutDeriv) cIn.data;
        //		Vector3 posConst = xto[childIndex].getCenter();;
        //		Object6D child;
        //		ArticulationCenter ac = child.getArticulationCentersAsChild();
        //		vector<ArticulationCenter> ACList;
        //		ac.getAcendantList(ACList);
        //		for(unsigned int acIt=0; acIt<ACList.size(); acIt++)
        //		{
        //			Vector3 posAc = ACList[acIt].getPos();
        //			OutDeriv T;
        //			T.getVCenter() = valueConst.getVCenter();
        //			T.getVOrientation() = T.getVOrientation() + cross(posConst - posAc, valueConst.getVCenter());
        //
        //			for(unsigned int articulationIndex=0; articulationIndex<ACList[acIt].articulationVector.size(); articulationIndex++)
        //			{
        //				Articulation& articulation = ACList[acIt].articulationVector[articulationIndex];
        //				Vector3 axis = xto[childIndex].getOrientation().rotate(articulation.axis);

        //				//InSparseDeriv constArt;
        //				//constArt.index = articulation.articulationIndex;
        //				//if (articulation.rotation)
        //				//{
        //				//	constArt.data = dot(axis, T.getVOrientation());
        //				//}
        //				//if (articulation.translation)
        //				//{
        //				//	constArt.data = dot(axis, T.getVCenter());
        //				//}
        //				//out[i].push_back(constArt);
        //			}
        //		}
        //	}
    }

}

template <class BasicMapping>
void ArticulatedSystemMapping<BasicMapping>::draw()
{
    //if (!getShow(this)) return;
    //glDisable (GL_LIGHTING);
    //glPointSize(10);
    //glColor4f (1,0,0,0);
    //glBegin (GL_POINTS);
    //std::vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator ac = articulationCenters.begin();
    //std::vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator acEnd = articulationCenters.end();
    //for (; ac != acEnd; ac++)
    //{
    //	helper::gl::glVertexT((*ac)->globalPosition.getValue());
    //}
    //glEnd();
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
