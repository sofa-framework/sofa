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
#ifndef SOFA_COMPONENT_ENGINE_PROXIMITYROI_INL
#define SOFA_COMPONENT_ENGINE_PROXIMITYROI_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <iostream>
#include <algorithm>
#include <vector>
#include <limits>

#include <SofaGeneralEngine/ProximityROI.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/BasicShapes.h>

namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
ProximityROI<DataTypes>::ProximityROI()
    : centers( initData(&centers, "centers", "Center(s) of the sphere(s)") )
    , radii( initData(&radii, "radii", "Radius(i) of the sphere(s)") )
    , f_num( initData (&f_num, "N", "Maximum number of points to select") )
    , f_X0( initData (&f_X0, "position", "Rest position coordinates of the degrees of freedom") )
    , f_indices( initData(&f_indices,"indices","Indices of the points contained in the ROI") )
    , f_pointsInROI( initData(&f_pointsInROI,"pointsInROI","Points contained in the ROI") )
    , f_distanceInROI( initData(&f_distanceInROI,"distance","distance between the points contained in the ROI and the closest center.") )
    , f_indicesOut( initData(&f_indicesOut,"indicesOut","Indices of the points not contained in the ROI") )
    , p_drawSphere( initData(&p_drawSphere,false,"drawSphere","Draw shpere(s)") )
    , p_drawPoints( initData(&p_drawPoints,false,"drawPoints","Draw Points") )
    , _drawSize( initData(&_drawSize,0.0,"drawSize","rendering size for box and topological elements") )
{
    //Adding alias to handle TrianglesInSphereROI input/output
    addAlias(&p_drawSphere,"isVisible");
    addAlias(&f_indices,"pointIndices");
    addAlias(&f_X0,"rest_position");

    f_indices.beginEdit()->push_back(0);
    f_indices.endEdit();
}

template <class DataTypes>
void ProximityROI<DataTypes>::init()
{
    if (!f_X0.isSet())
    {
        sofa::core::behavior::MechanicalState<DataTypes>* mstate;
        this->getContext()->get(mstate);
        if (mstate)
        {
            sofa::core::objectmodel::BaseData* parent = mstate->findData("rest_position");
            if (parent)
            {
                f_X0.setParent(parent);
                f_X0.setReadOnly(true);
            }
        }
        else
        {
            core::loader::MeshLoader* loader = NULL;
            this->getContext()->get(loader);
            if (loader)
            {
                sofa::core::objectmodel::BaseData* parent = loader->findData("position");
                if (parent)
                {
                    f_X0.setParent(parent);
                    f_X0.setReadOnly(true);
                }
            }
        }
    }

    addInput(&f_X0);

    addInput(&centers);
    addInput(&radii);
    addInput(&f_num);

    addOutput(&f_indices);
    addOutput(&f_pointsInROI);
    addOutput(&f_distanceInROI);
    addOutput(&f_indicesOut);

    setDirtyValue();
}

template <class DataTypes>
void ProximityROI<DataTypes>::reinit()
{
    update();
}

class SortingPair
{
public:
    SortingPair(int id, double d) {i=id; distance=d;}
    double distance;
    unsigned int i;
    bool operator<(const SortingPair& p) const
    {
        return distance<p.distance;
    }
};

template <class DataTypes>
void ProximityROI<DataTypes>::update()
{
    const helper::vector<Vec3>& cen = (centers.getValue());
    const helper::vector<Real>& rad = (radii.getValue());

    if (cen.empty())
        return;

    if (rad.empty())
    {
        serr << "The parameter 'Radius' must at least contains one value. This ROI is then disabled/useless." << sendl;
        return;
    }

    if (f_num.getValue()==0)
    {
        serr << "The parameter 'N' must have a value greater than zero. This ROI is then disabled/useless." << sendl;
        return;
    }

    // When there is sphere without corresponding radius, the missing radius
    // are filled with the last known value.
    if (cen.size() > rad.size())
    {
        helper::WriteAccessor< Data<helper::vector<Real> > > rada = radii;
        Real value=rad[rad.size()-1];
        for(unsigned int i=rad.size(); i<cen.size(); ++i)
        {
            rada.push_back(value);
        }
    }

    if(cen.size() < rad.size())
    {
        serr << "There parameter 'Radius' has more elements than parameters 'center'." << sendl;
    }


    const VecCoord* x0 = &f_X0.getValue();

    cleanDirty();


    // Write accessor for topological element indices
    SetIndex& indices = *(f_indices.beginWriteOnly());
    SetIndex& indicesOut = *(f_indicesOut.beginWriteOnly());

    // Write accessor for toplogical element
    helper::WriteOnlyAccessor< Data<VecCoord > > pointsInROI = f_pointsInROI;
    helper::WriteOnlyAccessor< Data<helper::vector<Real> > > distanceInROI = f_distanceInROI;

    // Clear lists
    indices.clear();
    indicesOut.clear();

    distanceInROI.clear();
    pointsInROI.clear();


    std::vector<SortingPair> sortingheap;

    std::make_heap(sortingheap.begin(), sortingheap.end());

    for( unsigned i=0; i<x0->size(); ++i )
    {
        Real mindist=std::numeric_limits<Real>::max();
        for (unsigned int j=0; j<cen.size(); ++j)
        {
            Real dist=(cen[j]-(*x0)[i]).norm();
            if(dist < rad[j] && mindist > dist)
                mindist = dist;
        }

        if(mindist==std::numeric_limits<Real>::max())
        {
            indicesOut.push_back(i);
            continue;
        }

        if(sortingheap.size()==0)
        {
            sortingheap.push_back(SortingPair(i, mindist));
            push_heap (sortingheap.begin(),sortingheap.end());
        }
        else
        {
            SortingPair& p=sortingheap.front();
            if(p.distance >= mindist)
            {
                if(sortingheap.size() >= f_num.getValue())
                {
                    // Remove the too large distance
                    pop_heap (sortingheap.begin(),sortingheap.end());
                    sortingheap.pop_back();
                    indicesOut.push_back(p.i);
                }
                // Insert the new one.
                sortingheap.push_back(SortingPair(i, mindist));
                push_heap (sortingheap.begin(),sortingheap.end());
            }
            else if(sortingheap.size() < f_num.getValue())
            {
                // Insert the new value.
                sortingheap.push_back(SortingPair(i, mindist));
                push_heap (sortingheap.begin(),sortingheap.end());
            }
            else
            {
                indicesOut.push_back(i);
            }
        }
    }



    for(std::vector<SortingPair>::iterator it=sortingheap.begin(); it!=sortingheap.end(); it++)
    {
        indices.push_back(it->i);
        distanceInROI.push_back((Real)it->distance);
        pointsInROI.push_back((*x0)[it->i]);
    }

    f_indices.endEdit();
    f_indicesOut.endEdit();
    f_pointsInROI.endEdit();
}

template <class DataTypes>
void ProximityROI<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (!vparams->displayFlags().getShowBehaviorModels())
        return;

    glColor3f(0.0, 1.0, 1.0);

    if(p_drawSphere.getValue()) // old classical drawing by points
    {
        ///draw the boxes
        const helper::vector<Vec3>& c=centers.getValue();
        const helper::vector<Real>& r=radii.getValue();

        for (unsigned int i=0; i<c.size() && i<r.size(); ++i)
        {
            helper::gl::drawWireSphere(c[i], (float)(r[i]/2.0));
        }
    }

    ///draw points in ROI
    if( p_drawPoints.getValue())
    {
        glDisable(GL_LIGHTING);
        glBegin(GL_POINTS);
        glPointSize(5.0);
        helper::ReadAccessor< Data<VecCoord > > pointsInROI = f_pointsInROI;
        for (unsigned int i=0; i<pointsInROI.size() ; ++i)
        {
            CPos p = DataTypes::getCPos(pointsInROI[i]);
            helper::gl::glVertexT(p);
        }
        glEnd();
    }
#endif /* SOFA_NO_OPENGL */
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
