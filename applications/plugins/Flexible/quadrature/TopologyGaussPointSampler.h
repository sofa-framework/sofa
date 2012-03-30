/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_TopologyGaussPointSAMPLER_H
#define SOFA_TopologyGaussPointSAMPLER_H

#include "initFlexible.h"
#include "BaseGaussPointSampler.h"
#include <sofa/core/topology/BaseMeshTopology.h>


namespace sofa
{
namespace component
{
namespace engine
{

using helper::vector;

/**
 * This class samples an object represented by a mesh
 */


class TopologyGaussPointSampler : public BaseGaussPointSampler
{
public:
    typedef BaseGaussPointSampler Inherited;
    SOFA_CLASS(TopologyGaussPointSampler,Inherited);

    typedef Inherited::Real Real;
    typedef Inherited::Coord Coord;
    typedef Inherited::SeqPositions SeqPositions;
    typedef Inherited::raPositions raPositions;
    typedef Inherited::waPositions waPositions;

    Data< SeqPositions > f_inPosition;
    typedef sofa::core::topology::BaseMeshTopology Topo;
    Topo* parentTopology;

    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const TopologyGaussPointSampler* = NULL) { return std::string();    }

    TopologyGaussPointSampler()    :   Inherited()
        , f_inPosition(initData(&f_inPosition,SeqPositions(),"inPosition","input node positions"))
    {
    }

    virtual void init()
    {
        Inherited::init();

        this->getContext()->get(parentTopology,core::objectmodel::BaseContext::SearchParents);
        if(!this->parentTopology) serr<<"MeshTopology not found"<<sendl;

        addInput(&f_inPosition);
        setDirtyValue();
    }

    virtual void reinit() { update(); }

protected:

    virtual void update()
    {
        cleanDirty();

        if(!this->parentTopology) return;

        raPositions parent(this->f_inPosition);
        if(!parent.size()) return;

        const Topo::SeqTetrahedra& tetrahedra = this->parentTopology->getTetrahedra();
        const Topo::SeqHexahedra& cubes = this->parentTopology->getHexahedra();
        const Topo::SeqTriangles& triangles = this->parentTopology->getTriangles();
        const Topo::SeqQuads& quads = this->parentTopology->getQuads();
        const Topo::SeqEdges& edges = this->parentTopology->getEdges();

        waPositions pos(this->f_position);
        waVolume vol(this->f_volume);
        waWeight w(this->f_weight);

        if(this->f_order.getValue()!=0) serr<<"High order volume integrals ot yet implemented"<<sendl;

        if(this->f_method.getValue().getSelectedId() == MIDPOINT)
        {
            if ( tetrahedra.empty() && cubes.empty() )
            {
                if ( triangles.empty() && quads.empty() )
                {
                    if ( edges.empty() ) return;
                    //no 3D elements, nor 2D elements -> map on 1D elements
                    pos.resize ( edges.size() );
                    vol.resize ( pos.size() );
                    for (unsigned int i=0; i<edges.size(); i++ )
                    {
                        const Coord& p1=parent[edges[i][0]],p2=parent[edges[i][1]];
                        pos[i] = (p1+p2)*0.5;
                        vol[i].resize(1); vol[i][0]=(p1-p2).norm();
                    }
                }
                else
                {
                    // no 3D elements -> map on 2D elements
                    pos.resize ( triangles.size() +quads.size() );
                    vol.resize ( pos.size() );
                    for ( unsigned int i = 0; i < triangles.size(); i++ )
                    {
                        const Coord& p1=parent[triangles[i][0]],p2=parent[triangles[i][1]],p3=parent[triangles[i][2]];
                        pos[i] = (p1+p2+p3)/(Real)3.;
                        vol[i].resize(1); vol[i][0] = cross(p2-p1,p3-p1).norm()*0.5;
                    }
                    int c0 = triangles.size();
                    for ( unsigned int i = 0; i < quads.size(); i++ )
                    {
                        const Coord& p1=parent[quads[i][0]],p2=parent[quads[i][1]],p3=parent[quads[i][2]],p4=parent[quads[i][3]];
                        pos[i+c0] = (p1+p2+p3+p4)*0.25;
                        vol[i+c0].resize(1); vol[i+c0][0] = cross(p2-p1,p3-p1).norm();
                    }
                }
            }
            else
            {
                // map on 3D elements
                pos.resize ( tetrahedra.size() +cubes.size() );
                vol.resize ( pos.size() );
                for ( unsigned int i = 0; i < tetrahedra.size(); i++ )
                {
                    const Coord& p1=parent[tetrahedra[i][0]],p2=parent[tetrahedra[i][1]],p3=parent[tetrahedra[i][2]],p4=parent[tetrahedra[i][3]];
                    pos[i] = (p1+p2+p3+p4)*0.25;
                    vol[i].resize(1); vol[i][0] = fabs(dot(cross(p4-p1,p3-p1),p2-p1))/(Real)6.;
                }
                int c0 = tetrahedra.size();
                for ( unsigned int i = 0; i < cubes.size(); i++ )
                {
                    const Coord& p1=parent[cubes[i][0]],p2=parent[cubes[i][1]],p3=parent[cubes[i][2]],p4=parent[cubes[i][3]],p5=parent[cubes[i][4]],p6=parent[cubes[i][5]],p7=parent[cubes[i][6]],p8=parent[cubes[i][7]];
                    pos[i+c0] = (p1+p2+p3+p4+p5+p6+p7+p8)*0.125;
                    vol[i+c0].resize(1); vol[i+c0][0] = fabs(dot(cross(p4-p1,p3-p1),p2-p1));
                }
            }

            w.resize(pos.size()); for (unsigned int i=0; i<pos.size(); i++) w[i]=(Real)1.;
        }
        else if(this->f_method.getValue().getSelectedId() == SIMPSON)
        {
            serr<<"SIMPSON quadrature not yet implemented"<<sendl;
        }
        else if(this->f_method.getValue().getSelectedId() == GAUSSLEGENDRE)
        {
            serr<<"GAUSSLEGENDRE quadrature not yet implemented"<<sendl;
        }

        if(this->f_printLog.getValue()) if(pos.size())    std::cout<<"TopologyGaussPointSampler: "<< pos.size() <<" generated samples"<<std::endl;
    }


    virtual void draw(const core::visual::VisualParams* vparams)
    {
        Inherited::draw(vparams);
    }



};

}
}
}

#endif
