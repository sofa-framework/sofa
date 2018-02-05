/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef FLEXIBLE_BezierShapeFunction_H
#define FLEXIBLE_BezierShapeFunction_H

#include <Flexible/config.h>
#include "../shapeFunction/BarycentricShapeFunction.h"
#include <sofa/core/topology/BaseMeshTopology.h>
#include <SofaHighOrderTopology/HighOrderTetrahedronSetTopologyContainer.h>
#include <SofaHighOrderTopology/BezierTetrahedronSetGeometryAlgorithms.h>

#include <algorithm>
#include <iostream>

namespace sofa
{
namespace component
{
namespace shapefunction
{

/**
Bezier shape functions are the Bezier coordinates of points inside cells (can be edges, triangles, quads, tetrahedra, hexahedra)
there are computed from barycentric coordinates
  */

template <class ShapeFunctionTypes_>
class BezierShapeFunction : public BarycentricShapeFunction<ShapeFunctionTypes_>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(BezierShapeFunction, ShapeFunctionTypes_) , SOFA_TEMPLATE(BarycentricShapeFunction, ShapeFunctionTypes_));
    typedef BarycentricShapeFunction<ShapeFunctionTypes_> Inherit;

    typedef typename Inherit::Real Real;
    typedef typename Inherit::Coord Coord;
    typedef typename Inherit::VCoord VCoord;
    typedef typename Inherit::VReal VReal;
    typedef typename Inherit::VGradient VGradient;
    typedef typename Inherit::VHessian VHessian;
    typedef typename Inherit::VRef VRef;
    typedef typename Inherit::Cell Cell;

    typedef typename Inherit::Gradient Gradient;
    typedef typename Inherit::Hessian Hessian;
    enum {spatial_dimensions=Inherit::spatial_dimensions};

    typedef topology::HighOrderTetrahedronSetTopologyContainer BezierTopoContainer;
    typedef defaulttype::StdVectorTypes<defaulttype::Vec<Inherit::spatial_dimensions,Real>,defaulttype::Vec<Inherit::spatial_dimensions,Real>,Real> VecSpatialDimensionType;
    typedef topology::BezierTetrahedronSetGeometryAlgorithms<VecSpatialDimensionType> BezierGeoAlg;
    typedef typename BezierGeoAlg::Vec4 Vec4;
    typedef typename BezierGeoAlg::Mat44 Mat44;

protected:
    BezierTopoContainer* container;
    BezierGeoAlg* geoAlgo;
    helper::vector<topology::TetrahedronIndexVector> tbiArray;

public:

    template<class Real1, class Real2,  int Dim1, int Dim2>
    inline defaulttype::Mat<Dim1, Dim2, Real2> covMN(const defaulttype::Vec<Dim1,Real1>& v1, const defaulttype::Vec<Dim2,Real2>& v2)
    {
        defaulttype::Mat<Dim1, Dim2, Real2> res;
        for ( unsigned int i = 0; i < Dim1; ++i)
            for ( unsigned int j = 0; j < Dim2; ++j)
            {
                res[i][j] = (Real2)v1[i] * v2[j];
            }
        return res;
    }

    void computeShapeFunction(const Coord& childPosition, VRef& ref, VReal& w, VGradient* dw=NULL,VHessian* ddw=NULL, const Cell cell=-1)
    {
        Inherit::computeShapeFunction(childPosition,ref,w,dw,ddw,cell);

        if(!this->container ||  !this->geoAlgo) return;
        if(this->cellIndex==-1) return;
        if(w.size()!=4) return;

       
        const BezierTopoContainer::VecPointID &indexArray=container->getGlobalIndexArrayOfControlPoints(this->cellIndex);

        size_t nbRef = tbiArray.size();
        //        this->f_nbRef.setValue(nbRef);
        VRef ref_n(nbRef);
        VReal w_n(nbRef);
        VGradient dw_n(nbRef);
        VHessian ddw_n(nbRef);
        for(size_t i=0; i<nbRef; ++i)
        {
            ref_n[i] = indexArray[i];
            Vec4 barycentricCoordinate(w[0],w[1],w[2],w[3]);
            w_n[i] = this->geoAlgo->computeShapeFunction(tbiArray[i],barycentricCoordinate);
            if(dw)
            {
                Vec4 dval = this->geoAlgo->computeShapeFunctionDerivatives(tbiArray[i],barycentricCoordinate);
                for(unsigned j=0; j<4; ++j) dw_n[i]+=(*dw)[j]*dval[j];
                if(ddw)
                {
                    Mat44 ddval = this->geoAlgo->computeShapeFunctionHessian(tbiArray[i],barycentricCoordinate);
                    for(unsigned j=0; j<4; ++j) ddw_n[i]+=(*ddw)[j]*dval[j];
                    for(unsigned j=0; j<4; ++j) for(unsigned k=0; k<4; ++k) ddw_n[i]+=covMN((*dw)[j],(*dw)[k])*ddval[j][k];
                }
            }
        }
        ref.assign(ref_n.begin(),ref_n.end());
        w.assign(w_n.begin(),w_n.end());
        if(dw)
        {
            dw->assign(dw_n.begin(),dw_n.end());
            if(ddw) ddw->assign(ddw_n.begin(),ddw_n.end());
        }
    }

    virtual void init()
    {
        Inherit::init();

        if(!this->container)
        {
            this->getContext()->get(container,core::objectmodel::BaseContext::SearchUp);
            if(!this->container) { serr<<"BezierTopologyContainer not found"<<sendl; return; }
        }
        if(!this->geoAlgo)
        {
            this->getContext()->get(geoAlgo,core::objectmodel::BaseContext::SearchUp);
            if(!this->geoAlgo) { serr<<"BezierGeometryAlgorithms not found"<<sendl; return; }
        }

        tbiArray=container->getTetrahedronIndexArray();
    }

protected:
    BezierShapeFunction()
        : Inherit()
        , container( NULL )
        , geoAlgo( NULL )
    {
    }

    virtual ~BezierShapeFunction()
    {

    }






};







}
}
}


#endif
