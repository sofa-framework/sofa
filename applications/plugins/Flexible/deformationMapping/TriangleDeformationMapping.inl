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
#ifndef SOFA_COMPONENT_MAPPING_TriangleDeformationMapping_INL
#define SOFA_COMPONENT_MAPPING_TriangleDeformationMapping_INL

#include "TriangleDeformationMapping.h"
#include <sofa/core/visual/VisualParams.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;


template <class TIn, class TOut>
TriangleDeformationMapping<TIn, TOut>::TriangleDeformationMapping()
    : Inherit()
    , f_inverseRestEdges(initData(&f_inverseRestEdges, "restLengths", "Rest lengths of the connections."))
    , f_scaleView(initData(&f_scaleView, (SReal)1.0, "scaleView", "Scale the display of the deformation gradients."))
{
}

template <class TIn, class TOut>
TriangleDeformationMapping<TIn, TOut>::~TriangleDeformationMapping()
{
}


template <class TIn, class TOut>
void TriangleDeformationMapping<TIn, TOut>::init()
{
    triangleContainer = dynamic_cast<topology::TriangleSetTopologyContainer*>( this->getContext()->getMeshTopology() );
    if( !triangleContainer ) serr<<"No TriangleSetTopologyContainer found ! "<<sendl;

    const SeqTriangles& triangles = triangleContainer->getTriangles();

    this->getToModel()->resize( triangles.size() );



    // compute the reference matrices if they are not known
    if( f_inverseRestEdges.getValue().size() != triangles.size() )
    {
        typename core::behavior::MechanicalState<In>::ReadVecCoord pos = this->getFromModel()->readPositions();
        helper::WriteOnlyAccessor< Data<VMMat> > inverseRestEdges(f_inverseRestEdges);
        inverseRestEdges.resize( triangles.size() );

        // look for the shape function, to get the material coordinates
        /*      ShapeFunction* shapeFunction=NULL;
              this->getContext()->get(shapeFunction,core::objectmodel::BaseContext::SearchUp);

              if( shapeFunction!=NULL && shapeFunction->f_position.getValue().size() == pos.size() ) // if material coordinates are available
              {
                  helper::ReadAccessor<Data<VMCoord> > mcoords(shapeFunction->f_position);
        //            cerr<<"TriangleDeformationMapping<TIn, TOut>::init(), found material coordinates : " << mcoords << endl ;
                  for(unsigned i=0; i<triangles.size(); i++ )
                  {
                      MMat m;
                      m[0] = mcoords[triangles[i][1]] - mcoords[triangles[i][0]];   // edge01
                      m[1] = mcoords[triangles[i][2]] - mcoords[triangles[i][0]];   // edge02
                      m.transpose();
                      bool inverted = invertMatrix(inverseRestEdges[i], m);

                      if( !inverted  ){
                          cerr<<"TriangleDeformationMapping<TIn, TOut>::init(), matrix not invertible: " << endl << m << endl;
                      }
        //                else {
        //                    cerr<<"TriangleDeformationMapping<TIn, TOut>::init(), edge matrix: " << endl << m << endl;
        //                    cerr<<"TriangleDeformationMapping<TIn, TOut>::init(), inverted matrix: " << endl << inverseRestEdges[i] << endl;
        //                    cerr<<"TriangleDeformationMapping<TIn, TOut>::init(), product: " << endl << m * inverseRestEdges[i] << endl;
        //                }
                  }
              }
              else*/
        for(unsigned i=0; i<triangles.size(); i++ ) // otherwise use the world coordinates to create a local parametrization
        {
            MMat m;
            InDeriv edge01 = pos[triangles[i][1]] - pos[triangles[i][0]];
            m[0] = MCoord( edge01.norm(), 0. );                      // first axis aligned with first edge
            InDeriv edge02 = pos[triangles[i][2]] - pos[triangles[i][0]];
            InDeriv normal = cross(edge01,edge02);
            InDeriv v = cross(normal,edge01);                        // second axis orthogonal to the first, in the plane of the triangle
            m[1] = MCoord( edge01*edge02/edge01.norm(), v*edge02 );  // second edge in the local orthonormal frame

            if( ! invertMatrix(inverseRestEdges[i], m) )
            {
                cerr<<"TriangleDeformationMapping<TIn, TOut>::init(), matrix not invertible: " << endl << m << endl;
            }
            //            else cerr<<"TriangleDeformationMapping<TIn, TOut>::init(), inverted matrix " << endl << m << endl;
        }
    }

    baseMatrices.resize( 1 );
    baseMatrices[0] = &jacobian;

    this->Inherit::init();  // applies the mapping, so after the Data init
}

// Return a 9*3 matrix made of three scaled identity matrices.
template <class TIn, class TOut>
inline typename TriangleDeformationMapping<TIn, TOut>::Block TriangleDeformationMapping<TIn, TOut>::makeBlock( Real middle, Real bottom )
{
    Block b;  // initialized to 0
    for(unsigned i=0; i<3; i++)
    {
        b[0+2*i][i] = middle;  // influence on the two axes, interleaved because the axes are the columns of the matrix
        b[1+2*i][i] = bottom;
    }
//    cerr<<"TriangleDeformationMapping<TIn, TOut>::createBlock " << endl << b << endl;
    return b;
}

template <class TIn, class TOut>
void TriangleDeformationMapping<TIn, TOut>::apply(const core::MechanicalParams * /*mparams*/ , Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    helper::WriteOnlyAccessor< Data<OutVecCoord> >  F = dOut;
    helper::ReadAccessor< Data<InVecCoord> >  pos = dIn;
    helper::ReadAccessor<Data<VMMat> > inverseRestEdges(f_inverseRestEdges);
    SeqTriangles triangles = triangleContainer->getTriangles();

    jacobian.resizeBlocks(F.size(),pos.size());

    for(unsigned i=0; i<triangles.size(); i++ )
    {
        Frame F1;
        for(unsigned j=0; j<Nin; j++)
        {
            F1[j][0] = pos[triangles[i][1]][j] - pos[triangles[i][0]][j];  // edge01
            F1[j][1] = pos[triangles[i][2]][j] - pos[triangles[i][0]][j];  // edge02
        }

        const MMat& M = inverseRestEdges[i];
        F[i].getF() = F1 * M;
//        cerr<<"TriangleDeformationMapping<TIn, TOut>::apply, ref matrix = " << inverseRestEdges[i] << endl;
//        cerr<<"TriangleDeformationMapping<TIn, TOut>::apply, edges = " << F1.transposed() << endl;
//        cerr<<"TriangleDeformationMapping<TIn, TOut>::apply, F.getF() = " << F[i].getF() << endl;
//        cerr<<"TriangleDeformationMapping<TIn, TOut>::apply, F.getF() transposed = " << F[i].getF().transposed() << endl;

        jacobian.beginBlockRow(i);
        // each block defined by its column and its contributions to centre, first axis, second axis, respectively
        jacobian.createBlock( triangles[i][0], makeBlock( -M[0][0]-M[1][0], -M[0][1]-M[1][1]) );
        jacobian.createBlock( triangles[i][1], makeBlock(  M[0][0]        ,  M[0][1]        ) );
        jacobian.createBlock( triangles[i][2], makeBlock(          M[1][0],          M[1][1]) );
        jacobian.endBlockRow();
    }

//    cerr<<"TriangleDeformationMapping<TIn, TOut>::apply, jacobian: "<<endl<< jacobian << endl;

}


template <class TIn, class TOut>
void TriangleDeformationMapping<TIn, TOut>::applyJ(const core::MechanicalParams * /*mparams*/ , Data<OutVecDeriv>& dOut, const Data<InVecDeriv>& dIn)
{
    if( jacobian.rowSize() > 0 )
        jacobian.mult(dOut,dIn);
//    cerr<<"TriangleDeformationMapping<TIn, TOut>::applyJ, in  = " << dIn.getValue() << endl;
//    cerr<<"TriangleDeformationMapping<TIn, TOut>::applyJ, out = " << dOut.getValue() << endl;
}

template <class TIn, class TOut>
void TriangleDeformationMapping<TIn, TOut>::applyJT(const core::MechanicalParams * /*mparams*/ , Data<InVecDeriv>& dIn, const Data<OutVecDeriv>& dOut)
{
//    cerr<<"TriangleDeformationMapping<TIn, TOut>::applyJT, child  = " << dOut.getValue() << endl;
//    cerr<<"TriangleDeformationMapping<TIn, TOut>::applyJT, parent before = " << dIn.getValue() << endl;
    if( jacobian.rowSize() > 0 )
        jacobian.addMultTranspose(dIn,dOut);
//    cerr<<"TriangleDeformationMapping<TIn, TOut>::applyJT, parent after = " << dIn.getValue() << endl;
}


template <class TIn, class TOut>
void TriangleDeformationMapping<TIn, TOut>::applyJT(const core::ConstraintParams*, Data<InMatrixDeriv>& , const Data<OutMatrixDeriv>& )
{
    //    cerr<<"TriangleDeformationMapping<TIn, TOut>::applyJT(const core::ConstraintParams*, Data<InMatrixDeriv>& , const Data<OutMatrixDeriv>& ) does nothing " << endl;
}


template <class TIn, class TOut>
const sofa::defaulttype::BaseMatrix* TriangleDeformationMapping<TIn, TOut>::getJ()
{
    return &jacobian;
}

template <class TIn, class TOut>
const helper::vector<sofa::defaulttype::BaseMatrix*>* TriangleDeformationMapping<TIn, TOut>::getJs()
{
    return &baseMatrices;
}

template <class TIn, class TOut>
void TriangleDeformationMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
{
    if( !vparams->displayFlags().getShowMechanicalMappings() ) return;

    typename core::behavior::MechanicalState<In>::ReadVecCoord vert = this->getFromModel()->readPositions();
    typename core::behavior::MechanicalState<Out>::ReadVecCoord pos = this->getToModel()->readPositions();
    SeqTriangles triangles = triangleContainer->getTriangles();


    // x axes
    helper::vector< Vector3 > points;
    for(unsigned i=0; i<triangles.size(); i++ )
    {
//        cerr<<"TriangleDeformationMapping<TIn, TOut>::draw, F = " << endl << pos[i].getF() << endl;
//        cerr<<"TriangleDeformationMapping<TIn, TOut>::draw, F = " << endl << pos[i].getVec() << endl;
        Vector3 centre = vert[triangles[i][0]]*1./3 + vert[triangles[i][1]]*1./3 + vert[triangles[i][2]]*1./3;
        points.push_back(centre);
        unsigned id=0; // x
        InDeriv axis( pos[i].getF()[0][id], pos[i].getF()[1][id], pos[i].getF()[2][id] ) ;
        points.push_back(centre+  (axis * f_scaleView.getValue()));
    }
//    cerr<<"TriangleDeformationMapping<TIn, TOut>::draw, red lines = " << points << endl;
    vparams->drawTool()->drawLines ( points, 1, Vec<4,float> ( 1,0,0,1 ) ); // red

    // y axes
    points.clear();
    for(unsigned i=0; i<triangles.size(); i++ )
    {
        Vector3 centre = vert[triangles[i][0]]*1./3 + vert[triangles[i][1]]*1./3 + vert[triangles[i][2]]*1./3;
        points.push_back(centre);
        unsigned id=1; // y
        InDeriv axis( pos[i].getF()[0][id], pos[i].getF()[1][id], pos[i].getF()[2][id] );
        points.push_back(centre+  (axis*f_scaleView.getValue()) );
    }
    vparams->drawTool()->drawLines ( points, 1, Vec<4,float> ( 0,1,0,1 ) ); // green

}


} // namespace mapping

} // namespace component

} // namespace sofa

#endif
