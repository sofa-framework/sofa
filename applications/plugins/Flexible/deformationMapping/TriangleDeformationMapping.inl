/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_MAPPING_TriangleDeformationMapping_INL
#define SOFA_COMPONENT_MAPPING_TriangleDeformationMapping_INL

#include "../deformationMapping/TriangleDeformationMapping.h"
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

    SeqTriangles triangles = triangleContainer->getTriangles();

    this->getToModel()->resize( triangles.size() );

    // compute the rest lengths if they are not known
    if( f_inverseRestEdges.getValue().size() != triangles.size() )
    {
        helper::WriteAccessor< Data<vector<InBlock> > > inverseRestEdges(f_inverseRestEdges);
        typename core::behavior::MechanicalState<In>::ReadVecCoord pos = this->getFromModel()->readPositions();
        inverseRestEdges.resize( triangles.size() );
        for(unsigned i=0; i<triangles.size(); i++ )
        {
            InDeriv edge01 = pos[triangles[i][1]] - pos[triangles[i][0]];
            InDeriv edge02 = pos[triangles[i][2]] - pos[triangles[i][0]];
            InDeriv normal = cross(edge01,edge02); // no need to normalize it, as only the two first coordinates of the deformation gradient will be used during the simulation.

            // the edges and the normal are the columns of the matrix
            InBlock m;
            for(unsigned j=0; j<3; j++)
            {
                m[j][0] = edge01[j];
                m[j][1] = edge02[j];
                m[j][2] = normal[j];
            }
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



template <class TIn, class TOut>
void TriangleDeformationMapping<TIn, TOut>::apply(const core::MechanicalParams * /*mparams*/ , Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    helper::WriteAccessor< Data<OutVecCoord> >  F = dOut;
    helper::ReadAccessor< Data<InVecCoord> >  pos = dIn;
    helper::ReadAccessor<Data<vector<InBlock> > > inverseRestEdges(f_inverseRestEdges);
    SeqTriangles triangles = triangleContainer->getTriangles();

    //    jacobian.clear();
    jacobian.resizeBlocks(F.size(),pos.size());
//    directions.resize(out.size());
//    invlengths.resize(out.size());

    for(unsigned i=0; i<triangles.size(); i++ )
    {
        F[i].getCenter() = ( pos[triangles[i][0]] + pos[triangles[i][1]] + pos[triangles[i][2]] ) *1.0/3;  // centre of the triangle
        Frame F1;
        for(unsigned j=0; j<Nin; j++)
        {
            F1[j][0] = pos[triangles[i][1]][j] - pos[triangles[i][0]][j];  // edge01
            F1[j][1] = pos[triangles[i][2]][j] - pos[triangles[i][0]][j];  // edge02
        }
        F[i].getF() = inverseRestEdges[i] * F1;
        cerr<<"TriangleDeformationMapping<TIn, TOut>::apply, ref = " << inverseRestEdges[i] << endl;
        cerr<<"TriangleDeformationMapping<TIn, TOut>::apply, edges = " << F1 << endl;
        cerr<<"TriangleDeformationMapping<TIn, TOut>::apply, F = " << F[i] << endl;
        cerr<<"TriangleDeformationMapping<TIn, TOut>::apply, F.getF() = " << F[i].getF() << endl;
        cerr<<"TriangleDeformationMapping<TIn, TOut>::apply, F.getVec() = " << F[i].getVec() << endl;

//        // insert in increasing row and column order
//        jacobian.beginRow(i);
//        if( triangles[i][1]<triangles[i][0]){
//            for(unsigned j=0; j<Nout; j++){
//                for(unsigned k=0; k<Nin; k++ ){
//                    jacobian.set( i*Nout+j, triangles[i][1]*Nin+k, gap[k] );
//                }
//                for(unsigned k=0; k<Nin; k++ ){
//                    jacobian.set( i*Nout+j, triangles[i][0]*Nin+k, -gap[k] );
//                }
//            }
//        }
//        else {
//            for(unsigned j=0; j<Nout; j++){
//                for(unsigned k=0; k<Nin; k++ ){
//                    jacobian.set( i*Nout+j, triangles[i][0]*Nin+k, -gap[k] );
//                }
//                for(unsigned k=0; k<Nin; k++ ){
//                    jacobian.set( i*Nout+j, triangles[i][1]*Nin+k, gap[k] );
//                }
//            }
//        }
    }

//    jacobian.endEdit();
    //      cerr<<"TriangleDeformationMapping<TIn, TOut>::apply, jacobian: "<<endl<< jacobian << endl;

}

//template <class TIn, class TOut>
//void TriangleDeformationMapping<TIn, TOut>::computeGeometricStiffness(const core::MechanicalParams *mparams)
//{

//}

template <class TIn, class TOut>
void TriangleDeformationMapping<TIn, TOut>::applyJ(const core::MechanicalParams * /*mparams*/ , Data<OutVecDeriv>& dOut, const Data<InVecDeriv>& dIn)
{
    if( jacobian.rowSize() > 0 )
        jacobian.mult(dOut,dIn);
}

template <class TIn, class TOut>
void TriangleDeformationMapping<TIn, TOut>::applyJT(const core::MechanicalParams * /*mparams*/ , Data<InVecDeriv>& dIn, const Data<OutVecDeriv>& dOut)
{
    if( jacobian.rowSize() > 0 )
        jacobian.addMultTranspose(dIn,dOut);
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
const vector<sofa::defaulttype::BaseMatrix*>* TriangleDeformationMapping<TIn, TOut>::getJs()
{
    return &baseMatrices;
}

template <class TIn, class TOut>
void TriangleDeformationMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
{
    typename core::behavior::MechanicalState<Out>::ReadVecCoord pos = this->getToModel()->readPositions();
    SeqTriangles triangles = triangleContainer->getTriangles();


    // x axes
    vector< Vec3d > points;
    for(unsigned i=0; i<triangles.size(); i++ )
    {
        points.push_back(pos[i].getCenter());
        unsigned id=0; // x
        InDeriv axis( pos[i].getF()[id][0], pos[i].getF()[id][1], pos[i].getF()[id][2] ); // not sure that  getF()[id] is really the axis. Probably need to transpose.
        points.push_back(pos[i].getCenter()+axis);
    }
    vparams->drawTool()->drawLines ( points, 1, Vec<4,float> ( 1,0,0,1 ) ); // red

    // y axes
    points.clear();
    for(unsigned i=0; i<triangles.size(); i++ )
    {
        points.push_back(pos[i].getCenter());
        unsigned id=1; // y
        InDeriv axis( pos[i].getF()[id][0], pos[i].getF()[id][1], pos[i].getF()[id][2] ); // not sure that  getF()[id] is really the axis. Probably need to transpose.
        points.push_back(pos[i].getCenter()+axis);
    }
    vparams->drawTool()->drawLines ( points, 1, Vec<4,float> ( 0,1,0,1 ) ); // green
}


} // namespace mapping

} // namespace component

} // namespace sofa

#endif
