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
#ifndef SOFA_COMPONENT_MAPPING_IMPLICITSURFACEMAPPING_H
#define SOFA_COMPONENT_MAPPING_IMPLICITSURFACEMAPPING_H
#include <SofaImplicitField/config.h>

#include <sofa/core/Mapping.h>
#include <sofa/component/topology/container/constant/MeshTopology.h>
#include <sofa/helper/MarchingCubeUtility.h>
#include <sofa/defaulttype/VecTypes.h>


namespace sofa
{

namespace component
{

namespace mapping
{

template <class In, class Out>
class ImplicitSurfaceMapping : public core::Mapping<In, Out>, public topology::container::constant::MeshTopology
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE2(ImplicitSurfaceMapping, In, Out), SOFA_TEMPLATE2(core::Mapping, In, Out),  topology::container::constant::MeshTopology);

    typedef core::Mapping<In, Out> Inherit;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename OutCoord::value_type OutReal;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename InCoord::value_type InReal;

    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;

    typedef Data<InVecCoord> InDataVecCoord;
    typedef Data<InVecDeriv> InDataVecDeriv;
    typedef Data<InMatrixDeriv> InDataMatrixDeriv;

    typedef Data<OutVecCoord> OutDataVecCoord;
    typedef Data<OutVecDeriv> OutDataVecDeriv;
    typedef Data<OutMatrixDeriv> OutDataMatrixDeriv;
protected:
    ImplicitSurfaceMapping()
        : Inherit(),
          mStep(initData(&mStep,0.5,"step","Step")),
          mRadius(initData(&mRadius,2.0,"radius","Radius")),
          mIsoValue(initData(&mIsoValue,0.5,"isoValue","Iso Value")),
          mGridMin(initData(&mGridMin,InCoord(-100,-100,-100),"min","Grid Min")),
          mGridMax(initData(&mGridMax,InCoord(100,100,100),"max","Grid Max"))
    {
    }

    ~ImplicitSurfaceMapping() override
    {
    }
public:
    void init() override;

    void parse(core::objectmodel::BaseObjectDescription* arg) override;

    double getStep() const { return mStep.getValue(); }
    void setStep(double val) { mStep.setValue(val); }

    double getRadius() const { return mRadius.getValue(); }
    void setRadius(double val) { mRadius.setValue(val); }

    double getIsoValue() const { return mIsoValue.getValue(); }
    void setIsoValue(double val) { mIsoValue.setValue(val); }

    const InCoord& getGridMin() const { return mGridMin.getValue(); }
    void setGridMin(const InCoord& val) { mGridMin.setValue(val); }
    void setGridMin(double x, double y, double z) { mGridMin.setValue( InCoord((InReal)x,(InReal)y,(InReal)z)); }

    const InCoord& getGridMax() const { return mGridMax.getValue(); }
    void setGridMax(const InCoord& val) { mGridMax.setValue(val); }
    void setGridMax(double x, double y, double z) { mGridMax.setValue( InCoord((InReal)x,(InReal)y,(InReal)z)); }

    void apply(const core::MechanicalParams *mparams, Data<OutVecCoord>& out, const Data<InVecCoord>& in) override;
    void applyJ(const core::MechanicalParams *mparams, Data<OutVecDeriv>& out, const Data<InVecDeriv>& in) override;
    void applyJT( const sofa::core::MechanicalParams* /*mparams*/, InDataVecDeriv& /*out*/, const OutDataVecDeriv& /*in*/) override
    {
        msg_error() << "applyJT(dx) is not implemented";
    }

    void applyJT( const sofa::core::ConstraintParams* /*cparams*/, InDataMatrixDeriv& /*out*/, const OutDataMatrixDeriv& /*in*/) override
    {
        msg_error() << "applyJT(constraint) is not implemented";
    }

protected:
    Data <double > mStep; ///< Step
    Data <double > mRadius; ///< Radius
    Data <double > mIsoValue; ///< Iso Value

    Data< InCoord > mGridMin; ///< Grid Min
    Data< InCoord > mGridMax; ///< Grid Max

    // Marching cube data

    /// For each cube, store the vertex indices on each 3 first edges, and the data value
    struct CubeData
    {
        int p[3];
        OutReal data;
        inline friend std::istream& operator >> ( std::istream& in, CubeData& c)
        {
            in >> c.p[0] >> c.p[1] >> c.p[2] >> c.data;

            return in;
        }

        inline friend std::ostream& operator << ( std::ostream& out, const CubeData& c)
        {
            out << c.p[0] << " " << c.p[1] << " " << c.p[2] << " " << c.data ;
            return out;
        }
    };

    Data < sofa::type::vector<CubeData> > planes;
    typename sofa::type::vector<CubeData>::iterator P0; /// Pointer to first plane
    typename sofa::type::vector<CubeData>::iterator P1; /// Pointer to second plane

    void newPlane();

    template<int C>
    int addPoint(OutVecCoord& out, int x,int y,int z, OutReal v0, OutReal v1, OutReal iso)
    {
        int p = out.size();
        OutCoord pos = OutCoord((OutReal)x,(OutReal)y,(OutReal)z);
        pos[C] -= (iso-v0)/(v1-v0);
        out.resize(p+1);
        out[p] = pos * mStep.getValue();
        return p;
    }

    int addFace(int p1, int p2, int p3, int nbp)
    {
        if ((unsigned)p1<(unsigned)nbp &&
            (unsigned)p2<(unsigned)nbp &&
            (unsigned)p3<(unsigned)nbp)
        {
            SeqTriangles& triangles = *d_seqTriangles.beginEdit();
            int f = triangles.size();
            triangles.push_back(Triangle(p1, p3, p2));
            d_seqTriangles.endEdit();
            return f;
        }
        else
        {
            msg_error() << "Invalid face "<<p1<<" "<<p2<<" "<<p3;
            return -1;
        }
    }

public:
    bool insertInNode( core::objectmodel::BaseNode* node ) override { Inherit1::insertInNode(node); Inherit2::insertInNode(node); return true; }
    bool removeInNode( core::objectmodel::BaseNode* node ) override { Inherit1::removeInNode(node); Inherit2::removeInNode(node); return true; }

};

// MARCHING CUBE TABLES
// ( table copied from an article of Paul Bourke )
// based on code by Cory Gene Bloyd

/* Convention:

       Z
       ^
       |
       4----4----5
      /|        /|
     7 |       5 |
    /  8      /  9
   7---+6----6   |
   |   |     |   |
   |   0----0+---1--> X
   11  /     10  /
   | 3       | 1
   |/        |/
   3----2----2
   /
  /
|_
Y

*/


#if  !defined(SOFA_COMPONENT_MAPPING_IMPLICITSURFACEMAPPING_CPP)
extern template class SOFA_SOFAIMPLICITFIELD_API ImplicitSurfaceMapping< defaulttype::Vec3dTypes, defaulttype::Vec3dTypes >;



#endif


} // namespace mapping

} // namespace component

} // namespace sofa

#endif
