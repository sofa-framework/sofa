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
#ifndef CustomJacobianBlock_H
#define CustomJacobianBlock_H

#include "../BaseJacobian.h"

#define V1(type) StdVectorTypes<Vec<1,type>,Vec<1,type>,type>


namespace sofa
{

namespace defaulttype
{

/** Template class used to implement one custom jacobian block with constant J matrix */
template<class In, class Out>
class CustomJacobianBlock : public BaseJacobianBlock<In,Out>
{
public:
    typedef BaseJacobianBlock<In,Out> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::KBlock KBlock;
    typedef typename Inherit::Real Real;

    enum { dim = Out::spatial_dimensions };
    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,dim,Real> MaterialToSpatial;

    MatBlock J;
    OutCoord X0;
    static const bool constant=true;

    void init(const MatBlock& _J, const SpatialCoord& SPos, const MaterialToSpatial& /*M*/)    { J=_J; X0=SPos; }
    void addapply( OutCoord& result, const InCoord& data )    { result +=  J*data + X0; }
    void addmult( OutDeriv& result,const InDeriv& data )    { result += J*data; }
    void addMultTranspose( InDeriv& result, const OutDeriv& data )    { result += J.multTranspose(data); }
    MatBlock getJ()    { return J; }
    KBlock getK(const OutDeriv& /*childForce*/, bool=false) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const SReal& /*kfactor */) {}
};


template<class InReal,class OutReal>
class CustomJacobianBlock< V1(InReal) , F331(OutReal) > :
    public  BaseJacobianBlock< V1(InReal) , F331(OutReal) >
{
public:
    typedef V1(InReal) In;
    typedef F331(OutReal) Out;

    typedef BaseJacobianBlock<In,Out> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::KBlock KBlock;
    typedef typename Inherit::Real Real;

    enum { dim = Out::spatial_dimensions };
    enum { mdim = Out::material_dimensions };
    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,mdim,Real> MaterialToSpatial;

    MatBlock J;
    OutCoord X0;
    static const bool constant=true;

    void init(const MatBlock& _J, const SpatialCoord& /*SPos*/, const MaterialToSpatial& M)    { J=_J; for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<mdim; ++j) X0.getF()[i][j]=M[i][j];  }
    void addapply( OutCoord& result, const InCoord& data )    { result.getVec() +=  J*data + X0.getVec(); }
    void addmult( OutDeriv& result,const InDeriv& data )    { result.getVec() += J*data; }
    void addMultTranspose( InDeriv& result, const OutDeriv& data )    { result += J.multTranspose(data.getVec()); }
    MatBlock getJ()    { return J; }
    KBlock getK(const OutDeriv& /*childForce*/, bool=false) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const SReal& /*kfactor */) {}
};

template<class InReal,class OutReal>
class CustomJacobianBlock< V1(InReal) , F321(OutReal) > :
    public  BaseJacobianBlock< V1(InReal) , F321(OutReal) >
{
public:
    typedef V1(InReal) In;
    typedef F321(OutReal) Out;

    typedef BaseJacobianBlock<In,Out> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::KBlock KBlock;
    typedef typename Inherit::Real Real;

    enum { dim = Out::spatial_dimensions };
    enum { mdim = Out::material_dimensions };
    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,mdim,Real> MaterialToSpatial;

    MatBlock J;
    OutCoord X0;
    static const bool constant=true;

    void init(const MatBlock& _J, const SpatialCoord& /*SPos*/, const MaterialToSpatial& M)    { J=_J; for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<mdim; ++j) X0.getF()[i][j]=M[i][j];  }
    void addapply( OutCoord& result, const InCoord& data )    { result.getVec() +=  J*data + X0.getVec(); }
    void addmult( OutDeriv& result,const InDeriv& data )    { result.getVec() += J*data; }
    void addMultTranspose( InDeriv& result, const OutDeriv& data )    { result += J.multTranspose(data.getVec()); }
    MatBlock getJ()    { return J; }
    KBlock getK(const OutDeriv& /*childForce*/, bool=false) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const SReal& /*kfactor */) {}
};

template<class InReal,class OutReal>
class CustomJacobianBlock< V1(InReal) , F311(OutReal) > :
    public  BaseJacobianBlock< V1(InReal) , F311(OutReal) >
{
public:
    typedef V1(InReal) In;
    typedef F311(OutReal) Out;

    typedef BaseJacobianBlock<In,Out> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::KBlock KBlock;
    typedef typename Inherit::Real Real;

    enum { dim = Out::spatial_dimensions };
    enum { mdim = Out::material_dimensions };
    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,mdim,Real> MaterialToSpatial;

    MatBlock J;
    OutCoord X0;
    static const bool constant=true;

    void init(const MatBlock& _J, const SpatialCoord& /*SPos*/, const MaterialToSpatial& M)    { J=_J; for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<mdim; ++j) X0.getF()[i][j]=M[i][j];  }
    void addapply( OutCoord& result, const InCoord& data )    { result.getVec() +=  J*data + X0.getVec(); }
    void addmult( OutDeriv& result,const InDeriv& data )    { result.getVec() += J*data; }
    void addMultTranspose( InDeriv& result, const OutDeriv& data )    { result += J.multTranspose(data.getVec()); }
    MatBlock getJ()    { return J; }
    KBlock getK(const OutDeriv& /*childForce*/, bool=false) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const SReal& /*kfactor */) {}
};

template<class InReal,class OutReal>
class CustomJacobianBlock< V1(InReal) , F332(OutReal) > :
    public  BaseJacobianBlock< V1(InReal) , F332(OutReal) >
{
public:
    typedef V1(InReal) In;
    typedef F332(OutReal) Out;

    typedef BaseJacobianBlock<In,Out> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::KBlock KBlock;
    typedef typename Inherit::Real Real;

    enum { dim = Out::spatial_dimensions };
    enum { mdim = Out::material_dimensions };
    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,mdim,Real> MaterialToSpatial;

    MatBlock J;
    OutCoord X0;
    static const bool constant=true;

    void init(const MatBlock& _J, const SpatialCoord& /*SPos*/, const MaterialToSpatial& M)    { J=_J; for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<mdim; ++j) X0.getF()[i][j]=M[i][j];  }
    void addapply( OutCoord& result, const InCoord& data )    { result.getVec() +=  J*data + X0.getVec(); }
    void addmult( OutDeriv& result,const InDeriv& data )    { result.getVec() += J*data; }
    void addMultTranspose( InDeriv& result, const OutDeriv& data )    { result += J.multTranspose(data.getVec()); }
    MatBlock getJ()    { return J; }
    KBlock getK(const OutDeriv& /*childForce*/, bool=false) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const SReal& /*kfactor */) {}
};

} // namespace defaulttype
} // namespace sofa



#endif
