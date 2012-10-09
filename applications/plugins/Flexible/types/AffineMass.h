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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef FLEXIBLE_AffineMASS_H
#define FLEXIBLE_AffineMASS_H

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/component/mass/DiagonalMass.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/rmath.h>
#include <sofa/component/mass/AddMToMatrixFunctor.h>
#include <sofa/component/topology/TopologyData.inl>
#include <sofa/component/topology/PointSetTopologyContainer.h>
#include <sofa/core/behavior/Mass.inl>

#include "../types/AffineTypes.h"

namespace sofa
{

namespace defaulttype
{

using std::endl;
using helper::vector;

/** Mass associated with an affine deformable frame */
template<int _spatial_dimensions,typename _Real>
class AffineMass
{
public:
    typedef _Real Real;
    Real mass;
    // operator to cast to const Real
    operator const Real() const    {        return mass;    }

    typedef Real value_type;

    static const unsigned int spatial_dimensions = _spatial_dimensions;  ///< Number of dimensions the frame is moving in, typically 3
    static const unsigned int VSize = StdAffineTypes<spatial_dimensions,Real>::deriv_total_size;

    typedef Mat<VSize, VSize, Real> MatNN;

    MatNN inertiaMatrix;       // Inertia matrix of the object
    MatNN invInertiaMatrix;    // inverse of inertiaMatrix

    AffineMass ( Real m = 1 )
    {
        mass = m;
        inertiaMatrix.identity();
        invInertiaMatrix.identity();
    }

    void operator= ( Real m )
    {
        mass = m;
        recalc();
    }

    void recalc()
    {
        invInertiaMatrix.invert ( inertiaMatrix );
    }

    inline friend std::ostream& operator << ( std::ostream& out, const AffineMass& m )
    {
        out << m.mass;
        out << " " << m.inertiaMatrix;
        return out;
    }

    inline friend std::istream& operator >> ( std::istream& in, AffineMass& m )
    {
        in >> m.mass;
        in >> m.inertiaMatrix;
        return in;
    }

    void operator *= ( Real fact )
    {
        mass *= fact;
        inertiaMatrix *= fact;
        invInertiaMatrix /= fact;
    }

    void operator /= ( Real fact )
    {
        mass /= fact;
        inertiaMatrix /= fact;
        invInertiaMatrix *= fact;
    }

    static const char* Name();
};


template<int _spatial_dimensions,typename _Real>
inline typename StdAffineTypes<_spatial_dimensions,_Real>::Deriv operator/(const typename StdAffineTypes<_spatial_dimensions,_Real>::Deriv& d, const AffineMass<_spatial_dimensions, _Real>& m)
{
    typename StdAffineTypes<_spatial_dimensions,_Real>::Deriv res;
    res.getVec() = m.invInertiaMatrix * d.getVec();
    return res;
}

template<int _spatial_dimensions,typename _Real>
inline typename StdAffineTypes<_spatial_dimensions,_Real>::Deriv operator*(const typename StdAffineTypes<_spatial_dimensions,_Real>::Deriv& d, const AffineMass<_spatial_dimensions, _Real>& m)
{
    typename StdAffineTypes<_spatial_dimensions,_Real>::Deriv res;
    res.getVec() = m.inertiaMatrix * d.getVec();
    return res;
}

typedef AffineMass<3, double> Affine3dMass;
typedef AffineMass<3, float> Affine3fMass;


#ifdef SOFA_FLOAT
typedef Affine3fMass Affine3Mass;
#else
typedef Affine3dMass Affine3Mass;
#endif



// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES

template<> struct DataTypeName< defaulttype::Affine3fMass > { static const char* name() { return "Affine3fMass"; } };
template<> struct DataTypeName< defaulttype::Affine3dMass > { static const char* name() { return "Affine3dMass"; } };

/// \endcond


} // namespace defaulttype


// ==========================================================================
// Diagonal Mass

namespace component {
namespace mass {



template <class DataTypes, class TMassType>
class FDiagonalMass : public core::behavior::Mass<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(FDiagonalMass,DataTypes,TMassType), SOFA_TEMPLATE(core::behavior::Mass,DataTypes));

    typedef core::behavior::Mass<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef sofa::defaulttype::Vec<3,Real> Vec3;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;
    typedef TMassType MassType;

    typedef helper::vector<TMassType> MassVector;
    typedef sofa::component::topology::PointData<MassVector> VecMass;

    VecMass f_mass;

    FDiagonalMass()
        : f_mass ( initData ( &f_mass,"f_mass","vector of lumped blocks of the mass matrix." ) )
   {
    }

    virtual ~FDiagonalMass() {}

    void clear()
    {
        MassVector& masses = *f_mass.beginEdit();
        masses.clear();
        f_mass.endEdit();
    }

    virtual void init()
    {
      //  Inherited::init();
    }


    virtual void reinit()
    {
       // updateMass();
        Inherited::reinit();
    }

    virtual void bwdInit()
    {

    }

    void addMass(const MassType& mass)
    {
        MassVector& masses = *f_mass.beginEdit();
        masses.push_back (mass);
        f_mass.endEdit();
    }

    void resize(int vsize)
    {
        MassVector& masses = *f_mass.beginEdit();
        masses.resize (vsize);
        f_mass.endEdit();
    }


    // -- Mass interface
    void addMDx(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& f, const DataVecDeriv& dx, double factor)
    {
//        const VecCoord& xfrom = *this->mstate->getX();
//        const MassVector& vecMass0 = f_mass0.getValue();
//        if ( vecMass0.size() != xfrom.size() || frameData->mappingHasChanged) // TODO remove the first condition when mappingHasChanged will be generalized
//            updateMass();

//        VecDeriv resCpy = res;

//        const MassVector& masses = f_mass.getValue();
//        if ( factor == 1.0 )
//        {
//            for ( unsigned int i=0;i<dx.size();i++ )
//            {
//                res[i] += dx[i] * masses[i];
//                if( this->f_printLog.getValue() )
//                    serr<<"FrameDiagonalMass<DataTypes, MassType>::addMDx, res = " << res[i] << sendl;
//            }
//        }
//        else
//        {
//            for ( unsigned int i=0;i<dx.size();i++ )
//            {
//                res[i] += ( dx[i]* masses[i] ) * ( Real ) factor; // damping.getValue() * invSqrDT;
//                if( this->f_printLog.getValue() )
//                    serr<<"FrameDiagonalMass<DataTypes, MassType>::addMDx, res = " << res[i] << sendl;
//            }
//        }
    }


    void accFromF(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& a, const DataVecDeriv& f)
    {
        const MassVector &masses= f_mass.getValue();
            helper::WriteAccessor< DataVecDeriv > _a = a;
            const VecDeriv& _f = f.getValue();

        for (unsigned int i=0;i<masses.size();i++)
        {
            _a[i] = _f[i] / masses[i];
        }
    }

    void addForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v)
    {
//        const VecCoord& xfrom = *this->mstate->getX();
//        const MassVector& vecMass0 = f_mass0.getValue();
//        const MassVector& vecMass = f_mass.getValue();
//        if ( vecMass0.size() != xfrom.size() || frameData->mappingHasChanged)
//            updateMass();

//        rotateMass();

//        //if gravity was added separately (in solver's "solve" method), then nothing to do here
//        if ( this->m_separateGravity.getValue() )
//            return;

//        // gravity
//        Vec3 g ( this->getContext()->getGravity() );
//        Deriv theGravity;
//        DataTypes::set ( theGravity, g[0], g[1], g[2] );

//        // add weight and inertia force
//        const double& invDt = 1./this->getContext()->getDt();
//        for (unsigned int i = 0; i < vecMass.size(); ++i)
//        {
//            Deriv fDamping = - (vecMass[i] * v[i] * damping.getValue() * invDt);
//            f[i] += theGravity*vecMass[i] + fDamping; //  + core::behavior::inertiaForce ( vframe,aframe,masses[i],x[i],v[i] );
//        }
//        if( this->f_printLog.getValue() )
//        {
//            serr << "FrameDiagonalMass<DataTypes, MassType>::addForce" << sendl;
//            serr << "Masse:" << sendl;
//            for(unsigned int i = 0; i < vecMass.size(); ++i)
//                serr << i << ": " << vecMass[i].inertiaMatrix << sendl;
//            serr << "Force_Masse: " << sendl;
//            for(unsigned int i = 0; i < f.size(); ++i)
//                serr << i << ": " << f[i] << sendl;
//        }
    }

    double getKineticEnergy(const core::MechanicalParams* mparams /* PARAMS FIRST */, const DataVecDeriv& v) const  ///< vMv/2 using dof->getV()
    {
        const MassVector& masses = f_mass.getValue();
                helper::ReadAccessor< DataVecDeriv > _v = v;
        double e = 0;
        for ( unsigned int i=0;i<masses.size();i++ )
        {
            e += _v[i]*masses[i]*_v[i]; // v[i]*v[i]*masses[i] would be more efficient but less generic
        }
        return e/2;
    }

    double getPotentialEnergy(const core::MechanicalParams* mparams /* PARAMS FIRST */, const DataVecCoord& x) const   ///< Mgx potential in a uniform gravity field, null at origin
    {
        double e = 0;
        const MassVector& masses = f_mass.getValue();
        // gravity
        Vec3 g ( this->getContext()->getGravity() );
//        VecIn theGravity;
//        theGravity[0]=g[0], theGravity[1]=g[1], theGravity[2]=g[2];
//        for ( unsigned int i=0;i<x.size();i++ )
//        {
//            VecIn translation;
//            translation[0]=(float)x[i].getCenter()[0],  translation[0]=(float)x[1].getCenter()[1], translation[2]=(float)x[i].getCenter()[2];
//            const MatInxIn& m = masses[i].inertiaMatrix;
//            e -= translation * (m * theGravity);
//        }
        return e;
    }

    defaulttype::Vec6d getMomentum(const core::MechanicalParams* mparams /* PARAMS FIRST */, const DataVecCoord& x, const DataVecDeriv& v) const  ///< (Mv,cross(x,Mv)+Iw)
    {
    return defaulttype::Vec6d();
    }

    void addGravityToV(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_v)
    {
        if ( this->mstate.get(mparams) )
        {
//            helper::WriteAccessor< DataVecDeriv > v = *vid[this->mstate.get(mparams)].write();

//            // gravity
//            Vec3 g ( this->getContext()->getGravity() );
//            Deriv theGravity;
//            DataTypes::set ( theGravity, g[0], g[1], g[2] );
//            Deriv hg = theGravity * ( typename DataTypes::Real ) mparams->dt();

//            for ( unsigned int i=0;i<v.size();i++ )
//            {
//                v[i] += hg;
//            }
        }
    }


    /// Add Mass contribution to global Matrix assembling
    // void addMToMatrix(defaulttype::BaseMatrix * mat, double mFact, unsigned int &offset);
    void addMToMatrix(const core::MechanicalParams *mparams /* PARAMS FIRST */, const sofa::core::behavior::MultiMatrixAccessor* matrix)
    {
        const MassVector &masses= f_mass.getValue();
        const int N = defaulttype::DataTypeInfo<Deriv>::size();
        AddMToMatrixFunctor<Deriv,MassType> calc;
            sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
            Real mFactor = (Real)mparams->mFactor();
        for (unsigned int i=0;i<masses.size();i++)
            calc(r.matrix, masses[i], r.offset + N*i, mFactor);
    }

    double getElementMass(unsigned int index) const
    {
    //  return ( SReal ) ( f_mass.getValue() [index] );
   //     cerr<<"WARNING : double FrameDiagonalMass<DataTypes, MassType>::getElementMass ( unsigned int index ) const IS NOT IMPLEMENTED" << endl;
        return 0;
    }

    void getElementMass(unsigned int index, defaulttype::BaseMatrix *m) const
    {
        const unsigned int dimension = defaulttype::DataTypeInfo<Deriv>::size();
        if ( m->rowSize() != dimension || m->colSize() != dimension ) m->resize ( dimension,dimension );

        m->clear();
        AddMToMatrixFunctor<Deriv,MassType>()(m, f_mass.getValue()[index], 0, 1);
    }

    bool isDiagonal(){return true;}

    void draw(const core::visual::VisualParams* /*vparams*/)
    {
    }


    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const FDiagonalMass<DataTypes, TMassType>* = NULL)
    {
        return DataTypes::Name();
    }

//    static std::string templateName(const sofa::core::behavior::ForceField<DataTypes>* = NULL)
//    {
//        std::string name;
//        name.append(DataTypes::Name());
//        name.append(MassType::Name());
//        return name;
//    }
};



#if defined(SOFA_EXTERN_TEMPLATE) && !defined(FLEXIBLE_AffineMASS_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_Flexible_API FDiagonalMass<defaulttype::Affine3dTypes,defaulttype::Affine3dMass>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_Flexible_API FDiagonalMass<defaulttype::Affine3fTypes,defaulttype::Affine3fMass>;
#endif
#endif

} // namespace container
} // namespace component


} // namespace sofa



#endif
