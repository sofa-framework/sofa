#ifndef SOFA_FLEXIBLE_IMAGEDENSITYMASS_INL
#define SOFA_FLEXIBLE_IMAGEDENSITYMASS_INL

#include "ImageDensityMass.h"
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/component/mass/AddMToMatrixFunctor.h>


#include <sofa/helper/gl/template.h>

namespace sofa
{

namespace component
{

namespace mass
{

using namespace	sofa::component::topology;
using namespace core::topology;
using namespace sofa::defaulttype;
using namespace sofa::core::behavior;








template <class DataTypes,class ShapeFunctionTypes,class MassTypes>
ImageDensityMass<DataTypes, ShapeFunctionTypes, MassTypes>::~ImageDensityMass()
{
}


///////////////////////////////////////////


template <class DataTypes,class ShapeFunctionTypes,class MassTypes>
void ImageDensityMass<DataTypes, ShapeFunctionTypes, MassTypes>::clear()
{
    f_masses.beginEdit()->clear();
    f_masses.endEdit();
}


template <class DataTypes,class ShapeFunctionTypes,class MassTypes>
void ImageDensityMass<DataTypes, ShapeFunctionTypes, MassTypes>::resize( int vsize )
{
    f_masses.beginEdit()->resize( vsize );
    f_masses.endEdit();
}







//////////////////////////////////



template <class DataTypes,class ShapeFunctionTypes,class MassTypes>
void ImageDensityMass<DataTypes, ShapeFunctionTypes, MassTypes>::init()
{
    Inherited::init();

    if( f_masses.getValue().empty() )
    {
        resize( this->mstate->getX()->size() );
    }

    reinit();
}



template <class DataTypes,class ShapeFunctionTypes,class MassTypes>
void ImageDensityMass<DataTypes, ShapeFunctionTypes, MassTypes>::reinit()
{

    // get the shape function component
    this->getContext()->get( m_shapeFunction, core::objectmodel::BaseContext::Local );
    if( !m_shapeFunction )
    {
        serr << "ShapeFunction<"<<ShapeFunctionTypes::Name()<<"> component not found" << sendl;
        return;
    }

    const VecCoord& DOFX0 = *this->mstate->getX0();


    VecMass& masses = *f_masses.beginEdit();
    for( unsigned i=0 ; i<masses.size() ; ++i )
        masses[i].clear();


    const TransformType& transform = f_transform.getValue();

    // get the density image
    const CImg<double>& densityImage = f_densityImage.getValue().getCImg(0);

    // for each density voxel
    cimg_forXYZ( densityImage, x, y, z )
    {
        // get the voxel density from the image
        double voxelDensity = densityImage( x, y, z );

        if( voxelDensity > 0 )
        {
            // the voxel position in space
            mCoord voxelPos = transform.fromImage( Coord( x, y, z ) );

            // compute interpolation points/weights
            VRef controlPoints;  ///< The cp indices. controlPoints[j] is the index of the j-th parent influencing child.
            VReal weights; ///< The cp weights. weights[j] is the weight of the j-th parent influencing child.
            MaterialToSpatial M; // what is that?!!
            //VGradient gradients;
            //VHessian hessians;
            m_shapeFunction->computeShapeFunction( voxelPos, M, controlPoints, weights/*, gradients, hessians*/ );

            // get the voxel density
            double voxelVolume = transform.getScale()[0] * transform.getScale()[1] * transform.getScale()[2];
            double voxelMass = voxelDensity * voxelVolume;

            // check the real number of control points
            unsigned nbControlPoints = 0;
            for( unsigned k=0; k<controlPoints.size() && weights[k]>0 ; ++k,++nbControlPoints );

            // precompute the interpolation matrix for each control points
            VecLinearJacobianBlock linearJacobians;
            linearJacobians.resize( nbControlPoints );
            for( unsigned k=0; k<nbControlPoints ; k++ )
                linearJacobians[k].init( DOFX0[controlPoints[k]], voxelPos, voxelPos, typename LinearJacobianBlock::MaterialToSpatial(), weights[k], Gradient(), Hessian() );

            // for each control point influencing the voxel
            for( unsigned k=0; k<nbControlPoints ; k++ )
            {
                // influence of the same dof with itself
                addJ1tmJ0( masses[controlPoints[k]], linearJacobians[k], linearJacobians[k], voxelMass );

                // influence of 2 different dofs
                for( unsigned l=k+1; l<controlPoints.size() && weights[l]>0 ; l++ )
                {
                    addJ1tmJ0( masses[controlPoints[k]], linearJacobians[k], linearJacobians[l], voxelMass );
                    addJ1tmJ0( masses[controlPoints[l]], linearJacobians[l], linearJacobians[k], voxelMass );
                }
            }
        }
    }

    f_masses.endEdit();
}



template <class DataTypes,class ShapeFunctionTypes,class MassTypes>
void ImageDensityMass<DataTypes, ShapeFunctionTypes, MassTypes>::addJ1tmJ0( MassTypes& mass, LinearJacobianBlock& J0, LinearJacobianBlock& J1, Real voxelMass )
{
    for( int w=0 ; w<DataTypes::deriv_total_size ; ++w ) // for all cinematic dof
    {
        Deriv m;
        Deriv acc; acc[w] = 1; // create a pseudo acceleration, to compute JtJ line by line
        defaulttype::Vec3Types::Deriv force;

        // map the pseudo acceleration from the dof level to the voxel level
        J0.addmult( force, acc );

        // compute a pseudo-force at voxel level f=ma
        force *= voxelMass;

        // bring back the pseudo-force at dof level
        J1.addMultTranspose( m , force );

        for( int v=0 ; v<DataTypes::deriv_total_size ; ++v ) // for all cinematic dof
            mass[w][v] += m[v];
    }
}





///////////////////////////////////////////












// -- Mass interface
template <class DataTypes,class ShapeFunctionTypes,class MassTypes>
void ImageDensityMass<DataTypes, ShapeFunctionTypes, MassTypes>::addMDx( const core::MechanicalParams* /* PARAMS FIRST */, DataVecDeriv& res, const DataVecDeriv& dx, double factor )
{
    helper::WriteAccessor< DataVecDeriv > _res = res;
    helper::ReadAccessor< DataVecDeriv > _dx = dx;
    const VecMass &masses = f_masses.getValue();
    if( factor == 1.0 )
    {
        for( unsigned int i=0 ; i<_dx.size() ; i++ )
        {
            _res[i] += masses[i] * _dx[i];
        }
    }
    else
    {
        for( unsigned int i=0 ; i<_dx.size() ; i++ )
        {
            _res[i] += masses[i] * _dx[i] * factor;
        }
    }
}

template <class DataTypes,class ShapeFunctionTypes,class MassTypes>
void ImageDensityMass<DataTypes, ShapeFunctionTypes, MassTypes>::accFromF(const core::MechanicalParams* /* PARAMS FIRST */, DataVecDeriv& , const DataVecDeriv&)
{
    serr<<"void ImageDensityMass<DataTypes, ShapeFunctionTypes, MassTypes>::accFromF(VecDeriv& a, const VecDeriv& f) not yet implemented (need the matrix assembly and inversion)"<<sendl;
}

template <class DataTypes,class ShapeFunctionTypes,class MassTypes>
double ImageDensityMass<DataTypes, ShapeFunctionTypes, MassTypes>::getKineticEnergy( const core::MechanicalParams* /* PARAMS FIRST */, const DataVecDeriv& ) const
{
    serr<<"void ImageDensityMass<DataTypes, ShapeFunctionTypes, MassTypes>::getKineticEnergy not yet implemented"<<sendl;
    return 0;
}

template <class DataTypes,class ShapeFunctionTypes,class MassTypes>
double ImageDensityMass<DataTypes, ShapeFunctionTypes, MassTypes>::getPotentialEnergy( const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& ) const
{
    serr<<"void ImageDensityMass<DataTypes, ShapeFunctionTypes, MassTypes>::getPotentialEnergy not yet implemented"<<sendl;
    return 0;
}



template <class DataTypes,class ShapeFunctionTypes,class MassTypes>
void ImageDensityMass<DataTypes, ShapeFunctionTypes, MassTypes>::addGravityToV(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_v)
{
    if(mparams)
    {
        VecDeriv& v = *d_v.beginEdit();

        // gravity
        Vec3d g ( this->getContext()->getGravity() * (mparams->dt()) );
        Deriv theGravity;
        DataTypes::set ( theGravity, g[0], g[1], g[2]);
        Deriv hg = theGravity * (mparams->dt());

        // add weight force
        for (unsigned int i=0; i<v.size(); i++)
        {
            v[i] += hg;
        }
        d_v.endEdit();
    }
}


template <class DataTypes,class ShapeFunctionTypes,class MassTypes>
void ImageDensityMass<DataTypes, ShapeFunctionTypes, MassTypes>::addForce(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& f, const DataVecCoord& /*x*/, const DataVecDeriv& /*v*/)
{
    //if gravity was added separately (in solver's "solve" method), then nothing to do here
    if(this->m_separateGravity.getValue())
        return;

    const VecMass &masses = f_masses.getValue();
    helper::WriteAccessor< DataVecDeriv > _f = f;

    // gravity
    Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set ( theGravity, g[0], g[1], g[2]);


    // add weight
    for (unsigned int i=0; i<masses.size(); i++)
    {
        _f[i] += masses[i]*theGravity;
    }
}

template <class DataTypes,class ShapeFunctionTypes,class MassTypes>
void ImageDensityMass<DataTypes, ShapeFunctionTypes, MassTypes>::addMToMatrix(const core::MechanicalParams *mparams /* PARAMS FIRST */, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    const VecMass &masses = f_masses.getValue();
    const int N = defaulttype::DataTypeInfo<Deriv>::size();
    AddMToMatrixFunctor<Deriv,MassType> calc;
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
    Real mFactor = (Real)mparams->mFactor();
    for (unsigned int i=0; i<masses.size(); i++)
        calc(r.matrix, masses[i], r.offset + N*i, mFactor);
}


///////////////////////

template <class DataTypes,class ShapeFunctionTypes,class MassTypes>
void ImageDensityMass<DataTypes, ShapeFunctionTypes, MassTypes>::draw(const core::visual::VisualParams* /*vparams*/)
{

}



} // namespace mass

} // namespace component

} // namespace sofa

#endif // SOFA_FLEXIBLE_IMAGEDENSITYMASS_INL
