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
#ifndef INTENSITYPROFILEREGISTRATIONFORCEFIELD_H
#define INTENSITYPROFILEREGISTRATIONFORCEFIELD_H

#include <Registration/config.h>
#include <image/ImageTypes.h>

#include <sofa/core/behavior/BaseForceField.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/helper/accessor.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/OptionsGroup.h>


#define INTERPOLATION_NEAREST 0
#define INTERPOLATION_LINEAR 1
#define INTERPOLATION_CUBIC 2

#define SIMILARITY_SSD 0
#define SIMILARITY_NCC 1

/** @name Locations
For Threshold Registration
*/
/**@{*/
#define IN_OBJECT 0
#define IN_BACKGROUND 1
#define IN_MASK 2
/**@}*/


namespace sofa
{

namespace component
{

namespace forcefield
{


template<class _DataTypes,class _ImageTypes>
class IntensityProfileRegistrationForceFieldInternalData
{
public:
};



template <class _DataTypes,class _ImageTypes>
class IntensityProfileRegistrationForceField : public core::behavior::ForceField<_DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(IntensityProfileRegistrationForceField,_DataTypes,_ImageTypes),SOFA_TEMPLATE(core::behavior::ForceField, _DataTypes));

    typedef core::behavior::ForceField<_DataTypes> Inherit;

    typedef _DataTypes DataTypes;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef Data<typename DataTypes::VecCoord> DataVecCoord;
    typedef Data<typename DataTypes::VecDeriv> DataVecDeriv;	

    typedef helper::ReadAccessor< Data< VecCoord > > RDataRefVecCoord;

    typedef core::behavior::MechanicalState<DataTypes> MechanicalState;
    enum { N=DataTypes::spatial_dimensions };
    typedef defaulttype::Mat<N,N,Real> MatNN;

    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;
    typedef helper::WriteOnlyAccessor<Data< ImageTypes > > waImage;
    Data< ImageTypes > refImage;
    Data< ImageTypes > image;

    typedef defaulttype::ImageLPTransform<Real> TransformType;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;
    Data< TransformType > refTransform;
    Data< TransformType > transform;

    Data< VecCoord > refDirections; ///< Profile reference directions.
    Data< VecCoord > directions; ///< Profile directions.

    Data< ImageTypes > refProfiles; ///< reference intensity profiles
    Data< ImageTypes > profiles; ///< computed intensity profiles

    typedef typename defaulttype::Image<Real> similarityTypes;
    typedef typename similarityTypes::T Ts;
    typedef helper::ReadAccessor<Data< similarityTypes > > raSimilarity;
    typedef helper::WriteOnlyAccessor<Data< similarityTypes > > waSimilarity;
    Data < similarityTypes > similarity; ///< similarity image
	
    // mask for values outside images
    Data < bool > maskOutside; ///< discard profiles outside images
    cimg_library::CImg<bool> refMask;
    cimg_library::CImg<bool> mask;
    cimg_library::CImg<bool> similarityMask;

    // use an anisotropic stiffness to cancel tangential stiffness
    Data<bool> useAnisotropicStiffness; ///< use more accurate but non constant stiffness matrix.

	
public:
    IntensityProfileRegistrationForceField(core::behavior::MechanicalState<DataTypes> *mm = NULL);
    virtual ~IntensityProfileRegistrationForceField() {}

    core::behavior::MechanicalState<DataTypes>* getObject() { return this->mstate; }

    static std::string templateName(const IntensityProfileRegistrationForceField<DataTypes,ImageTypes>* = NULL) { return DataTypes::Name()+ std::string(",")+ImageTypes::Name();    }
    virtual std::string getTemplateName() const    { return templateName(this);    }

    // -- ForceField interface
    void reinit();
    void init();
    void addForce(const core::MechanicalParams* /*mparams*/,DataVecDeriv& f , const DataVecCoord& x , const DataVecDeriv& v);
    void addDForce(const core::MechanicalParams* mparams ,DataVecDeriv&   df , const DataVecDeriv&   dx);
    SReal getPotentialEnergy(const core::MechanicalParams* ,const DataVecCoord&) const { return m_potentialEnergy; }
    void addKToMatrix( const core::MechanicalParams* mparams,const sofa::core::behavior::MultiMatrixAccessor* matrix);

    Real getStiffness() const{ return ks.getValue(); }
    Real getDamping() const{ return kd.getValue(); }
    void setStiffness(Real _ks){ ks.setValue(_ks); }
    void setDamping(Real _kd){ kd.setValue(_kd); }
    Real getArrowSize() const{return showArrowSize.getValue();}
    void setArrowSize(float s){showArrowSize.setValue(s);}
    int getDrawMode() const{return drawMode.getValue();}
    void setDrawMode(int m){drawMode.setValue(m);}

    void draw(const core::visual::VisualParams* vparams);

protected :

    sofa::helper::vector<MatNN>  dfdx;
    VecCoord targetPos;
    SReal m_potentialEnergy;

    /// compute intensity profile image by sampling the input image along 'direction'.
    /// can be done for the current or reference position/image
    /// Inward and outward profile sizes are defined by data 'Sizes' (+ searchRange, if done on current position)
    void udpateProfiles(bool ref=false);

    /// compute silarity image by convoluing current and reference profiles
    /// the width of the resulting image is 2*searchRange
    void udpateSimilarity();

    Data< defaulttype::Vec<2,unsigned int> > Sizes; ///< Inwards/outwards profile size.
    Data< Real > Step; ///< Spacing of the profile discretization.
    Data< helper::OptionsGroup > Interpolation;  ///< nearest, linear, cubi
    Data< helper::OptionsGroup > SimilarityMeasure;  ///< ssd,ncc

    Data< Real > threshold; ///< threshold for the distance minimization.
    Data< unsigned int > searchRange; ///< Number of inwards/outwards steps for searching the most similar profiles.
    Data<Real> ks; ///< uniform stiffness for the all springs
    Data<Real> kd; ///< uniform damping for the all springs

    Data<float> showArrowSize; ///< size of the axis
    Data<int> drawMode; ///< Draw Mode: 0=Line - 1=Cylinder - 2=Arrow
};

//#if defined(SOFA_EXTERN_TEMPLATE) && !defined(INTENSITYPROFILEREGISTRATIONFORCEFIELD_CPP)
//#ifndef SOFA_FLOAT
//extern template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3dTypes>;
//#endif
//#ifndef SOFA_DOUBLE
//extern template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3fTypes>;
//#endif
//#endif

} //

} //

} // namespace sofa

#endif
