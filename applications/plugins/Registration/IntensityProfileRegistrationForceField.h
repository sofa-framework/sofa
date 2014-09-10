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
#ifndef INTENSITYPROFILEREGISTRATIONFORCEFIELD_H
#define INTENSITYPROFILEREGISTRATIONFORCEFIELD_H

#include "initRegistration.h"
#include <ImageTypes.h>

#include <sofa/core/core.h>
#include <sofa/core/behavior/BaseForceField.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/helper/accessor.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/vector.h>
#include <sofa/component/component.h>
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

using helper::vector;

using namespace sofa::defaulttype;




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
    typedef helper::WriteAccessor<Data< ImageTypes > > waImage;
    Data< ImageTypes > refImage;
    Data< ImageTypes > image;

    typedef ImageLPTransform<Real> TransformType;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;
    Data< TransformType > refTransform;
    Data< TransformType > transform;

    Data< VecCoord > refDirections;
    Data< VecCoord > directions;

    Data< ImageTypes > refProfiles;
    Data< ImageTypes > profiles;

    typedef typename defaulttype::Image<Real> similarityTypes;
    typedef typename similarityTypes::T Ts;
    typedef helper::ReadAccessor<Data< similarityTypes > > raSimilarity;
    typedef helper::WriteAccessor<Data< similarityTypes > > waSimilarity;
    Data < similarityTypes > similarity;


    // mask for values outside images
    CImg<bool> refMask;
    CImg<bool> mask;
    CImg<bool> similarityMask;

    Data<bool> useAnisotropicStiffness;

    /*
        The threshold for the signal between two 'edges'
    */
    Data<Real> edgeIntensityThreshold;
    /*
        True if the point should look for a change from high to low 
        signal value in the direction of its normal.
    */
    Data<bool> highToLowSignal;



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
    double getPotentialEnergy(const core::MechanicalParams* ,const DataVecCoord&) const { return m_potentialEnergy; }
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
    double m_potentialEnergy;

    /// compute intensity profile image by sampling the input image along 'direction'.
    /// can be done for the current or reference position/image
    /// Inward and outward profile sizes are defined by data 'Sizes' (+ searchRange, if done on current position)
    void udpateProfiles(bool ref=false);

    /// compute silarity image by convoluing current and reference profiles
    /// the width of the resulting image is 2*searchRange
    void udpateSimilarity();
/*
    Finds the closest change in signal for each point
*/
    void updateThresholdInfo();
/*
    Determines whether a point's signal at a given location is within the threshold range
    @param signal the index of the location in the signal that we're interested in
    @param point the index of the point we are considering
    @return IN_OBJECT if the signal is within the threshold, IN_BACKGROUND if it is not
*/
    bool getSignalLocation(int signal, int point);

    /*
        Keeps the original location of each point at the start of each time step.
    */
    CImg<int> originalLocation;
    /*
        Holds the closest threshold change for each point
    */
    CImg<int> closestThreshold;
    /*
        True if the Threshold Registration algorithm is being used instead of 
        the traditional Intensity Profile Registration.
    */
    bool usingThresholdFinding;

    Data< Vec<2,unsigned int> > Sizes;
    Data< Real > Step;
    Data< helper::OptionsGroup > Interpolation;  ///< nearest, linear, cubi
    Data< helper::OptionsGroup > SimilarityMeasure;  ///< ssd,ncc

    Data< Real > threshold;
    Data< unsigned int > searchRange;
    Data<Real> ks;
    Data<Real> kd;

    Data<float> showArrowSize;
    Data<int> drawMode; //Draw Mode: 0=Line - 1=Cylinder - 2=Arrow
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
