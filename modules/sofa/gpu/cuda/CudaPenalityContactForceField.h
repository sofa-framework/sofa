#ifndef SOFA_GPU_CUDA_CUDAPENALITYCONTACTFORCEFIELD_H
#define SOFA_GPU_CUDA_CUDAPENALITYCONTACTFORCEFIELD_H

#include "CudaTypes.h"
#include <sofa/component/forcefield/PenalityContactForceField.h>
#include <sofa/gpu/cuda/CudaCollisionDetection.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

} // namespace cuda

} // namespace gpu

namespace component
{

namespace forcefield
{

using sofa::gpu::cuda::CudaVec3fTypes;

template<>
class PenalityContactForceField<CudaVec3fTypes> : public core::componentmodel::behavior::PairInteractionForceField<CudaVec3fTypes>
{
public:
    typedef CudaVec3fTypes DataTypes;
    typedef core::componentmodel::behavior::PairInteractionForceField<DataTypes> Inherit;
    typedef DataTypes DataTypes1;
    typedef DataTypes DataTypes2;
    typedef DataTypes::VecCoord VecCoord;
    typedef DataTypes::VecDeriv VecDeriv;
    typedef DataTypes::Coord Coord;
    typedef DataTypes::Deriv Deriv;
    typedef Coord::value_type Real;
    typedef core::componentmodel::behavior::MechanicalState<DataTypes> MechanicalState;

//protected:
    /*
    struct Contact
    {
        int m1, m2;   ///< the two extremities of the spring: masses m1 and m2
        Deriv norm;   ///< contact normal, from m1 to m2
        Real dist;    ///< minimum distance between the points
        Real ks;      ///< spring stiffness
        Real mu_s;    ///< coulomb friction coefficient (currently unused)
        Real mu_v;    ///< viscous friction coefficient
        Real pen;     ///< current penetration
        int age;      ///< how old is this contact
    };
    */
    sofa::gpu::cuda::CudaVector<sofa::defaulttype::Vec4f> contacts;
    sofa::gpu::cuda::CudaVector<float> pen;

    // contacts from previous frame
    //std::vector<Contact> prevContacts;

//public:

    PenalityContactForceField(MechanicalState* object1, MechanicalState* object2)
        : Inherit(object1, object2)
    {
    }

    PenalityContactForceField()
    {
    }

    void clear(int reserve = 0);

    void addContact(int m1, int m2, const Deriv& norm, Real dist, Real ks, Real mu_s = 0.0f, Real mu_v = 0.0f, int oldIndex = 0);

    void setContacts(Real distance, Real ks, sofa::core::componentmodel::collision::GPUDetectionOutputVector* outputs, defaulttype::Mat3x3f* normXForm = NULL);

    virtual void addForce(VecDeriv& f1, VecDeriv& f2, const VecCoord& x1, const VecCoord& x2, const VecDeriv& v1, const VecDeriv& v2);

    virtual void addDForce(VecDeriv& df1, VecDeriv& df2, const VecDeriv& dx1, const VecDeriv& dx2);

    virtual sofa::defaulttype::Vector3::value_type getPotentialEnergy(const VecCoord&, const VecCoord&);

    void draw();
};

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
