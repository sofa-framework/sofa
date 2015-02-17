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
#ifndef SOFA_COMPONENT_MAPPING_RIGIDMAPPING_INL
#define SOFA_COMPONENT_MAPPING_RIGIDMAPPING_INL

#include <SofaRigid/RigidMapping.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/Mapping.inl>
#include <sofa/core/State.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/helper/io/MassSpringLoader.h>
#include <sofa/helper/io/SphereLoader.h>
#include <sofa/helper/io/Mesh.h>
#include <sofa/helper/gl/template.h>

#include <sofa/simulation/common/Simulation.h>

#include <string.h>
#include <iostream>
#include <cassert>
#include <numeric>

namespace sofa
{

namespace component
{

namespace mapping
{

extern void rigidMappingDummyFunction(); ///< Used for setting breakpoints, since gdb sometimes fails at breaking within template methods. Implemented in RigidMapping.C


template <class TIn, class TOut>
class RigidMapping<TIn, TOut>::Loader : public helper::io::MassSpringLoader,
        public helper::io::SphereLoader
{
public:

    RigidMapping<TIn, TOut>* dest;
    helper::WriteAccessor<Data<VecCoord> > points;

    Loader(RigidMapping<TIn, TOut>* dest) :
        dest(dest),
        points(dest->points)
    {
    }
    virtual void addMass(SReal px, SReal py, SReal pz, SReal, SReal, SReal,
                         SReal, SReal, bool, bool)
    {
        Coord c;
        Out::set(c, px, py, pz);
        points.push_back(c); //Coord((Real)px,(Real)py,(Real)pz));
    }
    virtual void addSphere(SReal px, SReal py, SReal pz, SReal)
    {
        Coord c;
        Out::set(c, px, py, pz);
        points.push_back(c); //Coord((Real)px,(Real)py,(Real)pz));
    }
};

template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::load(const char *filename)
{
    points.beginEdit()->resize(0);
    points.endEdit();

    if (strlen(filename) > 4
            && !strcmp(filename + strlen(filename) - 4, ".xs3"))
    {
        Loader loader(this);
        loader.helper::io::MassSpringLoader::load(filename);
    }
    else if (strlen(filename) > 4
             && !strcmp(filename + strlen(filename) - 4, ".sph"))
    {
        Loader loader(this);
        loader.helper::io::SphereLoader::load(filename);
    }
    else if (strlen(filename) > 0)
    {
        // Default to mesh loader
        helper::io::Mesh* mesh = helper::io::Mesh::Create(filename);
        if (mesh != NULL)
        {
            helper::WriteAccessor<Data<VecCoord> > points = this->points;

            points.resize(mesh->getVertices().size());
            for (unsigned int i = 0; i < mesh->getVertices().size(); i++)
            {
                Out::set(points[i],
                         mesh->getVertices()[i][0],
                         mesh->getVertices()[i][1],
                         mesh->getVertices()[i][2]);
            }
            delete mesh;
        }
    }
}


template <class TIn, class TOut>
RigidMapping<TIn, TOut>::RigidMapping()
    : Inherit()
    , points(initData(&points, "initialPoints", "Local Coordinates of the points"))
    , index(initData(&index, (unsigned)0, "index", "input DOF index"))
    , fileRigidMapping(initData(&fileRigidMapping, "fileRigidMapping", "Filename"))
    , useX0(initData(&useX0, false, "useX0", "Use x0 instead of local copy of initial positions (to support topo changes)"))
    , indexFromEnd(initData(&indexFromEnd, false, "indexFromEnd", "input DOF index starts from the end of input DOFs vector"))
    , pointsPerFrame(initData(&pointsPerFrame, "repartition", "number of dest dofs per entry dof"))
    , globalToLocalCoords(initData(&globalToLocalCoords, "globalToLocalCoords", "are the output DOFs initially expressed in global coordinates"))
    , maskFrom(NULL)
    , maskTo(NULL)
    , matrixJ()
    , updateJ(false)
{
    //std::cout << "RigidMapping Creation\n";
    this->addAlias(&fileRigidMapping, "filename");


}

template <class TIn, class TOut>
int RigidMapping<TIn, TOut>::addPoint(const Coord& c)
{
    helper::WriteAccessor<Data<VecCoord> > points = this->points;
    int i = points.size();
    points.push_back(c);
    return i;
}

template <class TIn, class TOut>
int RigidMapping<TIn, TOut>::addPoint(const Coord& c, int indexFrom)
{
    // REALLY INEFFICIENT... used for collision detection
    // could be improved by using a parentIndexPerPoint (size of points)
    // rather than pointsPerFrame that imposes points to be ordered with the parents


    VecCoord& points = *this->points.beginEdit();


    int i = points.size();
//    if (!pointsPerFrame.getValue().empty())
//    {
//        helper::vector<unsigned int>& ppf = *pointsPerFrame.beginEdit();
//        assert( ppf.size() == this->fromModel->getSize() );

//        // parcourir pointsPerFrame jusqu'à indexFrom pour sommer index courant
//        unsigned int index = std::accumulate( ppf.begin(), ppf.begin()+indexFrom ,0 );

//        // inserer c au milieu à index courant dans points
//        // points.insert( points.begin()+index, c ); // not implemented on extvector that is not based on std::vector...
//        points.resize( points.size() + 1 );
//        for( size_t j=points.size()-1; j>index ; ++j )
//            points[j] = points[j-1];
//        points[index] = c;

//        ppf[indexFrom]++;

//        pointsPerFrame.endEdit();
//    }
//    else if (!i)
//    {
//        points.push_back(c);
//        index.setValue(indexFrom);
//    }
//    else if ((int) index.getValue() != indexFrom)
//    {
//        if(  indexFrom > index.getValue() )
//            points.push_back(c);
//        else
//        {
//            // if indexFrom < index.getValue() it comes first...
//            points.resize( points.size() + 1 );
//            for( size_t j=points.size()-1; j>0 ; ++j )
//                points[j] = points[j-1];
//            points[0] = indexFrom;
//        }

//        helper::vector<unsigned int>& ppf = *pointsPerFrame.beginEdit();
//        ppf.resize( this->fromModel->getSize() );
//        std::fill( ppf.begin(), ppf.end(), 0 );
//        ppf[index.getValue()] = i;
//        ppf[indexFrom] = 1;
//        pointsPerFrame.endEdit();
//    }

//    this->points.endEdit();

    return i;
}


template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::reinit()
{
    if (this->points.getValue().empty() && this->toModel != NULL && !useX0.getValue())
    {
        //        cerr<<"RigidMapping<TIn, TOut>::init(), from " << this->fromModel->getName() << " to " << this->toModel->getName() << endl;
        const VecCoord& xTo =this->toModel->read(core::ConstVecCoordId::position())->getValue();
        helper::WriteAccessor<Data<VecCoord> > points = this->points;
        points.resize(xTo.size());
        unsigned int i = 0;
        if (globalToLocalCoords.getValue())
        {
            //            cerr<<"globalToLocal is true, compute local coordinates"  << endl;
            const VecCoord& xTo =this->toModel->read(core::ConstVecCoordId::position())->getValue();
            points.resize(xTo.size());
            unsigned int i = 0, cpt = 0;
            const InVecCoord& xFrom =this->fromModel->read(core::ConstVecCoordId::position())->getValue();
            switch (pointsPerFrame.getValue().size())
            {
            case 0:
                for (i = 0; i < xTo.size(); i++)
                {
                    points[i] = xFrom[0].inverseRotate(xTo[i]- xFrom[0].getCenter());
                }
                break;
            case 1:
                for (i = 0; i < xFrom.size(); i++)
                {
                    for (unsigned int j = 0; j < pointsPerFrame.getValue()[0]; j++, cpt++)
                    {
                        points[cpt]
                                = xFrom[i].inverseRotate(xTo[cpt] - xFrom[i].getCenter());
                    }
                }
                break;
            default:
                for (i = 0; i < xFrom.size(); i++)
                {
                    for (unsigned int j = 0; j < pointsPerFrame.getValue()[i]; j++, cpt++)
                    {
                        points[cpt]
                                = xFrom[i].inverseRotate(xTo[cpt] - xFrom[i].getCenter());
                    }
                }
                break;

            }
        }
        else
        {
            for (i = 0; i < xTo.size(); i++)
            {
                points[i] = xTo[i];
            }
            //            cerr<<"globalToLocal is false, points in local coordinates : " << points << endl;
        }
    }
    else
    {
        //        cerr << "RigidMapping<TIn, TOut>::init(), points not empty or toModel is null or useX0" << endl;
    }
}

template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::init()
{
    //    rigidMappingDummyFunction();
    if (core::behavior::BaseMechanicalState* stateFrom = dynamic_cast<core::behavior::BaseMechanicalState*>(this->fromModel.get()))
    {
        maskFrom = &stateFrom->forceMask;
    }
    if (core::behavior::BaseMechanicalState* stateTo = dynamic_cast<core::behavior::BaseMechanicalState*>(this->toModel.get()))
    {
        maskTo = &stateTo->forceMask;
    }

    if (!fileRigidMapping.getValue().empty())
        this->load(fileRigidMapping.getFullPath().c_str());


#ifdef SOFA_HAVE_EIGEN2
    eigenJacobians.resize( 1 );
    eigenJacobians[0] = &eigenJacobian;
#endif

    this->reinit();

    this->Inherit::init();
}

/*
template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::disable()
{
 if (!this->points.getValue().empty() && this->toModel!=NULL)
 {
  VecCoord& x =this->toModel->read(core::ConstVecCoordId::position())->getValue();
  x.resize(points.getValue().size());
  for (unsigned int i=0;i<points.getValue().size();i++)
   x[i] = points.getValue()[i];
 }
}
*/

template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::clear(int reserve)
{
    helper::WriteAccessor<Data<VecCoord> > points = this->points;
    points.clear();
    if (reserve)
        points.reserve(reserve);
    this->pointsPerFrame.beginEdit()->clear();
    this->pointsPerFrame.endEdit();
}

template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::setRepartition(unsigned int value)
{
    vector<unsigned int>& rep = *this->pointsPerFrame.beginEdit();
    rep.clear();
    rep.push_back(value);
    this->pointsPerFrame.endEdit();
}

template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::setRepartition(sofa::helper::vector<
                                             unsigned int> values)
{
    vector<unsigned int>& rep = *this->pointsPerFrame.beginEdit();
    rep.clear();
    rep.reserve(values.size());
    //repartition.setValue(values);
    sofa::helper::vector<unsigned int>::iterator it = values.begin();
    while (it != values.end())
    {
        rep.push_back(*it);
        it++;
    }
    this->pointsPerFrame.endEdit();
}

template <class TIn, class TOut>
const typename RigidMapping<TIn, TOut>::VecCoord & RigidMapping<TIn, TOut>::getPoints()
{
    if (useX0.getValue())
    {
        const Data<VecCoord>* v = this->toModel.get()->read(core::VecCoordId::restPosition());
        if (v)
            return v->getValue();
        else
            serr
                    << "RigidMapping: ERROR useX0 can only be used in MechanicalMappings."
                    << sendl;
    }
    return points.getValue();
}

template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::apply(const core::MechanicalParams * /*mparams*/ /* PARAMS FIRST */, Data<VecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    helper::WriteAccessor< Data<VecCoord> > out = dOut;
    helper::ReadAccessor< Data<InVecCoord> > in = dIn;
    const VecCoord& pts = this->getPoints();

    updateJ = true;
#ifdef SOFA_HAVE_EIGEN2
    eigenJacobian.resizeBlocks(out.size(),in.size());
#endif

    rotatedPoints.resize(pts.size());
    out.resize(pts.size());

    unsigned repartitionCount = pointsPerFrame.getValue().size();

    if (repartitionCount > 1 && repartitionCount != in.size())
    {
        serr << "apply Error : mapping dofs repartition is not correct " << sendl;
        return;
    }

    unsigned inIdxBegin;
    unsigned inIdxEnd;

    if (repartitionCount == 0)
    {
        inIdxBegin = index.getValue();
        if (indexFromEnd.getValue())
        {
            inIdxBegin = in.size() - 1 - inIdxBegin;
        }
        inIdxEnd = inIdxBegin + 1;
    }
    else
    {
        inIdxBegin = 0;
        inIdxEnd = in.size();
    }

    unsigned outputPerInput;
    if (repartitionCount == 0)
    {
        outputPerInput = pts.size();
    }
    else
    {
        outputPerInput = pointsPerFrame.getValue()[0];
    }

    Coord translation;
    Mat rotation;

    for (unsigned inIdx = inIdxBegin, outIdx = 0; inIdx < inIdxEnd; ++inIdx)
    {
        if (repartitionCount > 1)
        {
            outputPerInput = pointsPerFrame.getValue()[inIdx];
        }

        translation = in[inIdx].getCenter();
        in[inIdx].writeRotationMatrix(rotation);

        for (unsigned iOutput = 0;
             iOutput < outputPerInput;
             ++iOutput, ++outIdx)
        {
            rotatedPoints[outIdx] = rotation * pts[outIdx];
            out[outIdx] = rotatedPoints[outIdx];
            out[outIdx] += translation;
        }
    }

    //    cerr<<"RigidMapping<TIn, TOut>::apply, " << this->getName() << endl;
    //    cerr<<"RigidMapping<TIn, TOut>::apply, in = " << dIn << endl;
    //    cerr<<"RigidMapping<TIn, TOut>::apply, points = " << pts << endl;
    //    cerr<<"RigidMapping<TIn, TOut>::apply, out = " << dOut << endl;
}

template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::applyJ(const core::MechanicalParams * /*mparams*/ /* PARAMS FIRST */, Data<VecDeriv>& dOut, const Data<InVecDeriv>& dIn)
{
    helper::WriteAccessor< Data<VecDeriv> > out = dOut;
    helper::ReadAccessor< Data<InVecDeriv> > in = dIn;

    const VecCoord& pts = this->getPoints();
    out.resize(pts.size());

    bool isMaskInUse = maskTo && maskTo->isInUse();
    unsigned repartitionCount = pointsPerFrame.getValue().size();

    if (repartitionCount > 1 && repartitionCount != in.size())
    {
        serr << "applyJ Error : mapping dofs repartition is not correct" << sendl;
        return;
    }

    unsigned inIdxBegin;
    unsigned inIdxEnd;

    if (repartitionCount == 0)
    {
        inIdxBegin = index.getValue();
        if (indexFromEnd.getValue())
        {
            inIdxBegin = in.size() - 1 - inIdxBegin;
        }
        inIdxEnd = inIdxBegin + 1;
    }
    else
    {
        inIdxBegin = 0;
        inIdxEnd = in.size();
    }

    unsigned outputPerInput;
    if (repartitionCount == 0)
    {
        outputPerInput = pts.size();
    }
    else
    {
        outputPerInput = pointsPerFrame.getValue()[0];
    }

    typedef helper::ParticleMask ParticleMask;
    ParticleMask::InternalStorage* indices = isMaskInUse ? &maskTo->getEntries() : NULL;
    ParticleMask::InternalStorage::const_iterator it;
    if (isMaskInUse) it = indices->begin();

    for (unsigned inIdx = inIdxBegin, outIdx = 0; inIdx < inIdxEnd; ++inIdx)
    {
        if (repartitionCount > 1)
        {
            outputPerInput = pointsPerFrame.getValue()[inIdx];
        }

        for (unsigned iOutput = 0;
             iOutput < outputPerInput && !(isMaskInUse && it == indices->end());
             ++iOutput, ++outIdx)
        {
            if (isMaskInUse)
            {
                if (outIdx != *it)
                {
                    continue;
                }
                ++it;
            }
            out[outIdx] = velocityAtRotatedPoint( in[inIdx], rotatedPoints[outIdx] );
        }
    }
}

template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::applyJT(const core::MechanicalParams * /*mparams*/ /* PARAMS FIRST */, Data<InVecDeriv>& dOut, const Data<VecDeriv>& dIn)
{
    helper::WriteAccessor< Data<InVecDeriv> > out = dOut;
    helper::ReadAccessor< Data<VecDeriv> > in = dIn;

    const VecCoord& pts = this->getPoints();

    bool isMaskInUse = maskTo && maskTo->isInUse();
    if (maskFrom) maskFrom->setInUse(isMaskInUse);

    unsigned repartitionCount = pointsPerFrame.getValue().size();

    if (repartitionCount > 1 && repartitionCount != out.size())
    {
        serr << "applyJT, Error : mapping dofs repartition is not correct" << sendl;
        return;
    }

    unsigned outIdxBegin;
    unsigned outIdxEnd;
    unsigned inputPerOutput;

    if (repartitionCount == 0)
    {
        outIdxBegin = index.getValue();
        if (indexFromEnd.getValue())
        {
            outIdxBegin = out.size() - 1 - outIdxBegin;
        }
        outIdxEnd = outIdxBegin + 1;
        inputPerOutput = pts.size();
    }
    else
    {
        outIdxBegin = 0;
        outIdxEnd = out.size();
        inputPerOutput = pointsPerFrame.getValue()[0];
    }


    typedef helper::ParticleMask ParticleMask;
    ParticleMask::InternalStorage* indices = isMaskInUse ? &maskTo->getEntries() : NULL;
    ParticleMask::InternalStorage::const_iterator it;
    if (isMaskInUse) it = indices->begin();

    for (unsigned outIdx = outIdxBegin, inIdx = 0; outIdx < outIdxEnd; ++outIdx)
    {
        if (repartitionCount > 1)
        {
            inputPerOutput = pointsPerFrame.getValue()[outIdx];
        }

        for (unsigned iInput = 0;
             iInput < inputPerOutput && !(isMaskInUse && it == indices->end());
             ++iInput, ++inIdx)
        {
            if (isMaskInUse)
            {
                if (inIdx != *it)
                {
                    continue;
                }
                ++it;
            }
            getVCenter(out[outIdx]) += in[inIdx];
            getVOrientation(out[outIdx]) +=  (typename InDeriv::Rot)cross(rotatedPoints[inIdx], in[inIdx]);
            //                        cerr<<"RigidMapping<TIn, TOut>::applyJT, inIdx = "<< inIdx << endl;
            //                        cerr<<"RigidMapping<TIn, TOut>::applyJT, in[inIdx] = "<< in[inIdx] << endl;
            //                        cerr<<"RigidMapping<TIn, TOut>::applyJT, rotatedPoint[inIdx] = "<< rotatedPoints[inIdx] << endl;
            //                        cerr<<"RigidMapping<TIn, TOut>::applyJT, cross(rotatedPoints[inIdx], in[inIdx]) = "<< cross(rotatedPoints[inIdx], in[inIdx]) << endl;
            //                        cerr<<"RigidMapping<TIn, TOut>::applyJT, force(out[outIdx]) = "<< getVCenter(out[outIdx]) << endl;
            //                        cerr<<"RigidMapping<TIn, TOut>::applyJT, torque(out[outIdx]) = "<< getVOrientation(out[outIdx]) << endl;

        }
        if (isMaskInUse)
        {
            maskFrom->insertEntry(outIdx);
        }
    }

}

//            using defaulttype::Vec;
//
//            /** Symmetric cross cross product.
//              Let [a×(.×c)] be the linear operator such that: a×(b×c) = [a×(.×c)]b, where × denotes the cross product.
//              This operator is not symmetric, and can mess up conjugate gradient solutions.
//              This method computes sym([a×(.×c)])b , where sym(M) = (M+M^T)/2
//              */
//            template<class Rp, class Rc>  // p for parent, c for child
//            Vec<3,Rp> symCrossCross( const Vec<3,Rc>& a,  const Vec<3,Rp>& b,  const Vec<3,Rc>& c  )
//            {
////                Rp m00 = a[1]*c[1]+a[2]*c[2], m01= -0.5*(a[1]*c[0]+a[0]*c[1]), m02 = -0.5*(a[2]*c[0]+a[0]*c[2]) ;
////                Rp                            m11=  a[0]*c[0]+a[2]*c[2],       m12 = -0.5*(a[2]*c[1]+a[1]*c[2]) ;
////                Rp                                                             m22=  a[0]*c[0]+a[1]*c[1];
//                Rp m00 = a[1]*c[1]+a[2]*c[2], m01= 0, m02 = 0 ;
//                Rp                            m11=  a[0]*c[0]+a[2]*c[2],       m12 = 0 ;
//                Rp                                                             m22=  a[0]*c[0]+a[1]*c[1];
//                return Vec<3,Rp>(
//                        m00*b[0] + m01*b[1] + m02*b[2],
//                        m01*b[0] + m11*b[1] + m12*b[2],
//                        m02*b[0] + m12*b[1] + m22*b[2]
//                        );
//            }
//
//            /** Symmetric cross cross product in 2D (see doc in 3D)
//              In 2D, this operator is a scalar so it is symmetric.
//              */
//            template<class Rp, class Rc> // p for parent, c for child
//            Rp symCrossCross( const Vec<2,Rc>& a,  const Rp& b,  const Vec<2,Rc>& c  )
//            {
//                return (a*c)*b;
//            }


template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::applyDJT(const core::MechanicalParams* mparams /* PARAMS FIRST */, core::MultiVecDerivId parentForceChangeId, core::ConstMultiVecDerivId )
{
    if( mparams->symmetricMatrix() )
        return;  // This method corresponds to a non-symmetric matrix, due to the non-commutativity of the group of rotations.

    helper::ReadAccessor<Data<VecDeriv> > childForces (*mparams->readF(this->toModel));
    helper::WriteAccessor<Data<InVecDeriv> > parentForces (*parentForceChangeId[this->fromModel.get(mparams)].write());
    helper::ReadAccessor<Data<InVecDeriv> > parentDisplacements (*mparams->readDx(this->fromModel));
    //    cerr<<"RigidMapping<TIn, TOut>::applyDJT, parent displacements = "<< parentDisplacements << endl;
    //    cerr<<"RigidMapping<TIn, TOut>::applyDJT, parent forces = "<< parentForces << endl;

    InReal kfactor = (InReal)mparams->kFactor();
    //    cerr<<"RigidMapping<TIn, TOut>::applyDJT, kfactor = "<< kfactor << endl;

    const VecCoord& pts = this->getPoints();

    bool isMaskInUse = maskTo && maskTo->isInUse();
    if (maskFrom) maskFrom->setInUse(isMaskInUse);

    unsigned repartitionCount = pointsPerFrame.getValue().size();

    if (repartitionCount > 1 && repartitionCount != parentForces.size())
    {
        serr << "applyDJT Error : mapping dofs repartition is not correct" << sendl;
        return;
    }


    unsigned parentIdxBegin;
    unsigned parentIdxEnd;
    unsigned inputPerOutput;

    if (repartitionCount == 0)
    {
        parentIdxBegin = index.getValue();
        if (indexFromEnd.getValue())
        {
            parentIdxBegin = parentForces.size() - 1 - parentIdxBegin;
        }
        parentIdxEnd = parentIdxBegin + 1;
        inputPerOutput = pts.size();
    }
    else
    {
        parentIdxBegin = 0;
        parentIdxEnd = parentForces.size();
        inputPerOutput = pointsPerFrame.getValue()[0];
    }


    typedef helper::ParticleMask ParticleMask;
    ParticleMask::InternalStorage* indices = isMaskInUse ? &maskTo->getEntries() : NULL;
    ParticleMask::InternalStorage::const_iterator it;
    if (isMaskInUse) it = indices->begin();

    for (unsigned parentIdx = parentIdxBegin, childIdx = 0; parentIdx < parentIdxEnd; ++parentIdx)
    {
        if (repartitionCount > 1)
        {
            inputPerOutput = pointsPerFrame.getValue()[parentIdx];
        }

        for (unsigned iInput = 0;
             iInput < inputPerOutput && !(isMaskInUse && it == indices->end());
             ++iInput, ++childIdx)
        {
            if (isMaskInUse)
            {
                if (childIdx != *it)
                {
                    continue;
                }
                ++it;
            }
            typename TIn::AngularVector& parentTorque = getVOrientation(parentForces[parentIdx]);
            const typename TIn::AngularVector& parentRotation = getVOrientation(parentDisplacements[parentIdx]);
            //                        const typename TIn::AngularVector& torqueDecrement = symCrossCross( childForces[childIdx], parentRotation, rotatedPoints[childIdx]) * kfactor;
            const typename TIn::AngularVector& torqueDecrement = TIn::crosscross( childForces[childIdx], parentRotation, rotatedPoints[childIdx]) * kfactor;
            parentTorque -=  torqueDecrement;
//            cerr<<"RigidMapping<TIn, TOut>::applyDJT, childForces[childIdx] = "<< childForces[childIdx] << endl;
//            cerr<<"RigidMapping<TIn, TOut>::applyDJT, parentRotation = "<< parentRotation << endl;
//            cerr<<"RigidMapping<TIn, TOut>::applyDJT, rotatedPoints[childIdx] = "<< rotatedPoints[childIdx] << endl;
//            cerr<<"RigidMapping<TIn, TOut>::applyDJT,  kfactor = "<<  kfactor << endl;
//            cerr<<"RigidMapping<TIn, TOut>::applyDJT, parentTorque increment = "<< -torqueDecrement << endl;
//            cerr<<"RigidMapping<TIn, TOut>::applyDJT, parentTorque = "<< parentTorque << endl;




        }
        if (isMaskInUse)
        {
            maskFrom->insertEntry(parentIdx);
        }
    }
}


// RigidMapping::applyJT( InMatrixDeriv& out, const OutMatrixDeriv& in ) //
// this function propagate the constraint through the rigid mapping :
// if one constraint along (vector n) with a value (v) is applied on the childModel (like collision model)
// then this constraint is transformed by (Jt.n) with value (v) for the rigid model
// There is a specificity of this propagateConstraint: we have to find the application point on the childModel
// in order to compute the right constaint on the rigidModel.
template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::applyJT(const core::ConstraintParams * /*cparams*/ /* PARAMS FIRST */, Data<InMatrixDeriv>& dOut, const Data<OutMatrixDeriv>& dIn)
{
    InMatrixDeriv& out = *dOut.beginEdit();
    const OutMatrixDeriv& in = dIn.getValue();

    if (this->f_printLog.getValue())
    {
        sout << "J on mapped DOFs == " << in << sendl;
        sout << "J on input  DOFs == " << out << sendl;
    }

    switch (pointsPerFrame.getValue().size())
    {
    case 0:
    {
        typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

        for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
        {
            DPos v;
            DRot omega = DRot();

            typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();

            for (typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                const Deriv f = colIt.val();
                v += f;
                omega += (DRot) cross(rotatedPoints[colIt.index()], f);
            }

            const InDeriv result(v, omega);
            typename InMatrixDeriv::RowIterator o = out.writeLine(rowIt.index());

            if (!indexFromEnd.getValue())
            {
                o.addCol(index.getValue(), result);
            }
            else
            {
                // Commented by PJ. Bug??
                // o.addCol(out.size() - 1 - index.getValue(), result);
                const unsigned int numDofs = this->getFromModel()->read(core::ConstVecCoordId::position())->getValue().size();
                o.addCol(numDofs - 1 - index.getValue(), result);
            }
        }

        break;
    }
    case 1:
    {
        const unsigned int numDofs = this->getFromModel()->read(core::ConstVecCoordId::position())->getValue().size();
        const unsigned int val = pointsPerFrame.getValue()[0];

        typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

        for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
        {
            unsigned int cpt = 0;

            typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();
            typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();

            for (unsigned int ito = 0; ito < numDofs; ito++)
            {
                DPos v;
                DRot omega = DRot();
                bool needToInsert = false;

                for (unsigned int r = 0; r < val && colIt != colItEnd; r++, cpt++)
                {
                    if (colIt.index() != cpt)
                        continue;

                    needToInsert = true;
                    const Deriv f = colIt.val();
                    v += f;
                    omega += (DRot) cross(rotatedPoints[cpt], f);

                    ++colIt;
                }

                if (needToInsert)
                {
                    const InDeriv result(v, omega);

                    typename InMatrixDeriv::RowIterator o = out.writeLine(rowIt.index());
                    o.addCol(ito, result);
                }
            }
        }

        break;
    }
    default:
    {
        const unsigned int numDofs = this->getFromModel()->read(core::ConstVecCoordId::position())->getValue().size();

        typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

        for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
        {
            unsigned int cpt = 0;

            typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();
            typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();

            for (unsigned int ito = 0; ito < numDofs; ito++)
            {
                DPos v;
                DRot omega = DRot();
                bool needToInsert = false;

                for (unsigned int r = 0; r < pointsPerFrame.getValue()[ito] && colIt
                     != colItEnd; r++, cpt++)
                {
                    if (colIt.index() != cpt)
                        continue;

                    needToInsert = true;
                    const Deriv f = colIt.val();
                    v += f;
                    omega += (DRot) cross(rotatedPoints[cpt], f);

                    ++colIt;
                }

                if (needToInsert)
                {
                    const InDeriv result(v, omega);

                    typename InMatrixDeriv::RowIterator o = out.writeLine(rowIt.index());
                    o.addCol(ito, result);
                }
            }
        }

        break;
    }
    }

    if (this->f_printLog.getValue())
    {
        sout << "new J on input  DOFs = " << out << sendl;
    }

    dOut.endEdit();
}


namespace impl {

template<class U, class Coord>
static void fill_block(Eigen::Matrix<U, 3, 6>& block, const Coord& v) {
	U x = v[0];
	U y = v[1];
	U z = v[2];
				
	// note: this is -hat(v)
	block.template rightCols<3>() <<
					
		0,   z,  -y,
		-z,  0,   x,
		y,  -x,   0;
}

template<class U, class Coord>
void fill_block(Eigen::Matrix<U, 2, 3>& block, const Coord& v) {
	U x = v[0];
	U y = v[1];
				
	// note: this is -hat(v)
	block.template rightCols<1>() <<
		-y,
		x;
}


}

#ifdef SOFA_HAVE_EIGEN2
template <class TIn, class TOut>
const helper::vector<sofa::defaulttype::BaseMatrix*>* RigidMapping<TIn, TOut>::getJs()
{
	const VecCoord& out =this->toModel->read(core::ConstVecCoordId::position())->getValue();
    const InVecCoord& in =this->fromModel->read(core::ConstVecCoordId::position())->getValue();
    const VecCoord& pts = this->getPoints();

	typename SparseMatrixEigen::CompressedMatrix& J = eigenJacobian.compressedMatrix;
	
	if( updateJ || J.size() == 0 ) {
		
		J.resize(out.size() * NOut, in.size() * NIn);
		J.setZero();

		// delicious copypasta... why do we have to deal with all this
		// crap *inside* the mapping in the first place? ideally, the
		// mapping should only have a (index, local_coords) list,
		// setup from the outside.
		unsigned repartitionCount = pointsPerFrame.getValue().size();

        if (repartitionCount > 1 && repartitionCount != in.size())
        {
            serr << "getJs Error : mapping dofs repartition is not correct" << sendl;
            return 0;
        }

        unsigned inIdxBegin;
        unsigned inIdxEnd;

        if (repartitionCount == 0)
        {
            inIdxBegin = index.getValue();
            if (indexFromEnd.getValue())
            {
                inIdxBegin = in.size() - 1 - inIdxBegin;
            }
            inIdxEnd = inIdxBegin + 1;
        }
        else
        {
            inIdxBegin = 0;
            inIdxEnd = in.size();
        }

        unsigned outputPerInput;
        if (repartitionCount == 0)
        {
            outputPerInput = pts.size();
        }
        else
        {
            outputPerInput = pointsPerFrame.getValue()[0];
        }

		// matrix chunk
		typedef typename TOut::Real real;
		typedef Eigen::Matrix<real, NOut, NIn> block_type;
		block_type block;
		
		// translation part
		block.template leftCols<NOut>().setIdentity();
		
		// col indices are strictly increasing
		for (unsigned inIdx = inIdxBegin, outIdx = 0; inIdx < inIdxEnd; ++inIdx) {
            if (repartitionCount > 1) {
				
				// max: wtf ? we just set outputPerInput earlier 
                outputPerInput = pointsPerFrame.getValue()[inIdx];
            }

            for (unsigned iOutput = 0;
                 iOutput < outputPerInput; 
                 ++iOutput, ++outIdx) {
				
				const Coord& v = rotatedPoints[outIdx];

				impl::fill_block(block, v);

				// block is set, now insert it in sparse matrix
				for(unsigned i = 0; i < NOut; ++i){
					unsigned row = outIdx * NOut + i;
					
					J.startVec( row );
					
					// TODO optimize identity off-diagonal and
					// skew-symmetric diagonal
					for(unsigned j = 0; j < NIn; ++j) {
						unsigned col = inIdx * NIn + j;

						if( block(i, j) != 0 ) {
							J.insertBack(row, col) = block(i, j);
						}

					}
				}
            }
        }

		J.finalize();		
	}
												
    return &eigenJacobians;
}
#endif

template <class TIn, class TOut>
const sofa::defaulttype::BaseMatrix* RigidMapping<TIn, TOut>::getJ()
{
    const VecCoord& out =this->toModel->read(core::ConstVecCoordId::position())->getValue();
    const InVecCoord& in =this->fromModel->read(core::ConstVecCoordId::position())->getValue();
    const VecCoord& pts = this->getPoints();
    assert(pts.size() == out.size());

    if (matrixJ.get() == 0 || updateJ)
    {
        updateJ = false;
        if (matrixJ.get() == 0 ||
                (unsigned int)matrixJ->rowBSize() != out.size() ||
                (unsigned int)matrixJ->colBSize() != in.size())
        {
            matrixJ.reset(new MatrixType(out.size() * NOut, in.size() * NIn));
        }
        else
        {
            matrixJ->clear();
        }

        //        bool isMaskInUse = maskTo->isInUse();
        unsigned repartitionCount = pointsPerFrame.getValue().size();

        if (repartitionCount > 1 && repartitionCount != in.size())
        {
            serr << "getJ Error : mapping dofs repartition is not correct" << sendl;
            return 0;
        }

        unsigned inIdxBegin;
        unsigned inIdxEnd;

        if (repartitionCount == 0)
        {
            inIdxBegin = index.getValue();
            if (indexFromEnd.getValue())
            {
                inIdxBegin = in.size() - 1 - inIdxBegin;
            }
            inIdxEnd = inIdxBegin + 1;
        }
        else
        {
            inIdxBegin = 0;
            inIdxEnd = in.size();
        }

        unsigned outputPerInput;
        if (repartitionCount == 0)
        {
            outputPerInput = pts.size();
        }
        else
        {
            outputPerInput = pointsPerFrame.getValue()[0];
        }

        //        typedef helper::ParticleMask ParticleMask;
        //        const ParticleMask::InternalStorage& indices = maskTo->getEntries();
        //        ParticleMask::InternalStorage::const_iterator it = indices.begin();

        for (unsigned inIdx = inIdxBegin, outIdx = 0; inIdx < inIdxEnd; ++inIdx)
        {
            if (repartitionCount > 1)
            {
                outputPerInput = pointsPerFrame.getValue()[inIdx];
            }

            for (unsigned iOutput = 0;
                 iOutput < outputPerInput; // iOutput < outputPerInput && !(isMaskInUse && it == indices.end());
                 ++iOutput, ++outIdx)
            {
                //                if (isMaskInUse)
                //                {
                //                    if (outIdx != *it)
                //                    {
                //                        continue;
                //                    }
                //                    ++it;
                //                }
                setJMatrixBlock(outIdx, inIdx);
            }
        }
    }
    matrixJ->compress();
    return matrixJ.get();
}

template<class Real>
struct RigidMappingMatrixHelper<2, Real>
{
    template <class Matrix, class Vector>
    static void setMatrix(Matrix& mat,
                          const Vector& vec)
    {
        mat[0][0] = (Real) 1     ;    mat[1][0] = (Real) 0     ;
        mat[0][1] = (Real) 0     ;    mat[1][1] = (Real) 1     ;
        mat[0][2] = (Real)-vec[1];    mat[1][2] = (Real) vec[0];
    }
};

template<class Real>
struct RigidMappingMatrixHelper<3, Real>
{
    template <class Matrix, class Vector>
    static void setMatrix(Matrix& mat,
                          const Vector& vec)
    {
        // out = J in
        // J = [ I -OM^ ]
        mat[0][0] = (Real) 1     ;    mat[1][0] = (Real) 0     ;    mat[2][0] = (Real) 0     ;
        mat[0][1] = (Real) 0     ;    mat[1][1] = (Real) 1     ;    mat[2][1] = (Real) 0     ;
        mat[0][2] = (Real) 0     ;    mat[1][2] = (Real) 0     ;    mat[2][2] = (Real) 1     ;
        mat[0][3] = (Real) 0     ;    mat[1][3] = (Real)-vec[2];    mat[2][3] = (Real) vec[1];
        mat[0][4] = (Real) vec[2];    mat[1][4] = (Real) 0     ;    mat[2][4] = (Real)-vec[0];
        mat[0][5] = (Real)-vec[1];    mat[1][5] = (Real) vec[0];    mat[2][5] = (Real) 0     ;
    }
};

template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::setJMatrixBlock(unsigned outIdx, unsigned inIdx)
{
    //    cerr<<"RigidMapping<TIn, TOut>::setJMatrixBlock, outIdx = " << outIdx << ", inIdx = " << inIdx << endl;
    MBloc& block = *matrixJ->wbloc(outIdx, inIdx, true);
    RigidMappingMatrixHelper<N, Real>::setMatrix(block, rotatedPoints[outIdx]);
}

template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowMappings() || this->toModel==NULL )
        return;
    std::vector<defaulttype::Vector3> points;
    defaulttype::Vector3 point;

    const VecCoord& x =this->toModel->read(core::ConstVecCoordId::position())->getValue();
    for (unsigned int i = 0; i < x.size(); i++)
    {
        point = OutDataTypes::getCPos(x[i]);
        points.push_back(point);
    }
    vparams->drawTool()->drawPoints(points, 7,
                                    defaulttype::Vec<4, float>(1, 1, 0,1));
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
