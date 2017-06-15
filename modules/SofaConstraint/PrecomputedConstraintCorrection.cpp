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
#define SOFA_COMPONENT_CONSTRAINTSET_PRECOMPUTEDCONSTRAINTCORRECTION_CPP

#include "PrecomputedConstraintCorrection.inl"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace constraintset
{

#ifndef SOFA_FLOAT

template<>
SOFA_CONSTRAINT_API void PrecomputedConstraintCorrection< defaulttype::Rigid3dTypes >::rotateConstraints(bool back)
{
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    const VecCoord& x0 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
    helper::WriteAccessor<Data<MatrixDeriv> > cData = *this->mstate->write(core::MatrixDerivId::holonomicC());
    MatrixDeriv& c = cData.wref();

    // On fait tourner les normales (en les ramenant dans le "pseudo" repere initial)

    MatrixDerivRowIterator rowItEnd = c.end();

    for (MatrixDerivRowIterator rowIt = c.begin(); rowIt != rowItEnd; ++rowIt)
    {
        MatrixDerivColIterator colItEnd = rowIt.end();

        for (MatrixDerivColIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
        {
            Deriv& n = colIt.val();
            const unsigned int localRowNodeIdx = colIt.index();

            sofa::defaulttype::Quat q;
            if (m_restRotations.getValue())
                q = x[localRowNodeIdx].getOrientation() * x0[localRowNodeIdx].getOrientation().inverse();
            else
                q = x[localRowNodeIdx].getOrientation();

            sofa::defaulttype::Vec3d n_i = q.inverseRotate(getVCenter(n));
            sofa::defaulttype::Vec3d wn_i= q.inverseRotate(getVOrientation(n));

            if(back)
            {
                n_i = q.rotate(getVCenter(n));
                wn_i= q.rotate(getVOrientation(n));
            }

            // on passe les normales du repere global au repere local
            getVCenter(n) = n_i;
            getVOrientation(n) = wn_i;
        }
    }
}



template<>
SOFA_CONSTRAINT_API void PrecomputedConstraintCorrection<defaulttype::Vec1dTypes>::rotateConstraints(bool /*back*/)
{
}



template<>
SOFA_CONSTRAINT_API void PrecomputedConstraintCorrection<defaulttype::Rigid3dTypes>::rotateResponse()
{
    helper::WriteAccessor<Data<VecDeriv> > dxData = *this->mstate->write(core::VecDerivId::dx());
    VecDeriv& dx = dxData.wref();
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    const VecCoord& x0 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
    for(unsigned int j = 0; j < dx.size(); j++)
    {
        // on passe les deplacements du repere local (au repos) au repere global
        Deriv temp ;
        sofa::defaulttype::Quat q;
        if (m_restRotations.getValue())
            q = x[j].getOrientation() * x0[j].getOrientation().inverse();
        else
            q = x[j].getOrientation();

        getVCenter(temp)		= q.rotate(getVCenter(dx[j]));
        getVOrientation(temp)  = q.rotate(getVOrientation(dx[j]));
        dx[j] = temp;
    }
}


template<>
SOFA_CONSTRAINT_API void PrecomputedConstraintCorrection<defaulttype::Vec1dTypes>::rotateResponse()
{
}


template<>
SOFA_CONSTRAINT_API void PrecomputedConstraintCorrection<defaulttype::Vec1dTypes>::draw(const core::visual::VisualParams* )
{
}

template<>
SOFA_CONSTRAINT_API void PrecomputedConstraintCorrection<defaulttype::Rigid3dTypes>::draw(const core::visual::VisualParams* )
{
}


#endif
#ifndef SOFA_DOUBLE


template<>
SOFA_CONSTRAINT_API void PrecomputedConstraintCorrection< defaulttype::Rigid3fTypes >::rotateConstraints(bool back)
{
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    const VecCoord& x0 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    helper::WriteAccessor<Data<MatrixDeriv> > cData = *this->mstate->write(core::MatrixDerivId::holonomicC());
    MatrixDeriv& c = cData.wref();

    // On fait tourner les normales (en les ramenant dans le "pseudo" repere initial)

    MatrixDerivRowIterator rowItEnd = c.end();

    for (MatrixDerivRowIterator rowIt = c.begin(); rowIt != rowItEnd; ++rowIt)
    {
        MatrixDerivColIterator colItEnd = rowIt.end();

        for (MatrixDerivColIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
        {
            Deriv& n = colIt.val();
            const unsigned int localRowNodeIdx = colIt.index();

            sofa::defaulttype::Quat q;
            if (m_restRotations.getValue())
                q = x[localRowNodeIdx].getOrientation() * x0[localRowNodeIdx].getOrientation().inverse();
            else
                q = x[localRowNodeIdx].getOrientation();

            sofa::defaulttype::Vec3f n_i = q.inverseRotate(getVCenter(n));
            sofa::defaulttype::Vec3f wn_i= q.inverseRotate(getVOrientation(n));

            if(back)
            {
                n_i = q.rotate(getVCenter(n));
                wn_i= q.rotate(getVOrientation(n));
            }


            // on passe les normales du repere global au repere local
            getVCenter(n) = n_i;
            getVOrientation(n) = wn_i;
        }
    }
}


template<>
SOFA_CONSTRAINT_API void PrecomputedConstraintCorrection<defaulttype::Vec1fTypes>::rotateConstraints(bool /*back*/)
{
}


template<>
SOFA_CONSTRAINT_API void PrecomputedConstraintCorrection<defaulttype::Rigid3fTypes>::rotateResponse()
{
    helper::WriteAccessor<Data<VecDeriv> > dxData = *this->mstate->write(core::VecDerivId::dx());
    VecDeriv& dx = dxData.wref();
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    const VecCoord& x0 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
    for(unsigned int j = 0; j < dx.size(); j++)
    {
        // on passe les deplacements du repere local (au repos) au repere global
        Deriv temp ;
        sofa::defaulttype::Quat q;
        if (m_restRotations.getValue())
            q = x[j].getOrientation() * x0[j].getOrientation().inverse();
        else
            q = x[j].getOrientation();

        getVCenter(temp)		= q.rotate(getVCenter(dx[j]));
        getVOrientation(temp)  = q.rotate(getVOrientation(dx[j]));
        dx[j] = temp;
    }
}


template<>
SOFA_CONSTRAINT_API void PrecomputedConstraintCorrection<defaulttype::Vec1fTypes>::rotateResponse()
{
}

template<>
SOFA_CONSTRAINT_API void PrecomputedConstraintCorrection<defaulttype::Vec1fTypes>::draw(const core::visual::VisualParams* )
{
}

template<>
SOFA_CONSTRAINT_API void PrecomputedConstraintCorrection<defaulttype::Rigid3fTypes>::draw(const core::visual::VisualParams* )
{
}

#endif

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(PrecomputedConstraintCorrection)

int PrecomputedConstraintCorrectionClass = core::RegisterObject("Component computing contact forces within a simulated body using the compliance method.")
#ifndef SOFA_FLOAT
        .add< PrecomputedConstraintCorrection<Vec3dTypes> >()
        .add< PrecomputedConstraintCorrection<Vec1dTypes> >()
        .add< PrecomputedConstraintCorrection<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< PrecomputedConstraintCorrection<Vec3fTypes> >()
        .add< PrecomputedConstraintCorrection<Vec1fTypes> >()
        .add< PrecomputedConstraintCorrection<Rigid3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_CONSTRAINT_API PrecomputedConstraintCorrection<Vec3dTypes>;
template class SOFA_CONSTRAINT_API PrecomputedConstraintCorrection<Vec1dTypes>;
template class SOFA_CONSTRAINT_API PrecomputedConstraintCorrection<Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_CONSTRAINT_API PrecomputedConstraintCorrection<Vec3fTypes>;
template class SOFA_CONSTRAINT_API PrecomputedConstraintCorrection<Vec1fTypes>;
template class SOFA_CONSTRAINT_API PrecomputedConstraintCorrection<Rigid3fTypes>;
#endif

} // namespace collision

} // namespace component

} // namespace sofa
