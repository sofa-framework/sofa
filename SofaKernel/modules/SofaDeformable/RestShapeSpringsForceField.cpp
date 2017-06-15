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
#define SOFA_COMPONENT_FORCEFIELD_RESTSHAPESPRINGSFORCEFIELD_CPP

#include <SofaDeformable/RestShapeSpringsForceField.inl>

#include <sofa/core/visual/DrawTool.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(RestShapeSpringsForceField)

///////////// SPECIALIZATION FOR RIGID TYPES //////////////


#ifndef SOFA_FLOAT

template<>
void RestShapeSpringsForceField<Rigid3dTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& /* v */)
{
    sofa::helper::WriteAccessor< DataVecDeriv > f1 = f;
    sofa::helper::ReadAccessor< DataVecCoord > p1 = x;

    sofa::helper::ReadAccessor< DataVecCoord > p0 = *getExtPosition();

    //std::cout<<"addForce with p_0 ="<<p0<<std::endl;

    f1.resize(p1.size());
    //std::cout<<" size p1:"<<p1.size()<<std::endl;

    if (recompute_indices.getValue())
    {
        recomputeIndices();
    }

    const VecReal& k = stiffness.getValue();
    const VecReal& k_a = angularStiffness.getValue();

    for (unsigned int i = 0; i < m_indices.size(); i++)
    {
        //std::cout<<"i="<<i<<std::endl;
        const unsigned int index = m_indices[i];
        unsigned int ext_index = m_indices[i];
        if(useRestMState)
            ext_index= m_ext_indices[i];

        // translation
        if (i >= m_pivots.size())
        {
            Vec3d dx = p1[index].getCenter() - p0[ext_index].getCenter();
            //std::cout<<"dx = "<< dx <<std::endl;
            getVCenter(f1[index]) -=  dx * (i < k.size() ? k[i] : k[0]) ;
        }
        else
        {
            CPos localPivot = p0[ext_index].getOrientation().inverseRotate(m_pivots[i] - p0[ext_index].getCenter());
            //std::cout << "localPivot = "e  << localPivot << std::endl;
            CPos rotatedPivot = p1[index].getOrientation().rotate(localPivot);
            CPos pivot2 = p1[index].getCenter() + rotatedPivot;
            CPos dx = pivot2 - m_pivots[i];
            //std::cout << "dx = " << dx << std::endl;
            getVCenter(f1[index]) -= dx * (i < k.size() ? k[i] : k[0]) ;

            //getVOrientation(f1[index]) -= cross(rotatedPivot, dx * k[i]);
        }

        // rotation
        Quatd dq = p1[index].getOrientation() * p0[ext_index].getOrientation().inverse();
        Vec3d dir;
        double angle=0;
        dq.normalize();

        if (dq[3] < 0)
        {
            //std::cout<<"WARNING inversion quaternion"<<std::endl;
            dq = dq * -1.0;
        }

        if (dq[3] < 0.999999999999999)
            dq.quatToAxis(dir, angle);

        //std::cout<<"dq : "<<dq <<"  dir :"<<dir<<"  angle :"<<angle<<"  index : "<<index<<"  f1.size() : "<<f1.size()<<std::endl;
        //Vec3d m1 = getVOrientation(f1[index]) ;
        //std::cout<<"m1 = "<<m1<<std::endl;


        getVOrientation(f1[index]) -= dir * angle * (i < k_a.size() ? k_a[i] : k_a[0]);
        //std::cout<<"dq : "<<dq <<"  dir :"<<dir<<"  angle :"<<angle<<std::endl;

    }

    //std::cout<<" f1 = "<<f1<<std::endl;
}


template<>
void RestShapeSpringsForceField<Rigid3dTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx)
{
    sofa::helper::WriteAccessor< DataVecDeriv > df1 = df;
    sofa::helper::ReadAccessor< DataVecDeriv > dx1 = dx;

    const VecReal& k = stiffness.getValue();
    const VecReal& k_a = angularStiffness.getValue();
    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    unsigned int curIndex = 0;

    for (unsigned int i=0; i<m_indices.size(); i++)
    {
        curIndex = m_indices[i];
        getVCenter(df1[curIndex])	 -=  getVCenter(dx1[curIndex]) * ( (i < k.size()) ? k[i] : k[0] ) * kFactor ;
        getVOrientation(df1[curIndex]) -=  getVOrientation(dx1[curIndex]) * (i < k_a.size() ? k_a[i] : k_a[0]) * kFactor ;
    }
}


template<>
void RestShapeSpringsForceField<Rigid3dTypes>::addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix )
{
    const VecReal& k = stiffness.getValue();
    const VecReal& k_a = angularStiffness.getValue();
    const int N = 6;
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef mref = matrix->getMatrix(this->mstate);
    sofa::defaulttype::BaseMatrix* mat = mref.matrix;
    unsigned int offset = mref.offset;
    Real kFact = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    unsigned int curIndex = 0;

    for (unsigned int index = 0; index < m_indices.size(); index++)
    {
        curIndex = m_indices[index];

        // translation
        for(int i = 0; i < 3; i++)
        {
            mat->add(offset + N * curIndex + i, offset + N * curIndex + i, -kFact * (index < k.size() ? k[index] : k[0]));
        }

        // rotation
        for(int i = 3; i < 6; i++)
        {
            mat->add(offset + N * curIndex + i, offset + N * curIndex + i, -kFact * (index < k_a.size() ? k_a[index] : k_a[0]));
        }
    }

    /* debug
    std::cout<<"MAT obtained : size: ("<<mat->rowSize()<<" * "<<mat->colSize()<<")\n"<<std::endl;

    for (unsigned int col=0; col<mat->colSize(); col++)
    {
        for (unsigned int row=0; row<mat->rowSize(); row++)
        {
                std::cout<<" "<<mat->element(row, col);
        }

        std::cout<<""<<std::endl;
    }
    */
}

template<>
void RestShapeSpringsForceField<Rigid3dTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields() || !drawSpring.getValue())
        return;  /// \todo put this in the parent class

    vparams->drawTool()->saveLastState();
    vparams->drawTool()->setLightingEnabled(false);

    sofa::helper::ReadAccessor< DataVecCoord > p0 = *getExtPosition();
    sofa::helper::ReadAccessor< DataVecCoord > p = this->mstate->read(core::VecCoordId::position());

    sofa::helper::vector< Vector3 > vertices;

    for (unsigned int i=0; i<m_indices.size(); i++)
    {
        const unsigned int index = m_indices[i];

        vertices.push_back(p[index].getCenter());

        if(useRestMState)
        {
            const unsigned int ext_index = m_ext_indices[i];
            vertices.push_back(p0[ext_index].getCenter());
        }
        else
        {
            vertices.push_back(p0[index].getCenter());
        }
    }
    vparams->drawTool()->drawLines(vertices,5, Vec4f(springColor.getValue()));
    vparams->drawTool()->restoreLastState();
}


#endif // SOFA_FLOAT

#ifndef SOFA_DOUBLE

template<>
void RestShapeSpringsForceField<Rigid3fTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& /* v */)
{
    sofa::helper::WriteAccessor< DataVecDeriv > f1 = f;
    sofa::helper::ReadAccessor< DataVecCoord > p1 = x;

    sofa::helper::ReadAccessor< DataVecCoord > p0 = *getExtPosition();

    f1.resize(p1.size());

    if (recompute_indices.getValue())
    {
        recomputeIndices();
    }

    const VecReal& k = stiffness.getValue();
    const VecReal& k_a = angularStiffness.getValue();

    for (unsigned int i=0; i<m_indices.size(); i++)
    {
        const unsigned int index = m_indices[i];
        const unsigned int ext_index = m_ext_indices[i];

        // translation
        Vec3f dx = p1[index].getCenter() - p0[ext_index].getCenter();
        getVCenter(f1[index]) -=  dx * (i < k.size() ? k[i] : k[0]) ;

        // rotation
        Quatf dq = p1[index].getOrientation() * p0[ext_index].getOrientation().inverse();
        Vec3f dir;
        Real angle=0;
        dq.normalize();
        if (dq[3] < 0.999999999999999)
            dq.quatToAxis(dir, angle);
        dq.quatToAxis(dir, angle);

        //std::cout<<"dq : "<<dq <<"  dir :"<<dir<<"  angle :"<<angle<<std::endl;
        getVOrientation(f1[index]) -= dir * angle * (i < k_a.size() ? k_a[i] : k_a[0]) ;
    }
}


template<>
void RestShapeSpringsForceField<Rigid3fTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx)
{
    sofa::helper::WriteAccessor< DataVecDeriv > df1 = df;
    sofa::helper::ReadAccessor< DataVecDeriv > dx1 = dx;
    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    const VecReal& k = stiffness.getValue();
    const VecReal& k_a = angularStiffness.getValue();

    unsigned int curIndex = 0;

    for (unsigned int i=0; i<m_indices.size(); i++)
    {
//		curIndex = m_indices[index];
        curIndex = m_indices[i];  // Fix by FF, just a guess
        getVCenter(df1[curIndex])      -=  getVCenter(dx1[curIndex])      * (i < k.size() ? k[i] : k[0])   * kFactor ;
        getVOrientation(df1[curIndex]) -=  getVOrientation(dx1[curIndex]) * (i < k_a.size() ? k_a[i] : k_a[0]) * kFactor ;
    }
}


template<>
void RestShapeSpringsForceField<Rigid3fTypes>::addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix )
{
    const VecReal& k = stiffness.getValue();
    const VecReal& k_a = angularStiffness.getValue();
    const int N = 6;

    sofa::core::behavior::MultiMatrixAccessor::MatrixRef mref = matrix->getMatrix(this->mstate);
    sofa::defaulttype::BaseMatrix* mat = mref.matrix;
    unsigned int offset = mref.offset;
    Real kFact = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    unsigned int curIndex = 0;


    for (unsigned int index = 0; index < m_indices.size(); index++)
    {
        curIndex = m_indices[index];

        // translation
        for(int i = 0; i < 3; i++)
        {
            mat->add(offset + N * curIndex + i, offset + N * curIndex + i, kFact * (index < k.size() ? k[index] : k[0]));
        }

        // rotation
        for(int i = 3; i < 6; i++)
        {
            mat->add(offset + N * curIndex + i, offset + N * curIndex + i, kFact * (index < k_a.size() ? k_a[index] : k_a[0]));
        }
    }
}

template<>
void RestShapeSpringsForceField<Rigid3fTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields() || !drawSpring.getValue())
        return;  /// \todo put this in the parent class

    vparams->drawTool()->saveLastState();
    vparams->drawTool()->setLightingEnabled(false);

    sofa::helper::ReadAccessor< DataVecCoord > p0 = *getExtPosition();
    sofa::helper::ReadAccessor< DataVecCoord > p = this->mstate->read(core::VecCoordId::position());

    sofa::helper::vector<sofa::defaulttype::Vector3> vertices;

    for (unsigned int i=0; i<m_indices.size(); i++)
    {
        const unsigned int index = m_indices[i];

        sofa::defaulttype::Vector3 v0(p[index].getCenter()[0],
                                      p[index].getCenter()[1],
                                      p[index].getCenter()[2]);
        unsigned int tempIndex = (useRestMState) ? m_ext_indices[i] : index;

        sofa::defaulttype::Vector3 v1(p0[tempIndex].getCenter()[0],
                                      p0[tempIndex].getCenter()[1],
                                      p0[tempIndex].getCenter()[2]);


        vertices.push_back(v0);
        vertices.push_back(v1);
    }

    vparams->drawTool()->drawLines(vertices,5,springColor.getValue());

    vparams->drawTool()->restoreLastState();
}

#endif // SOFA_DOUBLE

#ifndef SOFA_FLOAT

/*
template<>
void RestShapeSpringsForceField<Vec3dTypes>::addDForce(VecDeriv& df, const VecDeriv &dx, SReal kFactor, SReal )
{
const VecIndex& indices = points.getValue();
const VecReal& k = stiffness.getValue();

if (k.size()!= indices.size() )
{
    sout << "WARNING : stiffness is not defined on each point, first stiffness is used" << sendl;

    for (unsigned int i=0; i<indices.size(); i++)
    {
            df[indices[i]] -=  Springs_dir[i]  * k[0] * kFactor * dot(dx[indices[i]], Springs_dir[i]);
    }
}
else
{
    for (unsigned int i=0; i<indices.size(); i++)
    {
    //	df[ indices[i] ] -=  dx[indices[i]] * k[i] * kFactor ;
            df[indices[i]] -=   Springs_dir[i]  * k[indices[i]] * kFactor * dot(dx[indices[i]] , Springs_dir[i]);
    }
}
}
*/

#endif



int RestShapeSpringsForceFieldClass = core::RegisterObject("Simple elastic springs applied to given degrees of freedom between their current and rest shape position")
#ifndef SOFA_FLOAT
        .add< RestShapeSpringsForceField<Vec3dTypes> >()
//.add< RestShapeSpringsForceField<Vec2dTypes> >()
        .add< RestShapeSpringsForceField<Vec1dTypes> >()
//.add< RestShapeSpringsForceField<Vec6dTypes> >()
        .add< RestShapeSpringsForceField<Rigid3dTypes> >()
//.add< RestShapeSpringsForceField<Rigid2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< RestShapeSpringsForceField<Vec3fTypes> >()
//.add< RestShapeSpringsForceField<Vec2fTypes> >()
        .add< RestShapeSpringsForceField<Vec1fTypes> >()
//.add< RestShapeSpringsForceField<Vec6fTypes> >()
        .add< RestShapeSpringsForceField<Rigid3fTypes> >()
//.add< RestShapeSpringsForceField<Rigid2fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_DEFORMABLE_API RestShapeSpringsForceField<Vec3dTypes>;
//template class SOFA_DEFORMABLE_API RestShapeSpringsForceField<Vec2dTypes>;
template class SOFA_DEFORMABLE_API RestShapeSpringsForceField<Vec1dTypes>;
//template class SOFA_DEFORMABLE_API RestShapeSpringsForceField<Vec6dTypes>;
template class SOFA_DEFORMABLE_API RestShapeSpringsForceField<Rigid3dTypes>;
//template class SOFA_DEFORMABLE_API RestShapeSpringsForceField<Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_DEFORMABLE_API RestShapeSpringsForceField<Vec3fTypes>;
//template class SOFA_DEFORMABLE_API RestShapeSpringsForceField<Vec2fTypes>;
template class SOFA_DEFORMABLE_API RestShapeSpringsForceField<Vec1fTypes>;
//template class SOFA_DEFORMABLE_API RestShapeSpringsForceField<Vec6fTypes>;
template class SOFA_DEFORMABLE_API RestShapeSpringsForceField<Rigid3fTypes>;
//template class SOFA_DEFORMABLE_API RestShapeSpringsForceField<Rigid2fTypes>;
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa
