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
#ifdef SOFA_SMP
#include <sofa/component/container/MechanicalObjectTasks.inl>
#endif
#define SOFA_COMPONENT_CONTAINER_MECHANICALOBJECT_CPP
#include <sofa/component/container/MechanicalObject.inl>
#include <sofa/helper/Quater.h>
#include <sofa/core/ObjectFactory.h>


namespace sofa
{

namespace component
{

namespace container
{

using namespace core::behavior;
using namespace defaulttype;

SOFA_DECL_CLASS(MechanicalObject)

int MechanicalObjectClass = core::RegisterObject("mechanical state vectors")
#ifdef SOFA_FLOAT
        .add< MechanicalObject<Vec3fTypes> >(true) // default template
#else
        .add< MechanicalObject<Vec3dTypes> >(true) // default template
#ifndef SOFA_DOUBLE
        .add< MechanicalObject<Vec3fTypes> >()
#endif
#endif
#ifndef SOFA_FLOAT
        .add< MechanicalObject<Vec2dTypes> >()
        .add< MechanicalObject<Vec1dTypes> >()
        .add< MechanicalObject<Vec6dTypes> >()
        .add< MechanicalObject<Rigid3dTypes> >()
        .add< MechanicalObject<Rigid2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< MechanicalObject<Vec2fTypes> >()
        .add< MechanicalObject<Vec1fTypes> >()
        .add< MechanicalObject<Vec6fTypes> >()
        .add< MechanicalObject<Rigid3fTypes> >()
        .add< MechanicalObject<Rigid2fTypes> >()
#endif
        .add< MechanicalObject<LaparoscopicRigid3Types> >()
        ;

// template specialization must be in the same namespace as original namespace for GCC 4.1
// g++ 4.1 requires template instantiations to be declared on a parent namespace from the template class.
#ifndef SOFA_FLOAT
template class SOFA_BASE_MECHANICS_API MechanicalObject<Vec3dTypes>;
template class SOFA_BASE_MECHANICS_API MechanicalObject<Vec2dTypes>;
template class SOFA_BASE_MECHANICS_API MechanicalObject<Vec1dTypes>;
template class SOFA_BASE_MECHANICS_API MechanicalObject<Vec6dTypes>;
template class SOFA_BASE_MECHANICS_API MechanicalObject<Rigid3dTypes>;
template class SOFA_BASE_MECHANICS_API MechanicalObject<Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BASE_MECHANICS_API MechanicalObject<Vec3fTypes>;
template class SOFA_BASE_MECHANICS_API MechanicalObject<Vec2fTypes>;
template class SOFA_BASE_MECHANICS_API MechanicalObject<Vec1fTypes>;
template class SOFA_BASE_MECHANICS_API MechanicalObject<Vec6fTypes>;
template class SOFA_BASE_MECHANICS_API MechanicalObject<Rigid3fTypes>;
template class SOFA_BASE_MECHANICS_API MechanicalObject<Rigid2fTypes>;
#endif
template class SOFA_BASE_MECHANICS_API MechanicalObject<LaparoscopicRigid3Types>;



#ifndef SOFA_FLOAT
template<> SOFA_BASE_MECHANICS_API
void MechanicalObject<defaulttype::Rigid3dTypes>::applyRotation (const defaulttype::Quat q)
{
    helper::WriteAccessor< Data<VecCoord> > x = *this->write(core::VecCoordId::position());

    for (unsigned int i = 0; i < x.size(); i++)
    {
        x[i].getCenter() = q.rotate(x[i].getCenter());
        x[i].getOrientation() *= q;
    }
}

template<> SOFA_BASE_MECHANICS_API
void MechanicalObject<defaulttype::Rigid3dTypes>::addFromBaseVectorDifferentSize(core::VecId dest, const defaulttype::BaseVector* src, unsigned int &offset )
{

    if (dest.type == sofa::core::V_COORD)
    {

        helper::WriteAccessor< Data<VecCoord> > vDest = *this->write(VecCoordId(dest));
        const unsigned int coordDim = DataTypeInfo<Coord>::size();
        const unsigned int nbEntries = src->size()/coordDim;

        for (unsigned int i=0; i<nbEntries; i++)
        {
            for (unsigned int j=0; j<3; ++j)
            {
                Real tmp;
                DataTypeInfo<Coord>::getValue(vDest[i+offset],j,tmp);
                DataTypeInfo<Coord>::setValue(vDest[i+offset],j, tmp + src->element(i*coordDim+j));
            }

            helper::Quater<double> q_src;
            helper::Quater<double> q_dest;
            for (unsigned int j=0; j<4; j++)
            {
                Real tmp;
                DataTypeInfo<Coord>::getValue(vDest[i+offset],j+3,tmp);
                q_dest[j]=tmp;
                q_src[j]=src->element(i * coordDim + j+3);
            }
            //q_dest = q_dest*q_src;
            q_dest = q_src*q_dest;
            for (unsigned int j=0; j<4; j++)
            {
                Real tmp=q_dest[j];
                DataTypeInfo<Coord>::setValue(vDest[i+offset], j+3, tmp);
            }
        }
        offset += nbEntries;
    }
    else
    {
        helper::WriteAccessor< Data<VecDeriv> > vDest = *this->write(VecDerivId(dest));

        const unsigned int derivDim = DataTypeInfo<Deriv>::size();
        const unsigned int nbEntries = src->size()/derivDim;
        for (unsigned int i=0; i<nbEntries; i++)
        {
            for (unsigned int j=0; j<derivDim; ++j)
            {
                Real tmp;
                DataTypeInfo<Deriv>::getValue(vDest[i+offset],j,tmp);
                DataTypeInfo<Deriv>::setValue(vDest[i+offset],j, tmp + src->element(i*derivDim+j));
            }
        }
        offset += nbEntries;
    }


}

template<> SOFA_BASE_MECHANICS_API
void MechanicalObject<defaulttype::Rigid3dTypes>::addFromBaseVectorSameSize(core::VecId dest, const defaulttype::BaseVector* src, unsigned int &offset)
{
    if (dest.type == sofa::core::V_COORD)
    {
        helper::WriteAccessor< Data<VecCoord> > vDest = *this->write(VecCoordId(dest));
        const unsigned int coordDim = DataTypeInfo<Coord>::size();

        for (unsigned int i=0; i<vDest.size(); i++)
        {
            for (unsigned int j=0; j<3; j++)
            {
                Real tmp;
                DataTypeInfo<Coord>::getValue(vDest[i],j,tmp);
                DataTypeInfo<Coord>::setValue(vDest[i],j,tmp + src->element(offset + i * coordDim + j));
            }

            helper::Quater<double> q_src;
            helper::Quater<double> q_dest;
            for (unsigned int j=0; j<4; j++)
            {
                Real tmp;
                DataTypeInfo<Coord>::getValue(vDest[i],j+3,tmp);
                q_dest[j]=tmp;
                q_src[j]=src->element(offset + i * coordDim + j+3);
            }
            //q_dest = q_dest*q_src;
            q_dest = q_src*q_dest;
            for (unsigned int j=0; j<4; j++)
            {
                Real tmp=q_dest[j];
                DataTypeInfo<Coord>::setValue(vDest[i], j+3, tmp);
            }
        }

        offset += vDest.size() * coordDim;
    }
    else
    {
        helper::WriteAccessor< Data<VecDeriv> > vDest = *this->write(VecDerivId(dest));
        const unsigned int derivDim = DataTypeInfo<Deriv>::size();
        for (unsigned int i=0; i<vDest.size(); i++)
        {
            for (unsigned int j=0; j<derivDim; j++)
            {
                Real tmp;
                DataTypeInfo<Deriv>::getValue(vDest[i],j,tmp);
                DataTypeInfo<Deriv>::setValue(vDest[i], j, tmp + src->element(offset + i * derivDim + j));
            }
        }
        offset += vDest.size() * derivDim;
    }

}


template<> SOFA_BASE_MECHANICS_API
void MechanicalObject<defaulttype::Rigid3dTypes>::draw(const core::visual::VisualParams* vparams)
{
    Mat<4,4, GLfloat> modelviewM;
    Vec<3, SReal> sceneMinBBox, sceneMaxBBox;
    sofa::simulation::Node* context;
    if (showIndices.getValue())
    {
        context = dynamic_cast<sofa::simulation::Node*>(this->getContext());
        glColor3f(1.0,1.0,1.0);
        glDisable(GL_LIGHTING);
        sofa::simulation::getSimulation()->computeBBox((sofa::simulation::Node*)context, sceneMinBBox.ptr(), sceneMaxBBox.ptr());
        float scale = (sceneMaxBBox - sceneMinBBox).norm() * showIndicesScale.getValue();

        for (int i=0 ; i< vsize ; i++)
        {
            std::ostringstream oss;
            oss << i;
            std::string tmp = oss.str();
            const char* s = tmp.c_str();
            //glVertex3f(getPX(i),getPY(i),getPZ(i) );
            glPushMatrix();

            glTranslatef(getPX(i), getPY(i), getPZ(i));
            glScalef(scale,scale,scale);

            // Makes text always face the viewer by removing the scene rotation
            // get the current modelview matrix
            glGetFloatv(GL_MODELVIEW_MATRIX , modelviewM.ptr() );
            modelviewM.transpose();

            Vec3d temp(getPX(i), getPY(i), getPZ(i));
            temp = modelviewM.transform(temp);

            //glLoadMatrixf(modelview);
            glLoadIdentity();

            glTranslatef(temp[0], temp[1], temp[2]);
            glScalef(scale,scale,scale);

            while(*s)
            {
                glutStrokeCharacter(GLUT_STROKE_ROMAN, *s);
                s++;
            }

            glPopMatrix();
        }
    }

    if (showObject.getValue())
    {
        glPushAttrib(GL_LIGHTING_BIT);
        glDisable(GL_LIGHTING);
        const float& scale = showObjectScale.getValue();
        helper::ReadAccessor<Data<VecCoord> > x = *this->read(core::VecCoordId::position());
        for (int i = 0; i < vsize; ++i)
        {
            glPushMatrix();
            //glTranslatef(getPX(i), getPY(i), getPZ(i));
            float glTransform[16];
            x[i].writeOpenGlMatrix( glTransform);
            glMultMatrixf( glTransform);
            glScalef(scale,scale,scale);
            vparams->drawTool()->drawFrame(Vector3(), Quat(), Vector3(1.0,1.0,1.0));
            glPopMatrix();
        }
        glPopAttrib();
    }
}

#endif

#ifndef SOFA_DOUBLE
template<> SOFA_BASE_MECHANICS_API
void MechanicalObject<defaulttype::Rigid3fTypes>::applyRotation (const defaulttype::Quat q)
{
    helper::WriteAccessor< Data<VecCoord> > x = *this->write(core::VecCoordId::position());

    for (unsigned int i = 0; i < x.size(); i++)
    {
        x[i].getCenter() = q.rotate(x[i].getCenter());
        x[i].getOrientation() *= q;
    }
}


template<> SOFA_BASE_MECHANICS_API
void MechanicalObject<defaulttype::Rigid3fTypes>::addFromBaseVectorDifferentSize(core::VecId dest, const defaulttype::BaseVector* src, unsigned int &offset )
{
    if (dest.type == sofa::core::V_COORD)
    {

        helper::WriteAccessor< Data<VecCoord> > vDest = *this->write(VecCoordId(dest));
        const unsigned int coordDim = DataTypeInfo<Coord>::size();
        const unsigned int nbEntries = src->size()/coordDim;

        for (unsigned int i=0; i<nbEntries; i++)
        {
            for (unsigned int j=0; j<3; ++j)
            {
                Real tmp;
                DataTypeInfo<Coord>::getValue(vDest[i+offset],j,tmp);
                DataTypeInfo<Coord>::setValue(vDest[i+offset],j, tmp + src->element(i*coordDim+j));
            }

            helper::Quater<double> q_src;
            helper::Quater<double> q_dest;
            for (unsigned int j=0; j<4; j++)
            {
                Real tmp;
                DataTypeInfo<Coord>::getValue(vDest[i+offset],j+3,tmp);
                q_dest[j]=tmp;
                q_src[j]=src->element(i * coordDim + j+3);
            }
            //q_dest = q_dest*q_src;
            q_dest = q_src*q_dest;
            for (unsigned int j=0; j<4; j++)
            {
                Real tmp=q_dest[j];
                DataTypeInfo<Coord>::setValue(vDest[i+offset], j+3, tmp);
            }
        }
        offset += nbEntries;
    }
    else
    {
        helper::WriteAccessor< Data<VecDeriv> > vDest = *this->write(VecDerivId(dest));

        const unsigned int derivDim = DataTypeInfo<Deriv>::size();
        const unsigned int nbEntries = src->size()/derivDim;
        for (unsigned int i=0; i<nbEntries; i++)
        {
            for (unsigned int j=0; j<derivDim; ++j)
            {
                Real tmp;
                DataTypeInfo<Deriv>::getValue(vDest[i+offset],j,tmp);
                DataTypeInfo<Deriv>::setValue(vDest[i+offset],j, tmp + src->element(i*derivDim+j));
            }
        }
        offset += nbEntries;
    }

}

template<> SOFA_BASE_MECHANICS_API
void MechanicalObject<defaulttype::Rigid3fTypes>::addFromBaseVectorSameSize(core::VecId dest, const defaulttype::BaseVector* src, unsigned int &offset)
{
    if (dest.type == sofa::core::V_COORD)
    {
        helper::WriteAccessor< Data<VecCoord> > vDest = *this->write(VecCoordId(dest));
        const unsigned int coordDim = DataTypeInfo<Coord>::size();

        for (unsigned int i=0; i<vDest.size(); i++)
        {
            for (unsigned int j=0; j<3; j++)
            {
                Real tmp;
                DataTypeInfo<Coord>::getValue(vDest[i],j,tmp);
                DataTypeInfo<Coord>::setValue(vDest[i],j,tmp + src->element(offset + i * coordDim + j));
            }

            helper::Quater<double> q_src;
            helper::Quater<double> q_dest;
            for (unsigned int j=0; j<4; j++)
            {
                Real tmp;
                DataTypeInfo<Coord>::getValue(vDest[i],j+3,tmp);
                q_dest[j]=tmp;
                q_src[j]=src->element(offset + i * coordDim + j+3);
            }
            //q_dest = q_dest*q_src;
            q_dest = q_src*q_dest;
            for (unsigned int j=0; j<4; j++)
            {
                Real tmp=q_dest[j];
                DataTypeInfo<Coord>::setValue(vDest[i], j+3, tmp);
            }
        }

        offset += vDest.size() * coordDim;
    }
    else
    {
        helper::WriteAccessor< Data<VecDeriv> > vDest = *this->write(VecDerivId(dest));
        const unsigned int derivDim = DataTypeInfo<Deriv>::size();
        for (unsigned int i=0; i<vDest.size(); i++)
        {
            for (unsigned int j=0; j<derivDim; j++)
            {
                Real tmp;
                DataTypeInfo<Deriv>::getValue(vDest[i],j,tmp);
                DataTypeInfo<Deriv>::setValue(vDest[i], j, tmp + src->element(offset + i * derivDim + j));
            }
        }
        offset += vDest.size() * derivDim;
    }

}

// template <>
//     bool MechanicalObject<Vec1fTypes>::addBBox(double* /*minBBox*/, double* /*maxBBox*/)
// {
//     return false; // ignore 1D DOFs for 3D bbox
// }

template<> SOFA_BASE_MECHANICS_API
void MechanicalObject<defaulttype::Rigid3fTypes>::draw(const core::visual::VisualParams* vparams)
{
    Mat<4,4, GLfloat> modelviewM;
    Vec<3, SReal> sceneMinBBox, sceneMaxBBox;
    sofa::simulation::Node* context;
    if (showIndices.getValue())
    {
        context = dynamic_cast<sofa::simulation::Node*>(this->getContext());
        glColor3f(1.0,1.0,1.0);
        glDisable(GL_LIGHTING);
        sofa::simulation::getSimulation()->computeBBox((sofa::simulation::Node*)context, sceneMinBBox.ptr(), sceneMaxBBox.ptr());
        float scale = (sceneMaxBBox - sceneMinBBox).norm() * showIndicesScale.getValue();

        for (int i=0 ; i< vsize ; i++)
        {
            std::ostringstream oss;
            oss << i;
            std::string tmp = oss.str();
            const char* s = tmp.c_str();
            //glVertex3f(getPX(i),getPY(i),getPZ(i) );
            glPushMatrix();

            glTranslatef(getPX(i), getPY(i), getPZ(i));
            glScalef(scale,scale,scale);

            // Makes text always face the viewer by removing the scene rotation
            // get the current modelview matrix
            glGetFloatv(GL_MODELVIEW_MATRIX , modelviewM.ptr() );
            modelviewM.transpose();

            Vec3d temp(getPX(i), getPY(i), getPZ(i));
            temp = modelviewM.transform(temp);

            //glLoadMatrixf(modelview);
            glLoadIdentity();

            glTranslatef(temp[0], temp[1], temp[2]);
            glScalef(scale,scale,scale);

            while(*s)
            {
                glutStrokeCharacter(GLUT_STROKE_ROMAN, *s);
                s++;
            }

            glPopMatrix();
        }
    }

    if (showObject.getValue())
    {
        glPushAttrib(GL_LIGHTING_BIT);
        glDisable(GL_LIGHTING);
        const float& scale = showObjectScale.getValue();
        helper::ReadAccessor<Data<VecCoord> > x = *this->read(core::VecCoordId::position());
        for (int i = 0; i < vsize; ++i)
        {
            glPushMatrix();
            //glTranslatef(getPX(i), getPY(i), getPZ(i));
            float glTransform[16];
            x[i].writeOpenGlMatrix( glTransform);
            glMultMatrixf( glTransform);
            glScalef(scale,scale,scale);
            vparams->drawTool()->drawFrame(Vector3(), Quat(), Vector3(1.0,1.0,1.0));
            glPopMatrix();
        }
        glPopAttrib();
    }
}

#endif

template<> SOFA_BASE_MECHANICS_API
void MechanicalObject<defaulttype::LaparoscopicRigid3Types>::draw(const core::visual::VisualParams* vparams)
{
    Mat<4,4, GLfloat> modelviewM;
    Vec<3, SReal> sceneMinBBox, sceneMaxBBox;
    sofa::simulation::Node* context;
    if (showIndices.getValue())
    {
        context = dynamic_cast<sofa::simulation::Node*>(this->getContext());
        glColor3f(1.0,1.0,1.0);
        glDisable(GL_LIGHTING);
        sofa::simulation::getSimulation()->computeBBox((sofa::simulation::Node*)context, sceneMinBBox.ptr(), sceneMaxBBox.ptr());
        float scale = (sceneMaxBBox - sceneMinBBox).norm() * showIndicesScale.getValue();

        for (int i=0 ; i< vsize ; i++)
        {
            std::ostringstream oss;
            oss << i;
            std::string tmp = oss.str();
            const char* s = tmp.c_str();
            //glVertex3f(getPX(i),getPY(i),getPZ(i) );
            glPushMatrix();

            glTranslatef(getPX(i), getPY(i), getPZ(i));
            glScalef(scale,scale,scale);

            // Makes text always face the viewer by removing the scene rotation
            // get the current modelview matrix
            glGetFloatv(GL_MODELVIEW_MATRIX , modelviewM.ptr() );
            modelviewM.transpose();

            Vec3d temp(getPX(i), getPY(i), getPZ(i));
            temp = modelviewM.transform(temp);

            //glLoadMatrixf(modelview);
            glLoadIdentity();

            glTranslatef(temp[0], temp[1], temp[2]);
            glScalef(scale,scale,scale);

            while(*s)
            {
                glutStrokeCharacter(GLUT_STROKE_ROMAN, *s);
                s++;
            }

            glPopMatrix();
        }
    }

    if (showObject.getValue())
    {
        glPushAttrib(GL_LIGHTING_BIT);
        glDisable(GL_LIGHTING);
        const float& scale = showObjectScale.getValue();
        helper::ReadAccessor<Data<VecCoord> > x = *this->read(core::VecCoordId::position());
        for (int i = 0; i < vsize; ++i)
        {
            glPushMatrix();
            glTranslatef(getPX(i), getPY(i), getPZ(i));
            glScalef(scale,scale,scale);
            vparams->drawTool()->drawFrame(Vector3(), x[i].getOrientation(), Vector3(1.0,1.0,1.0));
            glPopMatrix();
        }
        glPopAttrib();
    }
}

}

} // namespace component

} // namespace sofa
