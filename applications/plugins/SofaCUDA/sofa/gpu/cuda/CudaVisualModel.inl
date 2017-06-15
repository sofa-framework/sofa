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
#ifndef SOFA_GPU_CUDA_CUDAVISUALMODEL_INL
#define SOFA_GPU_CUDA_CUDAVISUALMODEL_INL

#include "CudaVisualModel.h"
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/gl/template.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{
    void CudaVisualModelCuda3f_calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3f_calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3f_calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x);

    void CudaVisualModelCuda3f1_calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3f1_calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3f1_calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x);

#ifdef SOFA_GPU_CUDA_DOUBLE

    void CudaVisualModelCuda3d_calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3d_calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3d_calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x);

    void CudaVisualModelCuda3d1_calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3d1_calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
    void CudaVisualModelCuda3d1_calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x);

#endif // SOFA_GPU_CUDA_DOUBLE

} // extern "C"

template<>
class CudaKernelsCudaVisualModel<CudaVec3fTypes>
{
public:
    static void calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
    {   CudaVisualModelCuda3f_calcTNormals(nbElem, nbVertex, elems, fnormals, x); }
    static void calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
    {   CudaVisualModelCuda3f_calcQNormals(nbElem, nbVertex, elems, fnormals, x); }
    static void calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x)
    {   CudaVisualModelCuda3f_calcVNormals(nbElem, nbVertex, nbElemPerVertex, velems, vnormals, fnormals, x); }
};

template<>
class CudaKernelsCudaVisualModel<CudaVec3f1Types>
{
public:
    static void calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
    {   CudaVisualModelCuda3f1_calcTNormals(nbElem, nbVertex, elems, fnormals, x); }
    static void calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
    {   CudaVisualModelCuda3f1_calcQNormals(nbElem, nbVertex, elems, fnormals, x); }
    static void calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x)
    {   CudaVisualModelCuda3f1_calcVNormals(nbElem, nbVertex, nbElemPerVertex, velems, vnormals, fnormals, x); }
};

#ifdef SOFA_GPU_CUDA_DOUBLE

template<>
class CudaKernelsCudaVisualModel<CudaVec3dTypes>
{
public:
    static void calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
    {   CudaVisualModelCuda3d_calcTNormals(nbElem, nbVertex, elems, fnormals, x); }
    static void calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
    {   CudaVisualModelCuda3d_calcQNormals(nbElem, nbVertex, elems, fnormals, x); }
    static void calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x)
    {   CudaVisualModelCuda3d_calcVNormals(nbElem, nbVertex, nbElemPerVertex, velems, vnormals, fnormals, x); }
};

template<>
class CudaKernelsCudaVisualModel<CudaVec3d1Types>
{
public:
    static void calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
    {   CudaVisualModelCuda3d1_calcTNormals(nbElem, nbVertex, elems, fnormals, x); }
    static void calcQNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
    {   CudaVisualModelCuda3d1_calcQNormals(nbElem, nbVertex, elems, fnormals, x); }
    static void calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x)
    {   CudaVisualModelCuda3d1_calcVNormals(nbElem, nbVertex, nbElemPerVertex, velems, vnormals, fnormals, x); }
};

#endif // SOFA_GPU_CUDA_DOUBLE

} // namespace cuda

} // namespace gpu

namespace component
{

namespace visualmodel
{

using namespace gpu::cuda;

template<class TDataTypes>
void CudaVisualModel< TDataTypes >::init()
{
    this->getContext()->get(state);
    topology = this->getContext()->getMeshTopology();
    updateVisual();
}

template<class TDataTypes>
void CudaVisualModel< TDataTypes >::reinit()
{
    updateVisual();
}

template<class TDataTypes>
void CudaVisualModel< TDataTypes >::handleTopologyChange()
{
    std::list<const core::topology::TopologyChange *>::const_iterator itBegin=topology->beginChange();
    std::list<const core::topology::TopologyChange *>::const_iterator itEnd=topology->endChange();

    while( itBegin != itEnd )
    {
        core::topology::TopologyChangeType changeType = (*itBegin)->getChangeType();

        switch( changeType )
        {

// 		case core::topology::TRIANGLESADDED:
// 			{
// 			  printf("TRIANGLESADDED\n");
// 				needUpdateTopology = true;
// 				break;
// 			}

        case core::topology::TRIANGLESREMOVED:
        {
            needUpdateTopology = true;
            break;
        }

        case core::topology::QUADSADDED:
        {
            needUpdateTopology = true;
            break;
        }

        case core::topology::QUADSREMOVED:
        {
            needUpdateTopology = true;
            break;
        }
        default:
            break;
        }
        ++itBegin;
    }
}

template<class TDataTypes>
void CudaVisualModel< TDataTypes >::updateTopology()
{
    if (!topology || !state) return;
    if (!needUpdateTopology) return;
    needUpdateTopology = false;
    {
        const SeqTriangles& t = topology->getTriangles();
        triangles.clear();
        if (!t.empty())
        {
            triangles.fastResize(t.size());
            std::copy ( t.begin(), t.end(), triangles.hostWrite() );
        }
    }
    {
        const SeqQuads& q = topology->getQuads();
        quads.clear();
        if (!q.empty())
        {
            quads.fastResize(q.size());
            std::copy ( q.begin(), q.end(), quads.hostWrite() );
        }
    }
    const Triangle* tptr = triangles.hostRead();
    const Quad*     qptr = quads.hostRead();
    std::map<int,int> nelems;
    for (unsigned int i=0; i<triangles.size(); i++)
    {
        const Triangle& e = tptr[i];
        for (unsigned int j=0; j<e.size(); j++)
            ++nelems[e[j]];
    }
    for (unsigned int i=0; i<quads.size(); i++)
    {
        const Quad& e = qptr[i];
        for (unsigned int j=0; j<e.size(); j++)
            ++nelems[e[j]];
    }
    int nmax = 0;
    for (std::map<int,int>::const_iterator it = nelems.begin(); it != nelems.end(); ++it)
        if (it->second > nmax)
            nmax = it->second;
    int nbv = 0;
    if (!nelems.empty())
        nbv = nelems.rbegin()->first + 1;
    sout << "CUDA CudaVisualModel: "<<triangles.size()<<" triangles, "<<quads.size()<<" quads, "<<nbv<<"/"<<state->getSize()<<" attached points, max "<<nmax<<" elements per point."<<sendl;
    initV(triangles.size()+quads.size(), nbv, nmax);

    nelems.clear();
    for (unsigned int i=0; i<triangles.size(); i++)
    {
        const Triangle& e = tptr[i];
        for (unsigned int j=0; j<e.size(); j++)
            setV(e[j], nelems[e[j]]++, i);
    }
    int i0 = triangles.size();
    for (unsigned int i=0; i<quads.size(); i++)
    {
        const Quad& e = qptr[i];
        for (unsigned int j=0; j<e.size(); j++)
            setV(e[j], nelems[e[j]]++, i0+i);
    }
}


template<class TDataTypes>
void CudaVisualModel< TDataTypes >::updateNormals()
{
    if (!topology || !state || !state->getSize()) return;
    const VecCoord& x = state->read(core::ConstVecCoordId::position())->getValue();
    fnormals.resize(nbElement);
    vnormals.resize(x.size());
    if (triangles.size() > 0)
        Kernels::calcTNormals(
            triangles.size(),
            nbVertex,
            triangles.deviceRead(),
            fnormals.deviceWrite(),
            x.deviceRead());
    if (quads.size() > 0)
        Kernels::calcQNormals(
            quads.size(),
            nbVertex,
            quads.deviceRead(),
            fnormals.deviceWriteAt(triangles.size()),
            x.deviceRead());
    if (nbVertex > 0)
        Kernels::calcVNormals(
            nbElement,
            nbVertex,
            nbElementPerVertex,
            velems.deviceRead(),
            vnormals.deviceWrite(),
            fnormals.deviceRead(),
            x.deviceRead());
}

template<class TDataTypes>
void CudaVisualModel< TDataTypes >::updateVisual()
{
    updateTopology();
    if (computeNormals.getValue())
        updateNormals();
}

template<class TDataTypes>
void CudaVisualModel< TDataTypes >::drawVisual(const core::visual::VisualParams* vparams)
{
    bool transparent = (matDiffuse.getValue()[3] < 1.0);
    if (!transparent) internalDraw(vparams);
}

template<class TDataTypes>
void CudaVisualModel< TDataTypes >::drawTransparent(const core::visual::VisualParams* vparams)
{
    bool transparent = (matDiffuse.getValue()[3] < 1.0);
    if (transparent) internalDraw(vparams);
}

template<class TDataTypes>
void CudaVisualModel< TDataTypes >::drawShadow(const core::visual::VisualParams* vparams)
{
    bool transparent = (matDiffuse.getValue()[3] < 1.0);
    if (!transparent /* && getCastShadow() */ ) internalDraw(vparams);
}

template<class TDataTypes>
void CudaVisualModel< TDataTypes >::internalDraw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowVisualModels()) return;

    if (!topology || !state || !state->getSize()) return;

    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glEnable(GL_LIGHTING);

    bool transparent = (matDiffuse.getValue()[3] < 1.0);

    //Enable<GL_BLEND> blending;
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    defaulttype::Vec4f ambient = matAmbient.getValue();
    defaulttype::Vec4f diffuse = matDiffuse.getValue();
    defaulttype::Vec4f specular = matSpecular.getValue();
    defaulttype::Vec4f emissive = matEmissive.getValue();
    float shininess = matShininess.getValue();

    if (shininess == 0.0f)
    {
        specular.clear();
        shininess = 1;
    }

    if (transparent)
    {
        ambient[3] = 0;
        specular[3] = 0;
        emissive[3] = 0;
    }

    glMaterialfv (GL_FRONT_AND_BACK, GL_AMBIENT, ambient.ptr());
    glMaterialfv (GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse.ptr());
    glMaterialfv (GL_FRONT_AND_BACK, GL_SPECULAR, specular.ptr());
    glMaterialfv (GL_FRONT_AND_BACK, GL_EMISSION, emissive.ptr());
    glMaterialf (GL_FRONT_AND_BACK, GL_SHININESS, shininess);

    if (transparent)
    {
        glEnable(GL_BLEND);
        glDepthMask(GL_FALSE);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }

    //TODO: Const ? Read-Only ?
    //VecCoord& x = *state->getX();
    Data<VecCoord>* d_x = state->write(core::VecCoordId::position());
    VecCoord& x = *d_x->beginEdit();

    bool vbo = useVBO.getValue();

    GLuint vbo_x = vbo ? x.bufferRead(true) : 0;
    if (vbo_x)
    {
        glBindBuffer(GL_ARRAY_BUFFER, vbo_x);
        glVertexPointer (3, (sizeof(Real)==sizeof(double))?GL_DOUBLE:GL_FLOAT, sizeof(Coord), NULL);
    }
    else
        glVertexPointer (3, (sizeof(Real)==sizeof(double))?GL_DOUBLE:GL_FLOAT, sizeof(Coord), x.hostRead());

    if (computeNormals.getValue())
    {
        GLuint vbo_n = vbo ? vnormals.bufferRead(true) : 0;
        if (vbo_n)
        {
            glBindBuffer(GL_ARRAY_BUFFER, vbo_n);
            glNormalPointer ((sizeof(Real)==sizeof(double))?GL_DOUBLE:GL_FLOAT, sizeof(Coord), NULL);
        }
        else
            glNormalPointer ((sizeof(Real)==sizeof(double))?GL_DOUBLE:GL_FLOAT, sizeof(Coord), vnormals.hostRead());
        glEnableClientState(GL_NORMAL_ARRAY);
    }
    glEnableClientState(GL_VERTEX_ARRAY);

    if (triangles.size() > 0)
    {
        GLuint vbo_t = vbo ? triangles.bufferRead(true) : 0;
        if (vbo_t)
        {
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_t);
            glDrawElements(GL_TRIANGLES, triangles.size() * 3, GL_UNSIGNED_INT, NULL);
        }
        else
            glDrawElements(GL_TRIANGLES, triangles.size() * 3, GL_UNSIGNED_INT, triangles.hostRead());
    }

    if (quads.size() > 0)
    {
        GLuint vbo_q = vbo ? quads.bufferRead(true) : 0;
        if (vbo_q)
        {
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_q);
            glDrawElements(GL_QUADS, quads.size() * 4, GL_UNSIGNED_INT, NULL);
        }
        else
            glDrawElements(GL_QUADS, quads.size() * 4, GL_UNSIGNED_INT, quads.hostRead());
    }

    if (vbo)
    {
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisable(GL_LIGHTING);

    if (transparent)
    {
        glDisable(GL_BLEND);
        glDepthMask(GL_TRUE);
    }
    glDisable(GL_LIGHTING);

    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    if (vparams->displayFlags().getShowNormals())
    {
        glColor3f (1.0, 1.0, 1.0);
        for (unsigned int i = 0; i < x.size(); i++)
        {
            glBegin(GL_LINES);
            helper::gl::glVertexT(x[i]);
            Coord p = x[i] + vnormals[i]*0.01;
            helper::gl::glVertexT(p);
            glEnd();
        }
    }

    d_x->endEdit();
}

template<class TDataTypes>
void CudaVisualModel< TDataTypes >::computeBBox(const core::ExecParams* params, bool)
{
    const VecCoord& x = state->write(core::VecCoordId::position())->getValue();

    SReal minBBox[3] = {std::numeric_limits<Real>::max(),std::numeric_limits<Real>::max(),std::numeric_limits<Real>::max()};
    SReal maxBBox[3] = {-std::numeric_limits<Real>::max(),-std::numeric_limits<Real>::max(),-std::numeric_limits<Real>::max()};
    for (unsigned int i = 0; i < x.size(); i++)
    {
        const Coord& p = x[i];
        for (int c=0; c<3; c++)
        {
            if (p[c] > maxBBox[c]) maxBBox[c] = p[c];
            if (p[c] < minBBox[c]) minBBox[c] = p[c];
        }
    }
    this->f_bbox.setValue(params,sofa::defaulttype::TBoundingBox<SReal>(minBBox,maxBBox));
}


} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif
