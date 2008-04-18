/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_SKINNINGMAPPING_INL
#define SOFA_COMPONENT_MAPPING_SKINNINGMAPPING_INL

#include <sofa/component/mapping/SkinningMapping.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/io/MassSpringLoader.h>
#include <sofa/helper/io/SphereLoader.h>
#include <sofa/helper/io/Mesh.h>
#include <sofa/helper/gl/template.h>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.inl>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <string>
#include <iostream>

using std::cerr;
using std::endl;



namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;

template <class BasicMapping>
class SkinningMapping<BasicMapping>::Loader : public helper::io::MassSpringLoader, public helper::io::SphereLoader
{
public:
    SkinningMapping<BasicMapping>* dest;
    Loader(SkinningMapping<BasicMapping>* dest) : dest(dest) {}
    virtual void addMass(double /*px*/, double /*py*/, double /*pz*/, double, double, double, double, double, bool, bool)
    {
        /*Coord c;
        Out::DataTypes::set(c,px,py,pz);
        dest->initPos.push_back(c); //Coord((Real)px,(Real)py,(Real)pz));*/
    }
    virtual void addSphere(double /*px*/, double /*py*/, double /*pz*/, double)
    {
        /*Coord c;
        Out::DataTypes::set(c,px,py,pz);
        dest->initPos.push_back(c); //Coord((Real)px,(Real)py,(Real)pz));*/
    }
};

template <class BasicMapping>
void SkinningMapping<BasicMapping>::load(const char * /*filename*/)
{
    /*   initPos.resize(0);

       if (strlen(filename)>4 && !strcmp(filename+strlen(filename)-4,".xs3"))
       {
           Loader loader(this);
           loader.helper::io::MassSpringLoader::load(filename);
       }
       else
       if (strlen(filename)>4 && !strcmp(filename+strlen(filename)-4,".sph"))
       {
           Loader loader(this);
           loader.helper::io::SphereLoader::load(filename);
       }
       else if (strlen(filename)>0)
       { // Default to mesh loader
           helper::io::Mesh* mesh = helper::io::Mesh::Create(filename);
           if (mesh!=NULL)
           {
               initPos.resize(mesh->getVertices().size());
               for (unsigned int i=0;i<mesh->getVertices().size();i++)
               {
                   Out::DataTypes::set(initPos[i].getCenter(), mesh->getVertices()[i][0], mesh->getVertices()[i][1], mesh->getVertices()[i][2]);
               }
               delete mesh;
           }
       }*/
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::init()
{
    if (this->initPos.empty() && this->toModel!=NULL && computeWeights==true && coefs.getValue().size()==0)
    {
        VecCoord& xto = *this->toModel->getX();
        VecInCoord& xfrom = *this->fromModel->getX();
        initPos.resize(nbRefs.getValue()*xto.size());

        //init the arrays
        sofa::helper::vector<unsigned int> m_reps = repartition.getValue();
        m_reps.clear();
        m_reps.resize(nbRefs.getValue()*xto.size());
        sofa::helper::vector<double> m_coefs = coefs.getValue();
        m_coefs.clear();
        m_coefs.resize(nbRefs.getValue()*xto.size());

        Coord posTo;

        double * minDists = new double[nbRefs.getValue()];
        unsigned int * minInds = new unsigned int[nbRefs.getValue()];

        for (unsigned int i=0; i<xto.size(); i++)
        {
            posTo = xto[i];
            for (unsigned int h=0 ; h<nbRefs.getValue() ; h++)
                minDists[h] = 9999999. ;

            //search the nbRefs nearest "from" dofs of each "to" point
            for (unsigned int j=0; j<xfrom.size(); j++)
            {
                Real dist2 = (posTo - xfrom[j].getCenter()).norm();

                unsigned int k=0;
                while(k<nbRefs.getValue())
                {
                    if( dist2 < minDists[k] )
                    {
                        for(unsigned int m=nbRefs.getValue()-1 ; m>k ; m--)
                        {
                            minDists[m] = minDists[m-1];
                            minInds[m] = minInds[m-1];
                        }
                        minDists[k] = dist2;
                        minInds[k] = j;
                        k=nbRefs.getValue();
                    }
                    k++;
                }
            }

            //then compute the coefficients from the inverse distance (coef = 1/d)
            for (unsigned int k=0; k<nbRefs.getValue(); k++)
            {
                minDists[k] = 1 / minDists[k];
            }
            //minDists.normalize();
            //normalize the coefs vector such as the sum is equal to 1
            double norm=0.0;
            for (unsigned int h=0 ; h<nbRefs.getValue(); h++)
                norm += minDists[h]*minDists[h];
            norm = helper::rsqrt(norm);

            for (unsigned int g=0 ; g<nbRefs.getValue(); g++)
                minDists[g] /= norm;

            for (unsigned int m=0; m<nbRefs.getValue(); m++)
            {
                m_coefs[nbRefs.getValue()*i+m] = minDists[m]*minDists[m];
                m_reps[nbRefs.getValue()*i+m] = minInds[m];

                initPos[nbRefs.getValue()*i+m].getCenter() = xfrom[minInds[m]].getOrientation().inverseRotate(posTo - xfrom[minInds[m]].getCenter());
                initPos[nbRefs.getValue()*i+m].getOrientation() = xfrom[minInds[m]].getOrientation();
            }
        }
        repartition.setValue(m_reps);
        coefs.setValue(m_coefs);
        delete [] minInds;
        delete [] minDists;
    }
    else if (computeWeights == false || coefs.getValue().size()!=0)
    {
        sofa::helper::vector<unsigned int> m_reps = repartition.getValue();
        sofa::helper::vector<double> m_coefs = coefs.getValue();
        VecCoord& xto = *this->toModel->getX();
        VecInCoord& xfrom = *this->fromModel->getX();
        initPos.resize(nbRefs.getValue()*xto.size());
        Coord posTo;

        for (unsigned int i=0; i<xto.size(); i++)
        {
            posTo = xto[i];
            for (unsigned int m=0; m<nbRefs.getValue(); m++)
            {
                initPos[nbRefs.getValue()*i+m].getCenter() = xfrom[m_reps[nbRefs.getValue()*i+m]].getOrientation().inverseRotate(posTo - xfrom[m_reps[nbRefs.getValue()*i+m]].getCenter());
                initPos[nbRefs.getValue()*i+m].getOrientation() = xfrom[m_reps[nbRefs.getValue()*i+m]].getOrientation();
            }
        }
    }

    this->BasicMapping::init();
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::clear()
{
    this->initPos.clear();
}


template <class BasicMapping>
void SkinningMapping<BasicMapping>::setWeightCoefs(sofa::helper::vector<double> &weights)
{
    sofa::helper::vector<double> * m_coefs = coefs.beginEdit();
    m_coefs->clear();
    m_coefs->insert(m_coefs->begin(), weights.begin(), weights.end() );
    coefs.endEdit();
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::setRepartition(sofa::helper::vector<unsigned int> &rep)
{
    sofa::helper::vector<unsigned int> * m_reps = repartition.beginEdit();
    m_reps->clear();
    m_reps->insert(m_reps->begin(), rep.begin(), rep.end() );;
    repartition.endEdit();
}


template <class BasicMapping>
void SkinningMapping<BasicMapping>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    const sofa::helper::vector<unsigned int>& m_reps = repartition.getValue();
    const sofa::helper::vector<double>& m_coefs = coefs.getValue();
    rotatedPoints.resize(initPos.size());
    out.resize(initPos.size()/nbRefs.getValue());

    for (unsigned int i=0 ; i<out.size(); i++)
    {
        out[i] = Coord();
        for (unsigned int m=0 ; m<nbRefs.getValue(); m++)
        {
            translation = in[m_reps[nbRefs.getValue()*i+m] ].getCenter();
            Quat relativeRot = initPos[nbRefs.getValue()*i+m].getOrientation().inverse() * in[m_reps[nbRefs.getValue()*i+m] ].getOrientation();
            //in[repartition[nbRefs.getValue()*i+m].getValue()].writeRotationMatrix(rotation);
            relativeRot.toMatrix(rotation);
            rotatedPoints[nbRefs.getValue()*i+m] = rotation * (initPos[nbRefs.getValue()*i+m].getCenter());

            out[i] += initPos[nbRefs.getValue()*i+m].getOrientation().rotate(rotatedPoints[nbRefs.getValue()*i+m] * m_coefs[nbRefs.getValue()*i+m]);
            out[i] += translation * m_coefs[nbRefs.getValue()*i+m];
        }
    }
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    const sofa::helper::vector<unsigned int>& m_reps = repartition.getValue();
    const sofa::helper::vector<double>& m_coefs = coefs.getValue();

    Deriv v,omega;
    out.resize(initPos.size()/nbRefs.getValue());

    for(unsigned int i=0; i<out.size(); i++)
    {
        out[i] = Deriv();
        for (unsigned int m=0 ; m<nbRefs.getValue(); m++)
        {
            v = in[m_reps[nbRefs.getValue()*i+m]].getVCenter();
            omega = in[m_reps[nbRefs.getValue()*i+m]].getVOrientation();
            out[i] +=  (v - cross(rotatedPoints[nbRefs.getValue()*i+m],omega)) * m_coefs[nbRefs.getValue()*i+m];
        }
    }
}


template <class BasicMapping>
void SkinningMapping<BasicMapping>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    const sofa::helper::vector<unsigned int>& m_reps = repartition.getValue();
    const sofa::helper::vector<double>& m_coefs = coefs.getValue();

    Deriv v,omega;
    for(unsigned int i=0; i<in.size(); i++)
    {
        for (unsigned int m=0 ; m<nbRefs.getValue(); m++)
        {
            Deriv f = in[i];
            v = f;
            omega = cross(rotatedPoints[nbRefs.getValue()*i+m],f);
            out[m_reps[nbRefs.getValue()*i+m] ].getVCenter() += v * m_coefs[nbRefs.getValue()*i+m];
            out[m_reps[nbRefs.getValue()*i+m] ].getVOrientation() += omega * m_coefs[nbRefs.getValue()*i+m];
        }
    }

}


template <class BasicMapping>
void SkinningMapping<BasicMapping>::applyJT( typename In::VecConst& out, const typename Out::VecConst& in )
{
    const sofa::helper::vector<unsigned int>& m_reps = repartition.getValue();
    const sofa::helper::vector<double>& m_coefs = coefs.getValue();
    const unsigned int nbr = nbRefs.getValue();
    const unsigned int nbi = this->fromModel->getX()->size();
    Deriv omega;
    typename In::VecDeriv v;
    sofa::helper::vector<bool> flags;
    int outSize = out.size();
    out.resize(in.size() + outSize); // we can accumulate in "out" constraints from several mappings
    for(unsigned int j=0; j<in.size(); j++)
    {
        v.clear();
        v.resize(nbi);
        flags.clear();
        flags.resize(nbi);
        for (unsigned int id=0; id<in[0].size(); ++id)
        {
            unsigned int i = in[0][id].index;
            Deriv f = in[0][id].data;
            for (unsigned int m=0 ; m<nbr; m++)
            {
                omega = cross(rotatedPoints[nbr*i+m],f);
                flags[m_reps[nbr*i+m] ] = true;
                v[m_reps[nbr*i+m] ].getVCenter() += f * m_coefs[nbr*i+m];
                v[m_reps[nbr*i+m] ].getVOrientation() += omega * m_coefs[nbr*i+m];
            }
        }
        out[outSize+j].reserve(nbi);
        for (unsigned int i=0 ; i<nbi; i++)
        {
            //if (!(v[i] == typename In::Deriv()))
            if (flags[i])
                out[outSize+j].push_back(typename In::SparseDeriv(i,v[i]));
        }
    }
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::draw()
{
    if (!this->getShow()) return;
    glDisable (GL_LIGHTING);
    glPointSize(1);
    glColor4f (1,1,0,1);
    glBegin (GL_LINES);

    const typename Out::VecCoord& xOut = *this->toModel->getX();
    const typename In::VecCoord& xIn = *this->fromModel->getX();
    sofa::helper::vector<unsigned int> m_reps = repartition.getValue();
    sofa::helper::vector<double> m_coefs = coefs.getValue();

    for (unsigned int i=0; i<xOut.size(); i++)
    {
        for (unsigned int m=0 ; m<nbRefs.getValue(); m++)
        {
            if(m_coefs[nbRefs.getValue()*i+m] > 0.0)
            {
                glColor4d (m_coefs[nbRefs.getValue()*i+m],m_coefs[nbRefs.getValue()*i+m],0,1);
                helper::gl::glVertexT(xIn[m_reps[nbRefs.getValue()*i+m] ].getCenter());
                helper::gl::glVertexT(xOut[i]);
            }
        }
    }
    glEnd();
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
