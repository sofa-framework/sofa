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
    virtual void addMass(double px, double py, double pz, double, double, double, double, double, bool, bool)
    {
        Coord c;
        Out::DataTypes::set(c,px,py,pz);
        dest->initPos.push_back(c); //Coord((Real)px,(Real)py,(Real)pz));
    }
    virtual void addSphere(double px, double py, double pz, double)
    {
        Coord c;
        Out::DataTypes::set(c,px,py,pz);
        dest->initPos.push_back(c); //Coord((Real)px,(Real)py,(Real)pz));
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
    if (this->initPos.empty() && this->toModel!=NULL)
    {
        VecCoord& xto = *this->toModel->getX();
        VecInCoord& xfrom = *this->fromModel->getX();
        initPos.resize(nbRefs.getValue()*xto.size());

        //init the arrays
        repartition = new DataField<unsigned int>[nbRefs.getValue()*xto.size()];
        coefs = new DataField<double>[nbRefs.getValue()*xto.size()];

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
                coefs[nbRefs.getValue()*i+m].setValue(minDists[m]*minDists[m]);
                repartition[nbRefs.getValue()*i+m].setValue( minInds[m]);

                initPos[nbRefs.getValue()*i+m].getCenter() = (posTo - xfrom[minInds[m]].getCenter());
                initPos[nbRefs.getValue()*i+m].getOrientation() = xfrom[minInds[m]].getOrientation();
            }
        }
        delete [] minInds;
        delete [] minDists;
    }
    this->BasicMapping::init();
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::clear()
{
    this->initPos.clear();
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    rotatedPoints.resize(initPos.size());
    out.resize(initPos.size()/nbRefs.getValue());

    for (unsigned int i=0 ; i<out.size(); i++)
    {
        out[i] = Coord();
        for (unsigned int m=0 ; m<nbRefs.getValue(); m++)
        {
            translation = in[repartition[nbRefs.getValue()*i+m].getValue()].getCenter();
            Quat relativeRot = initPos[nbRefs.getValue()*i+m].getOrientation().inverse() * in[repartition[nbRefs.getValue()*i+m].getValue()].getOrientation();
            //in[repartition[nbRefs.getValue()*i+m].getValue()].writeRotationMatrix(rotation);
            relativeRot.toMatrix(rotation);
            rotatedPoints[nbRefs.getValue()*i+m] = rotation * (initPos[nbRefs.getValue()*i+m].getCenter());

            out[i] += rotatedPoints[nbRefs.getValue()*i+m] * coefs[nbRefs.getValue()*i+m].getValue() ;
            out[i] += translation * coefs[nbRefs.getValue()*i+m].getValue();
        }
    }
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    Deriv v,omega;
    out.resize(initPos.size()/nbRefs.getValue());

    for(unsigned int i=0; i<out.size(); i++)
    {
        out[i] = Deriv();
        for (unsigned int m=0 ; m<nbRefs.getValue(); m++)
        {
            v = in[repartition[nbRefs.getValue()*i+m].getValue()].getVCenter();
            omega = in[repartition[nbRefs.getValue()*i+m].getValue()].getVOrientation();
            out[i] +=  (v - cross(rotatedPoints[nbRefs.getValue()*i+m],omega)) * coefs[nbRefs.getValue()*i+m].getValue();
        }
    }
}


template <class BasicMapping>
void SkinningMapping<BasicMapping>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    Deriv v,omega;
    for(unsigned int i=0; i<in.size(); i++)
    {
        for (unsigned int m=0 ; m<nbRefs.getValue(); m++)
        {
            Deriv f = in[i];
            v = f;
            omega = cross(rotatedPoints[nbRefs.getValue()*i+m],f);
            out[repartition[nbRefs.getValue()*i+m].getValue()].getVCenter() += v * coefs[nbRefs.getValue()*i+m].getValue();
            out[repartition[nbRefs.getValue()*i+m].getValue()].getVOrientation() += omega * coefs[nbRefs.getValue()*i+m].getValue();
        }
    }

}


template <class BasicMapping>
void SkinningMapping<BasicMapping>::draw()
{
    if (!getShow(this)) return;
    glDisable (GL_LIGHTING);
    glPointSize(7);
    glColor4f (1,1,0,1);
    glBegin (GL_POINTS);
    const typename Out::VecCoord& x = *this->toModel->getX();
    for (unsigned int i=0; i<x.size(); i++)
    {
        helper::gl::glVertexT(x[i]);
    }
    glEnd();
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
