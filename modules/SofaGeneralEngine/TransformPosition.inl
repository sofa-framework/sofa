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
#ifndef SOFA_COMPONENT_ENGINE_TRANSFORMPOSITION_INL
#define SOFA_COMPONENT_ENGINE_TRANSFORMPOSITION_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <SofaGeneralEngine/TransformPosition.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/gl/template.h>
#include <math.h>
#include <sofa/helper/RandomGenerator.h>

#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>

namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
TransformPosition<DataTypes>::TransformPosition()
    : f_origin( initData(&f_origin, "origin", "A 3d point on the plane/Center of the scale") )
    , f_inputX( initData (&f_inputX, "input_position", "input array of 3d points") )
    , f_outputX( initData (&f_outputX, "output_position", "output array of 3d points projected on a plane") )
    , f_normal(initData(&f_normal, "normal", "plane normal") )
    , f_translation(initData(&f_translation, "translation", "translation vector ") )
    , f_rotation(initData(&f_rotation, "rotation", "rotation vector ") )
    , f_scale(initData(&f_scale, Coord(1.0,1.0,1.0), "scale", "scale factor") )
    , f_affineMatrix(initData(&f_affineMatrix, Mat4x4::s_identity, "matrix", "4x4 affine matrix") )
    , f_method(initData(&f_method, "method", "transformation method either translation or scale or rotation or random or projectOnPlane") )
    , f_seed(initData(&f_seed, (long) 0, "seedValue", "the seed value for the random generator") )
    , f_maxRandomDisplacement(initData(&f_maxRandomDisplacement, (Real) 1.0, "maxRandomDisplacement", "the maximum displacement around initial position for the random transformation") )
    , f_fixedIndices( initData(&f_fixedIndices,"fixedIndices","Indices of the entries that are not transformed") )
    , f_filename(initData(&f_filename, "filename", "filename of an affine matrix. Supported extensions are: .trm, .tfm, .xfm and .txt(read as .xfm)") )
    , f_drawInput(initData(&f_drawInput, false, "drawInput", "Draw input points") )
    , f_drawOutput(initData(&f_drawOutput, false, "drawOutput", "Draw output points") )
    , f_pointSize(initData(&f_pointSize, (Real)1.0, "pointSize", "Point size") )
{
    addAlias(&f_inputX, "inputPosition");
    addAlias(&f_outputX, "outputPosition");

    f_pointSize.setGroup("Visualization");

    f_method.beginEdit()->setNames(9,
        "projectOnPlane",
        "translation",
        "rotation",
        "random",
        "scale",
        "scaleTranslation",
        "scaleRotationTranslation",
        "affine",
        "fromFile");
    f_method.endEdit();
}

template <class DataTypes>
void TransformPosition<DataTypes>::selectTransformationMethod()
{
    if (f_method.getValue().getSelectedItem()=="projectOnPlane")
    {
        transformationMethod=PROJECT_ON_PLANE;
    }
    else if (f_method.getValue().getSelectedItem()=="translation")
    {
        transformationMethod=TRANSLATION;
    }
    else if (f_method.getValue().getSelectedItem()=="rotation")
    {
        transformationMethod=ROTATION;
    }
    else if (f_method.getValue().getSelectedItem()=="random")
    {
        transformationMethod=RANDOM;
    }
    else if (f_method.getValue().getSelectedItem()=="scale")
    {
        transformationMethod=SCALE;
    }
    else if (f_method.getValue().getSelectedItem()=="scaleTranslation")
    {
        transformationMethod=SCALE_TRANSLATION;
    }
    else if (f_method.getValue().getSelectedItem()=="scaleRotationTranslation")
    {
        transformationMethod=SCALE_ROTATION_TRANSLATION;
    }
    else if (f_method.getValue().getSelectedItem()=="affine")
    {
        transformationMethod=AFFINE;
    }
    else if (f_method.getValue().getSelectedItem()=="fromFile")
    {
        transformationMethod=AFFINE;
        if (f_filename.isSet())
        {
            std::string fname = f_filename.getValue();
            if (fname.size()>=4 && fname.substr(fname.size()-4)==".trm")
                getTransfoFromTrm();
            else if (fname.size()>=4 && (fname.substr(fname.size()-4)==".txt" || fname.substr(fname.size()-4)==".xfm"))
                getTransfoFromTxt();
            else if (fname.size()>=4 && fname.substr(fname.size()-4)==".tfm")
                getTransfoFromTfm();
            else
                serr << "Unknown extension. Will use affine instead." << sendl;
        }
        else
        {
            serr << "Filename not set. Will use affine instead" <<sendl;
        }
    }
    else
    {
        transformationMethod=TRANSLATION;
        serr << "Error : Method " << f_method.getValue().getSelectedItem() << " is unknown. Wil use translation instead." <<sendl;
    }
}

template <class DataTypes>
void TransformPosition<DataTypes>::init()
{
    Coord& normal = *(f_normal.beginEdit());

    /// check if the normal is of norm 1
    if (fabs((normal.norm2()-1.0))>1e-10)
        normal/=normal.norm();

    f_normal.endEdit();

    addInput(&f_inputX);
    addInput(&f_origin);
    addInput(&f_normal);
    addInput(&f_translation);
    addInput(&f_rotation);
    addInput(&f_scale);
    addInput(&f_affineMatrix);
    addInput(&f_fixedIndices);

    addOutput(&f_outputX);

    setDirtyValue();
}

template <class DataTypes>
void TransformPosition<DataTypes>::reinit()
{
    update();
}

/**************************************************
 * .tfm spec:
 * 12 values in the lines begining by "Parameters"
 **************************************************/
template <class DataTypes>
void TransformPosition<DataTypes>::getTransfoFromTfm()
{
    std::string fname(this->f_filename.getFullPath());
    sout << "Loading .tfm file " << fname << sendl;

    std::ifstream stream(fname.c_str());
    if (stream)
    {
        std::string line;
        Mat4x4 mat(Mat4x4::s_identity);

        bool found = false;
        while (getline(stream,line) && !found)
        {
            if (line.find("Parameters")!=std::string::npos)
            {
                found=true;

                typedef std::vector<std::string> vecString;
                vecString vLine;

                char *l = new char[line.size()];
                strcpy(l, line.c_str());
                char* p;
                for (p = strtok(l, " "); p; p = strtok(NULL, " "))
                    vLine.push_back(std::string(p));
                delete [] l;

                std::vector<Real> values;
                for (vecString::iterator it = vLine.begin(); it < vLine.end(); it++)
                {
                    std::string c = *it;
                    if ( c.find_first_of("1234567890.-") != std::string::npos)
                        values.push_back((Real)atof(c.c_str()));
                }

                if (values.size() != 12)
                    serr << "Error in file " << fname << sendl;
                else
                {
                    for(unsigned int i = 0 ; i < 3; i++)
                    {
                        for (unsigned int j = 0 ; j < 3; j++)
                        {
                            mat[i][j] = values[i*3+j];//rotation matrix
                        }
                        mat[i][3] = values[values.size()-1-i];//translation
                    }
                }
            }
        }

        if (!found) serr << "Transformation not found in " << fname << sendl;
        f_affineMatrix.setValue(mat);
    }
    else
    {
        serr << "Could not open file " << fname << sendl << "Matrix set to identity" << sendl;
    }
}

/**************************************************
 * .trm spec:
 * 1st line: 3 values for translation
 * then 3 lines of 3 values for the rotation matrix
 **************************************************/
template <class DataTypes>
void TransformPosition<DataTypes>::getTransfoFromTrm()
{
    std::string fname(this->f_filename.getFullPath());
    sout << "Loading .trm file " << fname << sendl;

    std::ifstream stream(fname.c_str());
    if (stream)
    {
        std::string line;
        unsigned int nbLines = 0;
        Mat4x4 mat(Mat4x4::s_identity);

        while (getline(stream,line))
        {
            if (line == "") continue;
            nbLines++;

            if (nbLines > 4)
            {
                serr << "File with more than 4 lines" << sendl;
                break;
            }

            std::vector<std::string> vLine;

            char *l = new char[line.size()];
            strcpy(l, line.c_str());
            char* p;
            for (p = strtok(l, " "); p; p = strtok(NULL, " "))
                vLine.push_back(std::string(p));
            delete [] l;

            if (vLine.size()>3 )
            {
                for (unsigned int i = 3; i < vLine.size();i++)
                {
                    if (vLine[i]!="")
                    {
                        serr << "Should be a line of 3 values" << sendl;
                        break;
                    }
                }
            }
            else if (vLine.size()<3) {serr << "Should be a line of 3 values" << sendl;continue;}

            if (nbLines == 1)
            {
                //translation vector
                Coord tr;
                for ( unsigned int i = 0; i < std::min((unsigned int)vLine.size(),(unsigned int)3); i++)
                {
                    tr[i] = mat[i][3] = (Real)atof(vLine[i].c_str());
                }
                f_translation.setValue(tr);

            }
            else
            {
                //rotation matrix
                for ( unsigned int i = 0; i < std::min((unsigned int)vLine.size(),(unsigned int)3); i++)
                    mat[nbLines-2][i] = (Real)atof(vLine[i].c_str());
            }

        }
        f_affineMatrix.setValue(mat);
    }
    else
    {
        serr << "Could not open file " << fname << sendl << "Matrix set to identity" << sendl;
    }

}

/**************************************************
 * .txt and .xfm spec:
 * 4 lines of 4 values for an affine matrix
 **************************************************/
template <class DataTypes>
void TransformPosition<DataTypes>::getTransfoFromTxt()
{
    std::string fname(this->f_filename.getFullPath());
    sout << "Loading matrix file " << fname << sendl;

    std::ifstream stream(fname.c_str());
    if (stream)
    {
        std::string line;
        unsigned int nbLines = 0;
        Mat4x4 mat(Mat4x4::s_identity);

        while (getline(stream,line))
        {
            if (line == "") continue;
            nbLines++;

            if (nbLines > 4)
            {
                serr << "Matrix is not 4x4" << sendl;
                break;
            }

            std::vector<std::string> vLine;

            char *l = new char[line.size()];
            strcpy(l, line.c_str());
            char* p;
            for (p = strtok(l, " "); p; p = strtok(NULL, " "))
                vLine.push_back(std::string(p));
            delete [] l;

            if (vLine.size()>4 )
            {
                for (unsigned int i = 4; i < vLine.size();i++)
                {
                    if (vLine[i]!="")
                    {
                        serr << "Matrix is not 4x4." << sendl;
                        break;
                    }
                }
            }
            else if (vLine.size()<4) {serr << "Matrix is not 4x4." << sendl;continue;}

            for ( unsigned int i = 0; i < std::min((unsigned int)vLine.size(),(unsigned int)4); i++)
                mat[nbLines-1][i] = (Real)atof(vLine[i].c_str());
        }
        f_affineMatrix.setValue(mat);

    }
    else
    {
        serr << "Could not open file " << fname << sendl << "Matrix set to identity" << sendl;
    }
}


template <class DataTypes>
void TransformPosition<DataTypes>::update()
{
    selectTransformationMethod();

    helper::ReadAccessor< Data<VecCoord> > in = f_inputX;

    helper::ReadAccessor< Data<Coord> > normal = f_normal;
    helper::ReadAccessor< Data<Coord> > origin = f_origin;
    helper::ReadAccessor< Data<Coord> > translation = f_translation;
    helper::ReadAccessor< Data<Coord> > scale = f_scale;
    helper::ReadAccessor< Data<Coord> > rotation = f_rotation;
    helper::ReadAccessor< Data<Mat4x4> > affineMatrix = f_affineMatrix;
    helper::ReadAccessor< Data<Real> > maxDisplacement = f_maxRandomDisplacement;
    helper::ReadAccessor< Data<long> > seed = f_seed;
    helper::ReadAccessor< Data<SetIndex> > fixedIndices = f_fixedIndices;

    cleanDirty();

    helper::WriteOnlyAccessor< Data<VecCoord> > out = f_outputX;

    out.resize(in.size());
    unsigned int i;
    switch(transformationMethod)
    {
    case PROJECT_ON_PLANE :
        for (i=0; i< in.size(); ++i)
        {
            out[i]=in[i]+normal.ref()*dot((origin.ref()-in[i]),normal.ref());
        }
        break;
    case RANDOM :
    {
        sofa::helper::RandomGenerator rg;
        double dis=(double) maxDisplacement.ref();
        if (seed.ref()!=0)
            rg.initSeed(seed.ref());
        for (i=0; i< in.size(); ++i)
        {

            out[i]=in[i]+Coord((Real)rg.random<double>(-dis,dis),(Real)rg.random<double>(-dis,dis),(Real)rg.random<double>(-dis,dis));
        }
    }
    break;
    case TRANSLATION :
        for (i=0; i< in.size(); ++i)
        {
            out[i]=in[i]+translation.ref();
        }
        break;
    case SCALE :
        for (i=0; i< in.size(); ++i)
        {
            out[i] = origin.ref() + (in[i]-origin.ref()).linearProduct(scale.ref());
        }
        break;
    case SCALE_TRANSLATION :
        for (i=0; i< in.size(); ++i)
        {
            out[i]=in[i].linearProduct(scale.ref()) +translation.ref();
        }
        break;
    case ROTATION :
    {
        sofa::defaulttype::Quaternion q=helper::Quater<Real>::createQuaterFromEuler( rotation.ref()*M_PI/180.0);

        for (i=0; i< in.size(); ++i)
        {
            out[i]=q.rotate(in[i]);
        }
    }
    break;
    case SCALE_ROTATION_TRANSLATION :
    {
        sofa::defaulttype::Quaternion q=helper::Quater<Real>::createQuaterFromEuler( rotation.ref()*M_PI/180.0);

        for (i=0; i< in.size(); ++i)
        {
            out[i]=q.rotate(in[i].linearProduct(scale.ref())) +translation.ref();
        }
        break;
    }
    case AFFINE:
        for (i=0; i< in.size(); ++i)
        {
            Vec4 coord = affineMatrix.ref()*Vec4(in[i], 1);
            if ( fabs(coord[3]) > 1e-10)
                out[i]=coord/coord[3];
        }
        break;
    }
    /// assumes the set of fixed indices is small compared to the whole set
    SetIndex::const_iterator it=fixedIndices.ref().begin();
    for (; it!=fixedIndices.ref().end(); ++it)
    {
        out[*it]=in[*it];
    }

}

template <class DataTypes>
void TransformPosition<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (f_drawInput.getValue())
    {
        helper::ReadAccessor< Data<VecCoord> > in = f_inputX;
        std::vector<sofa::defaulttype::Vector3> points;
        for (unsigned int i=0; i < in.size(); i++)
            points.push_back(in[i]);
        vparams->drawTool()->drawPoints(points, (float)f_pointSize.getValue(), sofa::defaulttype::Vec4f(0.8f, 0.2f, 0.2f, 1.0f));
    }

    if (f_drawOutput.getValue())
    {
        helper::ReadAccessor< Data<VecCoord> > out = f_outputX;
        std::vector<sofa::defaulttype::Vector3> points;
        for (unsigned int i=0; i < out.size(); i++)
            points.push_back(out[i]);
        vparams->drawTool()->drawPoints(points, (float)f_pointSize.getValue(), sofa::defaulttype::Vec4f(0.2f, 0.8f, 0.2f, 1.0f));
    }
}


} // namespace engine

} // namespace component

} // namespace sofa

#endif
