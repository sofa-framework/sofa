/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <ArticulatedSystemPlugin/config.h>

#include <ArticulatedSystemPlugin/bvh/BVHChannels.h>
#include <ArticulatedSystemPlugin/bvh/BVHOffset.h>
#include <ArticulatedSystemPlugin/bvh/BVHMotion.h>
#include <cstdio>		// fopen and friends
#include <cstring>

namespace sofa::helper::io::bvh
{

/**
*	A BVHJoint is a Graph Node that contains :
*		An Offset to set the position relatively to the parent's local frame.
*		Channels describing the local degrees of freedom that can transform the local frame.
*		A Motion that contains a set of key positions. Each of them contains the channels value that must be set at the current timestep.
*		The EndOfSite Flag is set to true if the Joint is a Leaf.
*/
class SOFA_ARTICULATEDSYSTEMPLUGIN_API BVHJoint
{
public:
    BVHJoint(const char *_name, bool _endSite=false, BVHJoint *_parent=nullptr);
    virtual ~BVHJoint();

    void addChild(BVHJoint *j) {children.push_back(j);}
    std::vector<BVHJoint *> &getChildren(void) {return children;}

    void setOffset(BVHOffset *o) {offset = o;}
    BVHOffset *setOffset(void) {return offset;}

    void setChannels(BVHChannels *c) {channels = c;}
    BVHChannels *getChannels(void) {return channels;}

    bool isEndSite(void) {return endSite;}
    void setEndSite(bool isEndSite) {endSite = isEndSite;}

    void setMotion(BVHMotion *m) {motion = m;}
    BVHMotion *getMotion(void)
    {
        if (motion == nullptr)
            motion = new BVHMotion();
        return motion;
    }
    void initMotion(double fTime, unsigned int fCount);

    // Debug Text display
    void debug(int tab=0);

    void display(int);
    void displayInGlobalFrame(void);

    // .dat File creation for each frame
    // Christian Duriez PHD Articulated Rigids File Format
    void dump(char *fName, char *rootJointName=nullptr);
    void dumpPosition(FILE *f, char *s=nullptr);
    void dumpSegment(FILE *f, char *s=nullptr);
    void dumpRotation(FILE *f, char *s=nullptr);
    void dumpRotationLimit(FILE *f, char *s=nullptr);
    void dumpRotationStiffness(FILE *f, char *s=nullptr);
    void dumpPosition(FILE *f, int beginIndex);
    void dumpSegment(FILE *f, int &cpt, int beginIndex);
    void dumpRotation(FILE *f, int &cpt, int beginIndex);
    void dumpRotationLimit(FILE *f, int &cpt);
    void dumpRotationStiffness(FILE *f, int &cpt);

    int getNumJoints(char *s=nullptr);
    int accumulateNumJoints(void);
    int getNumSegments(char *s=nullptr);
    int getId();
    char* getName();
    BVHOffset* getOffset();

    static int lastId;

private:
    BVHOffset *offset;
    BVHChannels* channels;
    BVHMotion* motion;

    BVHJoint* parent;
    std::vector<BVHJoint *> children;

    char name[128];
    bool endSite;
    int id;

    // Transformation matrix in the global frame
    double matrix[16];
};

} // namespace sofa::helper::io::bvh
