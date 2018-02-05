/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <SofaMiscCollision/SpatialGridPointModel.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseCollision/CubeModel.h>
#include <sofa/core/ObjectFactory.h>
#include <vector>
#include <sofa/helper/system/gl.h>

namespace sofa
{

namespace component
{

namespace collision
{

SOFA_DECL_CLASS(SpatialGridPointModel)

int SpatialGridPointModelClass = core::RegisterObject("Collision model which represents a set of points, spatially grouped using a SpatialGridContainer")
        .add< SpatialGridPointModel >()
        ;

SpatialGridPointModel::SpatialGridPointModel()
    : d_leafScale(initData(&d_leafScale,0,"leafScale","at which level should the first cube layer be constructed.\nNote that this must not be greater than GRIDDIM_LOG2"))
    , grid(NULL)
{
}

void SpatialGridPointModel::init()
{
    this->PointModel::init();
    this->getContext()->get(grid);

    if (grid==NULL)
    {
        serr <<"SpatialGridPointModel requires a Vec3 SpatialGridContainer" << sendl;
        return;
    }
}

bool SpatialGridPointModel::OctreeSorter::operator()(const Grid::Key& k1, const Grid::Key& k2)
{
    for (int scale = root_shift; scale >= 0; --scale)
    {
        for (int c=k1.size()-1; c>=0; --c)
        {
            if ((k1[c]>>scale) < (k2[c]>>scale))
                return true;
            if ((k1[c]>>scale) > (k2[c]>>scale))
                return false;
        }
    }
    // they are equal
    return false;
}

void SpatialGridPointModel::computeBoundingTree(int maxDepth)
{
    if (!grid)
    {
        this->PointModel::computeBoundingTree(maxDepth);
        return;
    }
    int lscale = d_leafScale.getValue();
    if (lscale > Grid::GRIDDIM_LOG2) lscale = Grid::GRIDDIM_LOG2;
    int ldim = (1<<lscale);
    int nleaf = Grid::GRIDDIM/ldim;
    CubeModel* cubeModel = createPrevious<CubeModel>();
    const int npoints = mstate->getSize();
    bool updated = false;
    if (npoints != size)
    {
        resize(npoints);
        updated = true;
    }
    if (updated) cubeModel->resize(0);
    if (!isMoving() && !cubeModel->empty() && !updated) return; // No need to recompute BBox if immobile

    std::vector<OctreeCell> cells;
    Grid* g = grid->getGrid();
    Grid::const_iterator itgbegin = g->gridBegin();
    Grid::const_iterator itgend = g->gridEnd();
    //sout << "input: ";
    bool sorted = true;
    for (Grid::const_iterator itg = itgbegin; itg != itgend; ++itg)
    {
        Grid::Key k = itg->first;
        Grid::Grid* g = itg->second;
        if (g->empty) continue;
        for (int z0 = 0; z0<nleaf; z0++)
            for (int y0 = 0; y0<nleaf; y0++)
                for (int x0 = 0; x0<nleaf; x0++)
                {
                    int pfirst = -1;
                    int plast = -1;
                    Grid::Key k2;
                    k2[0] = k[0]*nleaf + x0;
                    k2[1] = k[1]*nleaf + y0;
                    k2[2] = k[2]*nleaf + z0;
                    for (int z = 0; z<ldim; z++)
                        for (int y = 0; y<ldim; y++)
                            for (int x = 0; x<ldim; x++)
                            {
                                Grid::Cell* c = g->cell+((z0*ldim+z)*Grid::DZ+(y0*ldim+y)*Grid::DY+(x0*ldim+x)*Grid::DX);
                                if (!c->plist.empty())
                                {
                                    if (pfirst==-1)
                                        pfirst = c->plist.front().index;
                                    else if (c->plist.front().index != plast+1)
                                        sorted = false;
                                    plast = c->plist.back().index;
                                    if (c->plist.back().index - c->plist.front().index +1 != (int)c->plist.size())
                                        sorted = false;
                                }
                            }
                    if (pfirst == -1) continue;
                    cells.push_back(OctreeCell(k2, pfirst, plast));
                    //sout << "  " << k2;
                }
        /*
        int pfirst = -1;
        int plast = -1;
        for (int i=0; i<Grid::NCELL; ++i)
        {
            Grid::Cell* c = g->cell+i;
            if (!c->plist.empty())
            {
            pfirst = c->plist.front().index;
            break;
            }
        }
        if (pfirst == -1) continue; // empty
        for (int i=Grid::NCELL-1; i>=0; --i)
        {
            Grid::Cell* c = g->cell+i;
            if (!c->plist.empty())
            {
            plast = c->plist.back().index;
            break;
            }
        }
        cells.push_back(OctreeCell(k, pfirst, plast));
        //sout << "  " << k;
        */
    }
    if (!sorted)
    {
        serr << "ERROR(SpatialGridPointModel): points are not sorted in spatial grid."<<sendl;
    }
    //sout << sendl;
    cubeModel->resize(cells.size());
    if (cells.empty()) return;
    OctreeSorter s(maxDepth);
    defaulttype::Vector3::value_type cellSize = g->getCellWidth()*ldim; // *GRIDDIM;
    std::sort(cells.begin(), cells.end(), s);

    //sout << "sorted: ";
    for (unsigned int i=0; i<cells.size(); i++)
    {
        Grid::Key k = cells[i].k;
        //sout << "  " << k;
        int pfirst = cells[i].pfirst;
        int plast = cells[i].plast;
        defaulttype::Vector3 minElem, maxElem;
        for (unsigned int c=0; c<k.size(); ++c)
        {
            minElem[c] = k[c]*cellSize;
            maxElem[c] = (k[c]+1)*cellSize;
        }
        cubeModel->setLeafCube(i, std::make_pair(Iterator(this,pfirst),Iterator(this,plast+1)), minElem, maxElem); // define the bounding box of the current cell
    }
    //sout << sendl;
    //cubeModel->computeBoundingTree(maxDepth);
    int depth = 0;
    while (depth < maxDepth && cells.size() > 8)
    {
        msg_info() << "SpatialGridPointModel: cube depth "<<depth<<": "<<cells.size()<<" cells ("<<(size*100/cells.size())*0.01<<" points/cell)."<<sendl;
        // compact cells inplace
        int parent = -1;
        for (unsigned int i=0; i<cells.size(); ++i)
        {
            Grid::Key k = cells[i].k;
            //sout << "  " << k;
            for (unsigned int c=0; c<k.size(); ++c)
                k[c] >>= 1;
            if (parent == -1 || !(k == cells[parent].k))
            {
                // new parent
                //sout << "->"<<k;
                ++parent;
                cells[parent].k = k;
                cells[parent].pfirst = i;
                cells[parent].plast = i;
            }
            else
            {
                // continuing
                cells[parent].plast = i;
            }
        }
        //sout << sendl;
        if (cells.size() > (unsigned int)parent+1)
        {
            cells.resize(parent+1);
            CubeModel* prevCubeModel = cubeModel;
            cubeModel = cubeModel->createPrevious<CubeModel>();
            cubeModel->resize(0);
            for (unsigned int i=0; i<cells.size(); ++i)
            {
                //Grid::Key k = cells[i].k;
                int pfirst = cells[i].pfirst;
                int plast = cells[i].plast;
                Cube cfirst(prevCubeModel, pfirst);
                Cube clast(prevCubeModel, plast);
                cubeModel->addCube(Cube(prevCubeModel,pfirst),Cube(prevCubeModel,plast+1));
            }
        }
        ++depth;
    }
    CubeModel* root = cubeModel->createPrevious<CubeModel>();
    while (dynamic_cast<CubeModel*>(root->getPrevious()) != NULL)
    {
        root = dynamic_cast<CubeModel*>(root->getPrevious());
    }
    root->resize(0);
    root->addCube(Cube(cubeModel,0), Cube(cubeModel,cubeModel->getSize()));
}

} // namespace collision

} // namespace component

} // namespace sofa

