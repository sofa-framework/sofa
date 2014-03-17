#define BSIZE 32

typedef struct
{
    float cellWidth;
    float invCellWidth;
    int cellMask;
    float halfCellWidth;
    float invHalfCellWidth;
} GridParams;

// large prime numbers
#define HASH_PX 73856093
#define HASH_PY 19349663
#define HASH_PZ 83492791



Real4 getPos3(__global const Real* pos, int index0, int index)
{
    __local float ftemp[BSIZE*3];
    //return pos[index];
    int index03 = index0*3;
    int index3 = get_local_id(0)*3;
    ftemp[get_local_id(0)] = pos[index03+get_local_id(0)];
    ftemp[get_local_id(0)+BSIZE] = pos[index03+get_local_id(0)+BSIZE];
    ftemp[get_local_id(0)+2*BSIZE] = pos[index03+get_local_id(0)+2*BSIZE];

    barrier(CLK_LOCAL_MEM_FENCE);

    return (float4)(ftemp[index3],ftemp[index3+1],ftemp[index3+2],0);
}


__kernel void computeHashD(
    __global Real* pos,
    __global unsigned int* particleIndex8,
    int offsetIndex,
    __global unsigned int* particleHash8,
    int n,
    GridParams gridParams
)
{
    int index0 = get_local_size(0)*get_group_id(0);
    int index = index0 + get_local_id(0);

    Real4 p = getPos3(pos,index0,index);

    int hgpos_x,hgpos_y,hgpos_z;
    hgpos_x = (int)(p.x * gridParams.invHalfCellWidth);
    hgpos_y = (int)(p.y * gridParams.invHalfCellWidth);
    hgpos_z = (int)(p.z * gridParams.invHalfCellWidth);

    int halfcell = ((hgpos_x&1) + ((hgpos_y&1)<<1) + ((hgpos_z&1)<<2))^7;
    // compute the first cell to be influenced by the particle
    hgpos_x = (hgpos_x-1) >> 1;
    hgpos_y = (hgpos_y-1) >> 1;
    hgpos_z = (hgpos_z-1) >> 1;

    __local unsigned int hx[BSIZE];
    __local unsigned int hy[BSIZE];
    __local unsigned int hz[BSIZE];
    int x = get_local_id(0);

    hx[x] = (HASH_PX*hgpos_x << 3)+halfcell;
    hy[x] = HASH_PY*hgpos_y;
    hz[x] = HASH_PZ*hgpos_z;

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int dH_x,dH_y,dH_z;
    dH_x = (x&1 ? HASH_PX : 0);
    dH_y = (x&2 ? HASH_PY : 0);
    dH_z = (x&4 ? HASH_PZ : 0);
    int index0_8 = index0 << 3;
    for (unsigned int l = x; l < 8*BSIZE; l+=BSIZE)
    {
        particleIndex8[index0_8 + l+offsetIndex] = index0 + (l>>3);
        uint h_x,h_y,h_z;
        h_x = hx[l>>3];
        h_y = hy[l>>3];
        h_z = hz[l>>3];
        int hc = h_x & 7;
        h_x = (h_x>>3) + dH_x;
        h_y += dH_y;
        h_z += dH_z;
        unsigned int hash = ((h_x ^ h_y ^ h_z) & gridParams.cellMask)<<1;
        if (hc != (x&7)) ++hash;
        particleHash8[index0_8 + l] = hash;
    }


    /*	if(index<1000)
    	{
    		particleHash8[get_global_id(0)*3] = (hgpos_x);
    		particleHash8[get_global_id(0)*3+1] = (hgpos_y);
    		particleHash8[get_global_id(0)*3+2] = (hgpos_z);
    	}
    	else
    	{
    		particleHash8[get_global_id(0)*3] = 999;
    		particleHash8[get_global_id(0)*3+1] = 999;
    		particleHash8[get_global_id(0)*3+2] = 999;
    	}*/
}


__kernel void findCellRangeD(
    int index0,
    __global unsigned int* particleHash,
    __global int * cellRange,
    __global int * cellGhost,
    int n
)
{
    unsigned int i = get_global_id(0);
    unsigned int i_local = get_local_id(0);
    __local unsigned int hash[BSIZE];

    if (i < n)
    {
        hash[get_local_id(0)] = particleHash[i];
    }
    else
    {
        hash[get_local_id(0)] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < n)
    {
        bool firstInCell;
        bool firstGhost;
        unsigned int cur = hash[get_local_id(0)];
        if (i == 0)
        {
            firstInCell = true;
            firstGhost = cur&1;
        }
        else
        {
            unsigned int prev;
            if (get_local_id(0) > 0)
                prev = hash[get_local_id(0)-1];
            else
                prev = particleHash[i-1];
            firstInCell = ((prev>>1) != (cur>>1));
            firstGhost = ((prev != cur) && (cur&1));
            if (firstInCell)
            {
                if ((prev>>1) < (cur>>1)-1)
                    cellRange[ (prev>>1)+1 ] =  (index0+i) | (1U<<31);
                if (!(prev&1)) // no ghost particles in previous cell
                    cellGhost[ prev>>1 ] = index0+i;
            }
        }
        if (firstInCell)
            cellRange[ cur>>1 ] = index0+i;
        if (firstGhost)
            cellGhost[ cur>>1 ] = index0+i;
        if (i == n-1)
        {
            cellRange[ (cur>>1)+1 ] = (index0+n) | (1<<31);
            if (!(cur&1))
                cellGhost[ cur>>1 ] = index0+n;
        }
    }
}

