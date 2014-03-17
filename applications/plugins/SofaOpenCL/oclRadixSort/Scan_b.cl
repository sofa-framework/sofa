/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */



//All three kernels run 512 threads per workgroup
//Must be a power of two
#define WORKGROUP_SIZE 256



////////////////////////////////////////////////////////////////////////////////
// Scan codelets
////////////////////////////////////////////////////////////////////////////////
#if(1)
//Naive inclusive scan: O(N * log2(N)) operations
//Allocate 2 * 'size' local memory, initialize the first half
//with 'size' zeros avoiding if(pos >= offset) condition evaluation
//and saving instructions
inline uint scan1Inclusive(uint idata, __local uint *l_Data, uint size)
{
    uint pos = 2 * get_local_id(0) - (get_local_id(0) & (size - 1));
    l_Data[pos] = 0;
    pos += size;
    l_Data[pos] = idata;

    for(uint offset = 1; offset < size; offset <<= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        uint t = l_Data[pos] + l_Data[pos - offset];
        barrier(CLK_LOCAL_MEM_FENCE);
        l_Data[pos] = t;
    }

    return l_Data[pos];
}

inline uint scan1Exclusive(uint idata, __local uint *l_Data, uint size)
{
    return scan1Inclusive(idata, l_Data, size) - idata;
}

#else
#define LOG2_WARP_SIZE 5U
#define      WARP_SIZE (1U << LOG2_WARP_SIZE)

//Almost the same as naiveScan1 but doesn't need barriers
//assuming size <= WARP_SIZE
inline uint warpScanInclusive(uint idata, __local uint *l_Data, uint size)
{
    uint pos = 2 * get_local_id(0) - (get_local_id(0) & (size - 1));
    l_Data[pos] = 0;
    pos += size;
    l_Data[pos] = idata;

    for(uint offset = 1; offset < size; offset <<= 1)
        l_Data[pos] += l_Data[pos - offset];

    return l_Data[pos];
}

inline uint warpScanExclusive(uint idata, __local uint *l_Data, uint size)
{
    return warpScanInclusive(idata, l_Data, size) - idata;
}

inline uint scan1Inclusive(uint idata, __local uint *l_Data, uint size)
{
    if(size > WARP_SIZE)
    {
        //Bottom-level inclusive warp scan
        uint warpResult = warpScanInclusive(idata, l_Data, WARP_SIZE);

        //Save top elements of each warp for exclusive warp scan
        //sync to wait for warp scans to complete (because l_Data is being overwritten)
        barrier(CLK_LOCAL_MEM_FENCE);
        if( (get_local_id(0) & (WARP_SIZE - 1)) == (WARP_SIZE - 1) )
            l_Data[get_local_id(0) >> LOG2_WARP_SIZE] = warpResult;

        //wait for warp scans to complete
        barrier(CLK_LOCAL_MEM_FENCE);
        if( get_local_id(0) < (WORKGROUP_SIZE / WARP_SIZE) )
        {
            //grab top warp elements
            uint val = l_Data[get_local_id(0)];
            //calculate exclsive scan and write back to shared memory
            l_Data[get_local_id(0)] = warpScanExclusive(val, l_Data, size >> LOG2_WARP_SIZE);
        }

        //return updated warp scans with exclusive scan results
        barrier(CLK_LOCAL_MEM_FENCE);
        return warpResult + l_Data[get_local_id(0) >> LOG2_WARP_SIZE];
    }
    else
    {
        return warpScanInclusive(idata, l_Data, size);
    }
}

inline uint scan1Exclusive(uint idata, __local uint *l_Data, uint size)
{
    return scan1Inclusive(idata, l_Data, size) - idata;
}
#endif


//Vector scan: the array to be scanned is stored
//in work-item private memory as uint4
inline uint4 scan4Inclusive(uint4 data4, __local uint *l_Data, uint size)
{
    //Level-0 inclusive scan
    data4.y += data4.x;
    data4.z += data4.y;
    data4.w += data4.z;

    //Level-1 exclusive scan
    uint val = scan1Inclusive(data4.w, l_Data, size / 4) - data4.w;

    return (data4 + (uint4)val);
}

inline uint4 scan4Exclusive(uint4 data4, __local uint *l_Data, uint size)
{
    return scan4Inclusive(data4, l_Data, size) - data4;
}


////////////////////////////////////////////////////////////////////////////////
// Scan kernels
////////////////////////////////////////////////////////////////////////////////
__kernel __attribute__((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
void scanExclusiveLocal1(
    __global uint4 *d_Dst,
    __global uint4 *d_Src,
    __local uint* l_Data,
    uint size
)
{
    //Load data
    uint4 idata4 = d_Src[get_global_id(0)];

    //Calculate exclusive scan
    uint4 odata4  = scan4Exclusive(idata4, l_Data, size);

    //Write back
    d_Dst[get_global_id(0)] = odata4;
}

//Exclusive scan of top elements of bottom-level scans (4 * THREADBLOCK_SIZE)
__kernel __attribute__((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
void scanExclusiveLocal2(
    __global uint *d_Buf,
    __global uint *d_Dst,
    __global uint *d_Src,
    __local uint* l_Data,
    uint N,
    uint arrayLength
)
{
    //Load top elements
    //Convert results of bottom-level scan back to inclusive
    //Skip loads and stores for inactive work-items of the work-group with highest index(pos >= N)
    uint data = 0;
    if(get_global_id(0) < N)
        data =
            d_Dst[(4 * WORKGROUP_SIZE - 1) + (4 * WORKGROUP_SIZE) * get_global_id(0)] +
            d_Src[(4 * WORKGROUP_SIZE - 1) + (4 * WORKGROUP_SIZE) * get_global_id(0)];

    //Compute
    uint odata = scan1Exclusive(data, l_Data, arrayLength);

    //Avoid out-of-bound access
    if(get_global_id(0) < N)
        d_Buf[get_global_id(0)] = odata;
}

//Final step of large-array scan: combine basic inclusive scan with exclusive scan of top elements of input arrays
__kernel __attribute__((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
void uniformUpdate(
    __global uint4 *d_Data,
    __global uint *d_Buf
)
{
    __local uint buf[1];

    uint4 data4 = d_Data[get_global_id(0)];

    if(get_local_id(0) == 0)
        buf[0] = d_Buf[get_group_id(0)];

    barrier(CLK_LOCAL_MEM_FENCE);
    data4 += (uint4)buf[0];
    d_Data[get_global_id(0)] = data4;
}
