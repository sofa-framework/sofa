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

//----------------------------------------------------------------------------
// Scans each warp in parallel ("warp-scan"), one element per thread.
// uses 2 numElements of shared memory per thread (64 = elements per warp)
//----------------------------------------------------------------------------
#define WARP_SIZE 32
uint scanwarp(uint val, volatile __local uint* sData, int maxlevel)
{
    // The following is the same as 2 * RadixSort::WARP_SIZE * warpId + threadInWarp =
    // 64*(threadIdx.x >> 5) + (threadIdx.x & (RadixSort::WARP_SIZE - 1))
    int localId = get_local_id(0);
    int idx = 2 * localId - (localId & (WARP_SIZE - 1));
    sData[idx] = 0;
    idx += WARP_SIZE;
    sData[idx] = val;

    if (0 <= maxlevel) { sData[idx] += sData[idx - 1]; }
    if (1 <= maxlevel) { sData[idx] += sData[idx - 2]; }
    if (2 <= maxlevel) { sData[idx] += sData[idx - 4]; }
    if (3 <= maxlevel) { sData[idx] += sData[idx - 8]; }
    if (4 <= maxlevel) { sData[idx] += sData[idx -16]; }

    return sData[idx] - val;  // convert inclusive -> exclusive
}

//----------------------------------------------------------------------------
// scan4 scans 4*RadixSort::CTA_SIZE numElements in a block (4 per thread), using
// a warp-scan algorithm
//----------------------------------------------------------------------------
uint4 scan4(uint4 idata, __local uint* ptr)
{

    uint idx = get_local_id(0);

    uint4 val4 = idata;
    uint sum[3];
    sum[0] = val4.x;
    sum[1] = val4.y + sum[0];
    sum[2] = val4.z + sum[1];

    uint val = val4.w + sum[2];

    val = scanwarp(val, ptr, 4);
    barrier(CLK_LOCAL_MEM_FENCE);

    if ((idx & (WARP_SIZE - 1)) == WARP_SIZE - 1)
    {
        ptr[idx >> 5] = val + val4.w + sum[2];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (idx < WARP_SIZE)
        ptr[idx] = scanwarp(ptr[idx], ptr, 2);

    barrier(CLK_LOCAL_MEM_FENCE);

    val += ptr[idx >> 5];

    val4.x = val;
    val4.y = val + sum[0];
    val4.z = val + sum[1];
    val4.w = val + sum[2];

    return val4;
}

#ifdef MAC
__kernel uint4 rank4(uint4 preds, __local uint* sMem)
#else
uint4 rank4(uint4 preds, __local uint* sMem)
#endif
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);

    uint4 address = scan4(preds, sMem);

    __local uint numtrue[1];
    if (localId == localSize - 1)
    {
        numtrue[0] = address.w + preds.w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint4 rank;
    int idx = localId*4;
    rank.x = (preds.x) ? address.x : numtrue[0] + idx - address.x;
    rank.y = (preds.y) ? address.y : numtrue[0] + idx + 1 - address.y;
    rank.z = (preds.z) ? address.z : numtrue[0] + idx + 2 - address.z;
    rank.w = (preds.w) ? address.w : numtrue[0] + idx + 3 - address.w;

    return rank;
}

void radixSortBlockKeysOnly(uint4 *key, uint nbits, uint startbit, __local uint* sMem)
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);

    for(uint shift = startbit; shift < (startbit + nbits); ++shift)
    {
        uint4 lsb;
        lsb.x = !(((*key).x >> shift) & 0x1);
        lsb.y = !(((*key).y >> shift) & 0x1);
        lsb.z = !(((*key).z >> shift) & 0x1);
        lsb.w = !(((*key).w >> shift) & 0x1);

        uint4 r;

        r = rank4(lsb, sMem);

        // This arithmetic strides the ranks across 4 CTA_SIZE regions
        sMem[(r.x & 3) * localSize + (r.x >> 2)] = (*key).x;
        sMem[(r.y & 3) * localSize + (r.y >> 2)] = (*key).y;
        sMem[(r.z & 3) * localSize + (r.z >> 2)] = (*key).z;
        sMem[(r.w & 3) * localSize + (r.w >> 2)] = (*key).w;
        barrier(CLK_LOCAL_MEM_FENCE);

        // The above allows us to read without 4-way bank conflicts:
        (*key).x = sMem[localId];
        (*key).y = sMem[localId +     localSize];
        (*key).z = sMem[localId + 2 * localSize];
        (*key).w = sMem[localId + 3 * localSize];

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void radixSortBlocksKeysOnly(__global uint4* keysIn,
        __global uint4* keysOut,
        uint nbits,
        uint startbit,
        uint numElements,
        uint totalBlocks,
        __local uint* sMem)
{
    int globalId = get_global_id(0);

    uint4 key;
    key = keysIn[globalId];

    barrier(CLK_LOCAL_MEM_FENCE);

    radixSortBlockKeysOnly(&key, nbits, startbit, sMem);

    keysOut[globalId] = key;
}

//----------------------------------------------------------------------------
// Given an array with blocks sorted according to a 4-bit radix group, each
// block counts the number of keys that fall into each radix in the group, and
// finds the starting offset of each radix in the block.  It then writes the radix
// counts to the counters array, and the starting offsets to the blockOffsets array.
//
// Template parameters are used to generate efficient code for various special cases
// For example, we have to handle arrays that are a multiple of the block size
// (fullBlocks) differently than arrays that are not. "loop" is used when persistent
// CTAs are used.
//
// By persistent CTAs we mean that we launch only as many thread blocks as can
// be resident in the GPU and no more, rather than launching as many threads as
// we have elements. Persistent CTAs loop over blocks of elements until all work
// is complete.  This can be faster in some cases.  In our tests it is faster
// for large sorts (and the threshold is higher on compute version 1.1 and earlier
// GPUs than it is on compute version 1.2 GPUs.
//
//----------------------------------------------------------------------------
__kernel void findRadixOffsets(__global uint2* keys,
        __global uint* counters,
        __global uint* blockOffsets,
        uint startbit,
        uint numElements,
        uint totalBlocks,
        __local uint* sRadix1)
{
    __local uint  sStartPointers[16];

    uint groupId = get_group_id(0);
    uint localId = get_local_id(0);
    uint groupSize = get_local_size(0);

    uint2 radix2;

    radix2 = keys[get_global_id(0)];


    sRadix1[2 * localId]     = (radix2.x >> startbit) & 0xF;
    sRadix1[2 * localId + 1] = (radix2.y >> startbit) & 0xF;

    // Finds the position where the sRadix1 entries differ and stores start
    // index for each radix.
    if(localId < 16)
    {
        sStartPointers[localId] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if((localId > 0) && (sRadix1[localId] != sRadix1[localId - 1]) )
    {
        sStartPointers[sRadix1[localId]] = localId;
    }
    if(sRadix1[localId + groupSize] != sRadix1[localId + groupSize - 1])
    {
        sStartPointers[sRadix1[localId + groupSize]] = localId + groupSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(localId < 16)
    {
        blockOffsets[groupId*16 + localId] = sStartPointers[localId];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute the sizes of each block.
    if((localId > 0) && (sRadix1[localId] != sRadix1[localId - 1]) )
    {
        sStartPointers[sRadix1[localId - 1]] =
            localId - sStartPointers[sRadix1[localId - 1]];
    }
    if(sRadix1[localId + groupSize] != sRadix1[localId + groupSize - 1] )
    {
        sStartPointers[sRadix1[localId + groupSize - 1]] =
            localId + groupSize - sStartPointers[sRadix1[localId + groupSize - 1]];
    }


    if(localId == groupSize - 1)
    {
        sStartPointers[sRadix1[2 * groupSize - 1]] =
            2 * groupSize - sStartPointers[sRadix1[2 * groupSize - 1]];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(localId < 16)
    {
        counters[localId * totalBlocks + groupId] = sStartPointers[localId];
    }
}

// a naive scan routine that works only for array that
// can fit into a single block, just for debugging purpose,
// not used in the sort now
__kernel void scanNaive(__global uint *g_odata,
        __global uint *g_idata,
        uint n,
        __local uint* temp)
{

    int localId = get_local_id(0);

    int pout = 0;
    int pin = 1;

    // Cache the computational window in shared memory
    temp[pout*n + localId] = (localId > 0) ? g_idata[localId-1] : 0;

    for (int offset = 1; offset < n; offset *= 2)
    {
        pout = 1 - pout;
        pin  = 1 - pout;
        barrier(CLK_LOCAL_MEM_FENCE);

        temp[pout*n+localId] = temp[pin*n+localId];

        if (localId >= offset)
            temp[pout*n+localId] += temp[pin*n+localId - offset];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    g_odata[localId] = temp[pout*n+localId];
}

//----------------------------------------------------------------------------
// reorderData shuffles data in the array globally after the radix offsets
// have been found. On compute version 1.1 and earlier GPUs, this code depends
// on RadixSort::CTA_SIZE being 16 * number of radices (i.e. 16 * 2^nbits).
//
// On compute version 1.1 GPUs ("manualCoalesce=true") this function ensures
// that all writes are coalesced using extra work in the kernel.  On later
// GPUs coalescing rules have been relaxed, so this extra overhead hurts
// performance.  On these GPUs we set manualCoalesce=false and directly store
// the results.
//
// Template parameters are used to generate efficient code for various special cases
// For example, we have to handle arrays that are a multiple of the block size
// (fullBlocks) differently than arrays that are not.  "loop" is used when persistent
// CTAs are used.
//
// By persistent CTAs we mean that we launch only as many thread blocks as can
// be resident in the GPU and no more, rather than launching as many threads as
// we have elements. Persistent CTAs loop over blocks of elements until all work
// is complete.  This can be faster in some cases.  In our tests it is faster
// for large sorts (and the threshold is higher on compute version 1.1 and earlier
// GPUs than it is on compute version 1.2 GPUs.
//----------------------------------------------------------------------------
__kernel void reorderDataKeysOnly(__global uint  *outKeys,
        __global uint2  *keys,
        __global uint  *blockOffsets,
        __global uint  *offsets,
        __global uint  *sizes,
        uint startbit,
        uint numElements,
        uint totalBlocks,
        __local uint2* sKeys2)
{
    __local uint sOffsets[16];
    __local uint sBlockOffsets[16];

    __local uint *sKeys1 = (__local uint*)sKeys2;

    uint groupId = get_group_id(0);

    uint globalId = get_global_id(0);
    uint localId = get_local_id(0);
    uint groupSize = get_local_size(0);

    sKeys2[localId]   = keys[globalId];

    if(localId < 16)
    {
        sOffsets[localId]      = offsets[localId * totalBlocks + groupId];
        sBlockOffsets[localId] = blockOffsets[groupId * 16 + localId];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint radix = (sKeys1[localId] >> startbit) & 0xF;
    uint globalOffset = sOffsets[radix] + localId - sBlockOffsets[radix];

    if (globalOffset < numElements)
    {
        outKeys[globalOffset]   = sKeys1[localId];
    }

    radix = (sKeys1[localId + groupSize] >> startbit) & 0xF;
    globalOffset = sOffsets[radix] + localId + groupSize - sBlockOffsets[radix];

    if (globalOffset < numElements)
    {
        outKeys[globalOffset]   = sKeys1[localId + groupSize];
    }


}
