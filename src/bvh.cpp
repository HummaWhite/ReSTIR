#include "bvh.h"

/**
* MTBVH builder
* What is MTBVH? It's a variant of Bounding Volume Hierarchy invented by Toshiya Hachisuka.
* MTBVH enables stackless BVH traversal on GPU, saving many registers that were used in stack-based
* traversal. It's simple and efficient.
* https://cs.uwaterloo.ca/~thachisu/tdf2015.pdf
*/
int BVHBuilder::build(
    const std::vector<glm::vec3>& vertices,
    std::vector<AABB>& boundingBoxes,
    std::vector<std::vector<MTBVHNode>>& BVHNodes
) {
    std::cout << "[BVH building...]" << std::endl;
    int numPrims = vertices.size() / 3;
    int BVHSize = numPrims * 2 - 1;

    std::vector<PrimInfo> primInfo(numPrims);
    std::vector<NodeInfo> nodeInfo(BVHSize);
    boundingBoxes.resize(BVHSize);

    for (int i = 0; i < numPrims; i++) {
        primInfo[i].primId = i;
        primInfo[i].bound = AABB(vertices[i * 3 + 0], vertices[i * 3 + 1], vertices[i * 3 + 2]);
        primInfo[i].center = primInfo[i].bound.center();
    }

    // Use array stack just for faster
    std::vector<BuildInfo> stack(BVHSize);
    int stackTop = 0;
    stack[stackTop++] = { 0, 0, numPrims - 1 };

    const int NumBuckets = 16;
    // Using non-recursive approach to build BVH data directly flattened
    int depth = 0;
    while (stackTop) {
        depth = std::max(depth, stackTop);
        stackTop--;
        int offset = stack[stackTop].offset;
        int start = stack[stackTop].start;
        int end = stack[stackTop].end;

        int numSubPrims = end - start + 1;
        int nodeSize = numSubPrims * 2 - 1;
        bool isLeaf = nodeSize == 1;
        nodeInfo[offset] = { isLeaf, isLeaf ? primInfo[start].primId : nodeSize };

        AABB nodeBound, centerBound;
        for (int i = start; i <= end; i++) {
            nodeBound = nodeBound(primInfo[i].bound);
            centerBound = centerBound(primInfo[i].center);
        }
        boundingBoxes[offset] = nodeBound;

        /*std::cout << std::setw(4) << nodeInfo[offset].primIdOrSize << " " << offset << 
            " " << start << " " << end << " " << nodeBound.toString() << "\n";*/

        if (isLeaf) {
            continue;
        }

        int splitAxis = centerBound.longestAxis();

        if (nodeSize == 2) {
            if (primInfo[start].center[splitAxis] > primInfo[end].center[splitAxis]) {
                std::swap(primInfo[start], primInfo[end]);
            }
            boundingBoxes[offset + 1] = primInfo[start].bound;
            boundingBoxes[offset + 2] = primInfo[end].bound;
            nodeInfo[offset + 1] = { true, primInfo[start].primId };
            nodeInfo[offset + 2] = { true, primInfo[end].primId };
        }

        AABB bucketBounds[NumBuckets];
        int bucketCounts[NumBuckets];
        memset(bucketCounts, 0, sizeof(bucketCounts));

        float dimMin = centerBound.pMin[splitAxis];
        float dimMax = centerBound.pMax[splitAxis];

        for (int i = start; i <= end; i++) {
            int bid = glm::clamp(int((primInfo[i].center[splitAxis] - dimMin) / (dimMax - dimMin) * NumBuckets),
                0, NumBuckets - 1);
            bucketBounds[bid] = bucketBounds[bid](primInfo[i].bound);
            bucketCounts[bid]++;
        }

        AABB lBounds[NumBuckets];
        AABB rBounds[NumBuckets];
        int countPrefix[NumBuckets];

        lBounds[0] = bucketBounds[0];
        rBounds[NumBuckets - 1] = bucketBounds[NumBuckets - 1];
        countPrefix[0] = bucketCounts[0];
        for (int i = 1, j = NumBuckets - 2; i < NumBuckets; i++, j--) {
            lBounds[i] = lBounds[i](bucketBounds[i - 1]);
            rBounds[j] = rBounds[j](bucketBounds[j + 1]);
            countPrefix[i] = countPrefix[i - 1] + bucketCounts[i];
        }

        float minSAH = FLT_MAX;
        int divBucket = 0;
        for (int i = 0; i < NumBuckets - 1; i++) {
            float SAH = glm::mix(lBounds[i].surfaceArea(), rBounds[i + 1].surfaceArea(),
                float(countPrefix[i]) / numSubPrims);
            if (SAH < minSAH) {
                minSAH = SAH;
                divBucket = i;
            }
        }

        std::vector<PrimInfo> temp(numSubPrims);
        memcpy(temp.data(), primInfo.data() + start, numSubPrims * sizeof(PrimInfo));

        int divPrim = start, divEnd = end;
        for (int i = 0; i < numSubPrims; i++) {
            int bid = glm::clamp(int((temp[i].center[splitAxis] - dimMin) / (dimMax - dimMin) * NumBuckets),
                0, NumBuckets - 1);
            (bid <= divBucket ? primInfo[divPrim++] : primInfo[divEnd--]) = temp[i];
        }
        divPrim = glm::clamp(divPrim - 1, start, end - 1);
        int lSize = 2 * (divPrim - start + 1) - 1;

        stack[stackTop++] = { offset + 1 + lSize, divPrim + 1, end };
        stack[stackTop++] = { offset + 1, start, divPrim };
    }
    std::cout << "\t[Size = " << BVHSize << ", depth = " << depth << "]" << std::endl;
    buildMTBVH(boundingBoxes, nodeInfo, BVHSize, BVHNodes);
    return BVHSize;
}

void BVHBuilder::buildMTBVH(
    const std::vector<AABB>& boundingBoxes,
    const std::vector<NodeInfo>& nodeInfo,
    int BVHSize,
    std::vector<std::vector<MTBVHNode>>& BVHNodes
) {
    BVHNodes.resize(6);
    for (auto& node : BVHNodes) {
        node.resize(BVHSize);
    }
    std::vector<int> stack(BVHSize);

    /*
    for (auto& info : nodeInfo) {
        std::cout << (info.isLeaf ? info.primIdOrSize : 0) << " ";
    }
    std::cout << "\n";
    for (auto& info : nodeInfo) {
        std::cout << (info.isLeaf ? 0 : info.primIdOrSize) << " ";
    }
    std::cout << "\n\n";
    */

    for (int i = 0; i < 6; i++) {
        auto& nodes = BVHNodes[i];
        nodes.resize(BVHSize);

        int stackTop = 0;
        stack[stackTop++] = 0;
        int nodeIdNew = 0;
        while (stackTop) {
            int nodeIdOrig = stack[--stackTop];
            bool isLeaf = nodeInfo[nodeIdOrig].isLeaf;
            int nodeSize = isLeaf ? 1 : nodeInfo[nodeIdOrig].primIdOrSize;

            nodes[nodeIdNew] = {
                isLeaf ? nodeInfo[nodeIdOrig].primIdOrSize : NullPrimitive,
                nodeIdOrig,
                nodeIdNew + nodeSize
            };
            nodeIdNew++;

            if (isLeaf) {
                continue;
            }
            bool isLeftLeaf = nodeInfo[nodeIdOrig + 1].isLeaf;
            int leftSize = isLeftLeaf ? 1 : nodeInfo[nodeIdOrig + 1].primIdOrSize;

            int left = nodeIdOrig + 1;
            int right = nodeIdOrig + 1 + leftSize;

            int dim = i / 2;
            bool lesser = i & 1;
            if ((boundingBoxes[left].center()[dim] < boundingBoxes[right].center()[dim]) ^ lesser) {
                std::swap(left, right);
            }

            stack[stackTop++] = right;
            stack[stackTop++] = left;
        }
    }

    /*for (const auto& nodes : BVHNodes) {
        for (const auto& node : nodes) {
            std::cout << std::setw(3) << node.primitiveId << " " << node.nextNodeIfMiss << " " <<
                vec3ToString(boundingBoxes[node.boundingBoxId].center()) << "\n";
        }
        std::cout << "\n";
    }*/
}