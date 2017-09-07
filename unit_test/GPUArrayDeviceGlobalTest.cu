#include "GPUArrayDeviceGlobal.h"

#include <gtest/gtest.h>

#include <bitset>
#include <vector>

__global__ void simpleAdd(int arraySize, float *x, float *y, float *z)
{
    int idx = GETIDX();
    if (idx < arraySize) {
        z[idx] = x[idx] + y[idx];
    }
}

class GPUArrayDeviceGlobalTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        testArray = GPUArrayDeviceGlobal<float>(5);
        testArraySameSize = GPUArrayDeviceGlobal<float>(5);
        testArraySmaller = GPUArrayDeviceGlobal<float>(3);
        testArrayLarger = GPUArrayDeviceGlobal<float>(8);
    }

    GPUArrayDeviceGlobal<float> emptyTestArray;
    GPUArrayDeviceGlobal<float> testArray;
    GPUArrayDeviceGlobal<float> testArraySameSize;
    GPUArrayDeviceGlobal<float> testArraySmaller;
    GPUArrayDeviceGlobal<float> testArrayLarger;

};

TEST_F(GPUArrayDeviceGlobalTest, SizeTest)
{
    EXPECT_EQ(0, emptyTestArray.size());

    GPUArrayDeviceGlobal<float> largeTestArray(12);

    EXPECT_EQ(12, largeTestArray.size());

    largeTestArray = testArray;

    EXPECT_EQ(5, largeTestArray.size());

    largeTestArray = testArrayLarger;

    EXPECT_EQ(8, largeTestArray.size());
}

TEST_F(GPUArrayDeviceGlobalTest, SetDataTest)
{
    std::vector<float> data1;
    data1.push_back(0.1259203131);
    data1.push_back(3.2303401938);
    data1.push_back(9.3241923485);
    data1.push_back(12.543012049);
    data1.push_back(-5.2903432852);

    std::vector<float> data2;
    data2.push_back(-3.2349102049);
    data2.push_back(3.2009445672);
    data2.push_back(10.348291674);
    data2.push_back(7.2390810394);
    data2.push_back(1.4350929888);

    GPUArrayDeviceGlobal<float> resultArray(5);
    std::vector<float> resultVec(5);

    testArray.set(data1.data());
    testArraySameSize.set(data2.data());

    simpleAdd<<<NBLOCK(5), PERBLOCK>>>(resultArray.size(),
                                       testArray.data(),
                                       testArraySameSize.data(),
                                       resultArray.data());

    resultArray.get(resultVec.data());

    EXPECT_FLOAT_EQ(data1.at(0)+data2.at(0),resultVec.at(0));
    EXPECT_FLOAT_EQ(data1.at(1)+data2.at(1),resultVec.at(1));
    EXPECT_FLOAT_EQ(data1.at(2)+data2.at(2),resultVec.at(2));
    EXPECT_FLOAT_EQ(data1.at(3)+data2.at(3),resultVec.at(3));
    EXPECT_FLOAT_EQ(data1.at(4)+data2.at(4),resultVec.at(4));
}

TEST_F(GPUArrayDeviceGlobalTest, MemsetTest)
{
    float val1 = 0;
    float val2 = 0.385215432;
    testArray.memset(val1);
    testArraySameSize.memsetByVal(val2);

    std::vector<float> resultVec1(5);
    std::vector<float> resultVec2(5);
    std::vector<float> resultVec3(5);

    testArray.get(resultVec1.data());
    testArraySameSize.get(resultVec2.data());

    GPUArrayDeviceGlobal<float> resultArray(5);
    simpleAdd<<<NBLOCK(5), PERBLOCK>>>(resultArray.size(),
                                       testArray.data(),
                                       testArraySameSize.data(),
                                       resultArray.data());

    resultArray.get(resultVec3.data());

    for (int i=0; i<5; ++i) {
        EXPECT_FLOAT_EQ(val1,resultVec1.at(i));
        EXPECT_FLOAT_EQ(val2,resultVec2.at(i));
        EXPECT_FLOAT_EQ(val2,resultVec3.at(i));
    }
}

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc,argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}
