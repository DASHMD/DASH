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

__global__ void atmAdd(int arraySize, float *x, float *y)
{
    int idx = GETIDX();
    if (idx < arraySize) {
        atomicAdd(y, x[idx]);
    }
}

__global__ void simpleMult(int arraySize, float *x, float *y, float *z)
{
    int idx = GETIDX();
    if (idx < arraySize) {
        z[idx] = x[idx] * y[idx];
    }
}

class CudaMathTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        testArray1 = GPUArrayDeviceGlobal<float>(5);
        testArray2 = GPUArrayDeviceGlobal<float>(5);
        resultArray = GPUArrayDeviceGlobal<float>(5);

        values1.push_back(0.23498103985);
        values1.push_back(3.23908183929);
        values1.push_back(2.09098737838);
        values1.push_back(4.19042387819);
        values1.push_back(-2.3290200039);

        values2.push_back(10.3489293842);
        values2.push_back(-0.0123498299182);
        values2.push_back(2.30881294928);
        values2.push_back(1.43898892981);
        values2.push_back(4.10000000001);

        testArray1.set(values1.data());
        testArray2.set(values2.data());

        resultVec = std::vector<float>(5);
    }

    GPUArrayDeviceGlobal<float> testArray1;
    GPUArrayDeviceGlobal<float> testArray2;
    GPUArrayDeviceGlobal<float> resultArray;

    std::vector<float> values1;
    std::vector<float> values2;
    std::vector<float> resultVec;
};

TEST_F(CudaMathTest, AdditionTest)
{
    simpleAdd<<<NBLOCK(5), PERBLOCK>>>(resultArray.size(),
                                       testArray1.data(),
                                       testArray2.data(),
                                       resultArray.data());

    resultArray.get(resultVec.data());

    for (int i=0; i<5; ++i) {
        EXPECT_FLOAT_EQ(values1.at(i)+values2.at(i),resultVec.at(i));
    }
}

TEST_F(CudaMathTest, MultiplicationTest)
{
    simpleMult<<<NBLOCK(5), PERBLOCK>>>(resultArray.size(),
                                        testArray1.data(),
                                        testArray2.data(),
                                        resultArray.data());

    resultArray.get(resultVec.data());

    for (int i=0; i<5; ++i) {
        EXPECT_FLOAT_EQ(values1.at(i)*values2.at(i),resultVec.at(i));
    }
}

TEST_F(CudaMathTest, AtomicAddTest)
{
    float valCPU1 = 0;
    float valCPU2 = 0;

    for (int i=0; i<5; ++i) {
        valCPU1 += values1.at(i);
        valCPU2 += values2.at(i);
    }

    resultArray.memset(0);

    atmAdd<<<NBLOCK(5), PERBLOCK>>>(resultArray.size(),
                                   testArray1.data(),
                                   resultArray.data());
    atmAdd<<<NBLOCK(5), PERBLOCK>>>(resultArray.size(),
                                   testArray2.data(),
                                   resultArray.data()+1);

    resultArray.get(resultVec.data());

    EXPECT_FLOAT_EQ(resultVec.at(0), valCPU1);
    EXPECT_FLOAT_EQ(resultVec.at(1), valCPU2);
}

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc,argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}
