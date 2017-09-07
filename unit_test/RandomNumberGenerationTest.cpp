#include "State.h"

#include <gtest/gtest.h>

class RandomNumberGenerationTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        mState.seedRNG(123);

        uniformValue1 = 0.71295532158028085;
        uniformValue2 = 0.42847092502893397;
        uniformValue3 = 0.69088485143772593;
        uniformValue4 = 0.71915030888079545;
    }

    State mState;

    double uniformValue1;
    double uniformValue2;
    double uniformValue3;
    double uniformValue4;
};

TEST_F(RandomNumberGenerationTest, RandomNumberTest) {
    std::mt19937 &generator = mState.getRNG();
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    EXPECT_DOUBLE_EQ(uniformValue1, dist(generator));
    EXPECT_DOUBLE_EQ(uniformValue2, dist(generator));
    EXPECT_DOUBLE_EQ(uniformValue3, dist(generator));
    EXPECT_DOUBLE_EQ(uniformValue4, dist(generator));
}

TEST_F(RandomNumberGenerationTest, MultipleInstancesTest) {
    std::mt19937 &generator1 = mState.getRNG();
    std::mt19937 &generator2 = mState.getRNG();
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    EXPECT_DOUBLE_EQ(uniformValue1, dist(generator1));
    EXPECT_DOUBLE_EQ(uniformValue2, dist(generator2));
    EXPECT_DOUBLE_EQ(uniformValue3, dist(generator1));
    EXPECT_DOUBLE_EQ(uniformValue4, dist(generator2));
}

TEST_F(RandomNumberGenerationTest, ReSeedTest) {
    std::mt19937 &generator1 = mState.getRNG();
    std::mt19937 &generator2 = mState.getRNG();
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    EXPECT_DOUBLE_EQ(uniformValue1, dist(generator1));
    EXPECT_DOUBLE_EQ(uniformValue2, dist(generator1));

    mState.seedRNG(123);

    EXPECT_DOUBLE_EQ(uniformValue1, dist(generator2));
    EXPECT_DOUBLE_EQ(uniformValue2, dist(generator2));
}

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}
