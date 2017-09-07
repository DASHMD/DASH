#include "Vector.h"

#include <gtest/gtest.h>

class VectorTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        vDouble1 = Vector(1.1,2.2,3.3);
        vDouble2 = Vector(-2.3,1.5,-0.1);
        vInt1 = VectorInt(1,2,3);
        vInt2 = VectorInt(-2,1,0);
    }

    Vector emptyVector;
    Vector vDouble1;
    Vector vDouble2;
    Vector vInt1;
    Vector vInt2;
};

TEST_F(VectorTest, DefaultEntriesTest) {
    EXPECT_DOUBLE_EQ(0, emptyVector[0]);
    EXPECT_DOUBLE_EQ(0, emptyVector[1]);
    EXPECT_DOUBLE_EQ(0, emptyVector[2]);
}

TEST_F(VectorTest, AccessElementsTest) {
    EXPECT_DOUBLE_EQ(1.1, vDouble1[0]);
    EXPECT_DOUBLE_EQ(2.2, vDouble1[1]);
    EXPECT_DOUBLE_EQ(3.3, vDouble1[2]);

    EXPECT_DOUBLE_EQ(1.1, vDouble1.get(0));
    EXPECT_DOUBLE_EQ(2.2, vDouble1.get(1));
    EXPECT_DOUBLE_EQ(3.3, vDouble1.get(2));

    EXPECT_EQ(1, vInt1[0]);
    EXPECT_EQ(2, vInt1[1]);
    EXPECT_EQ(3, vInt1[2]);

    EXPECT_EQ(1, vInt1.get(0));
    EXPECT_EQ(2, vInt1.get(1));
    EXPECT_EQ(3, vInt1.get(2));

    vDouble1.set(0, 4.3);
    vInt1.set(2, 9);

    EXPECT_DOUBLE_EQ(4.3, vDouble1[0]);
    EXPECT_EQ(9, vInt1[2]);
}

TEST_F(VectorTest, SetValuesZero) {
    // Double valued vector
    EXPECT_DOUBLE_EQ(1.1, vDouble1[0]);
    EXPECT_DOUBLE_EQ(2.2, vDouble1[1]);
    EXPECT_DOUBLE_EQ(3.3, vDouble1[2]);

    vDouble1.zero();

    EXPECT_DOUBLE_EQ(0.0, vDouble1[0]);
    EXPECT_DOUBLE_EQ(0.0, vDouble1[1]);
    EXPECT_DOUBLE_EQ(0.0, vDouble1[2]);

    // Int valued vector
    EXPECT_EQ(1, vInt1[0]);
    EXPECT_EQ(2, vInt1[1]);
    EXPECT_EQ(3, vInt1[2]);

    vInt1.zero();

    EXPECT_EQ(0, vInt1[0]);
    EXPECT_EQ(0, vInt1[1]);
    EXPECT_EQ(0, vInt1[2]);
}

TEST_F(VectorTest, SumTest) {
    EXPECT_DOUBLE_EQ(6.6, vDouble1.sum());
    EXPECT_DOUBLE_EQ(-0.9, vDouble2.sum());
    EXPECT_EQ(6, vInt1.sum());
    EXPECT_EQ(-1, vInt2.sum());
}

TEST_F(VectorTest, ProdTest) {
    EXPECT_DOUBLE_EQ(7.986, vDouble1.prod());
    EXPECT_DOUBLE_EQ(0.345, vDouble2.prod());
    EXPECT_EQ(6, vInt1.prod());
    EXPECT_EQ(0, vInt2.prod());
}

TEST_F(VectorTest, VectorMultiplicationIntWithDouble) {
    Vector result1 = vDouble1 * vInt1;
    Vector result2 = vInt2 * vDouble2;

    EXPECT_DOUBLE_EQ(1.1, result1[0]);
    EXPECT_DOUBLE_EQ(4.4, result1[1]);
    EXPECT_DOUBLE_EQ(9.9, result1[2]);

    EXPECT_DOUBLE_EQ(4.6, result2[0]);
    EXPECT_DOUBLE_EQ(1.5, result2[1]);
    EXPECT_DOUBLE_EQ(0.0, result2[2]);
}

TEST_F(VectorTest, VectorAdditionIntWithDouble) {
    Vector result1 = vDouble1 + vInt1;
    Vector result2 = vInt2 + vDouble2;

    EXPECT_DOUBLE_EQ(2.1, result1[0]);
    EXPECT_DOUBLE_EQ(4.2, result1[1]);
    EXPECT_DOUBLE_EQ(6.3, result1[2]);

    EXPECT_DOUBLE_EQ(-4.3, result2[0]);
    EXPECT_DOUBLE_EQ( 2.5, result2[1]);
    EXPECT_DOUBLE_EQ(-0.1, result2[2]);
}

TEST_F(VectorTest, VectorCopyDoubleAndInt) {
    Vector copyDoubleToDouble1(vDouble1);
    VectorInt copyIntToInt1(vInt1);
    Vector copyIntToDouble1(vInt2);
    VectorInt copyDoubleToInt1(vDouble2);

    Vector copyDoubleToDouble2 = vDouble1;
    VectorInt copyIntToInt2 = vInt1;
    Vector copyIntToDouble2 = vInt2;
    VectorInt copyDoubleToInt2 = vDouble2;

    EXPECT_DOUBLE_EQ(1.1, copyDoubleToDouble1[0]);
    EXPECT_DOUBLE_EQ(2.2, copyDoubleToDouble1[1]);
    EXPECT_DOUBLE_EQ(3.3, copyDoubleToDouble1[2]);

    EXPECT_EQ(1, copyIntToInt1[0]);
    EXPECT_EQ(2, copyIntToInt1[1]);
    EXPECT_EQ(3, copyIntToInt1[2]);

    EXPECT_DOUBLE_EQ(-2.0, copyIntToDouble1[0]);
    EXPECT_DOUBLE_EQ( 1.0, copyIntToDouble1[1]);
    EXPECT_DOUBLE_EQ( 0.0, copyIntToDouble1[2]);

    EXPECT_EQ(-2, copyDoubleToInt1[0]);
    EXPECT_EQ( 1, copyDoubleToInt1[1]);
    EXPECT_EQ( 0, copyDoubleToInt1[2]);

    EXPECT_DOUBLE_EQ(1.1, copyDoubleToDouble2[0]);
    EXPECT_DOUBLE_EQ(2.2, copyDoubleToDouble2[1]);
    EXPECT_DOUBLE_EQ(3.3, copyDoubleToDouble2[2]);

    EXPECT_EQ(1, copyIntToInt2[0]);
    EXPECT_EQ(2, copyIntToInt2[1]);
    EXPECT_EQ(3, copyIntToInt2[2]);

    EXPECT_DOUBLE_EQ(-2.0, copyIntToDouble2[0]);
    EXPECT_DOUBLE_EQ( 1.0, copyIntToDouble2[1]);
    EXPECT_DOUBLE_EQ( 0.0, copyIntToDouble2[2]);

    EXPECT_EQ(-2, copyDoubleToInt2[0]);
    EXPECT_EQ( 1, copyDoubleToInt2[1]);
    EXPECT_EQ( 0, copyDoubleToInt2[2]);
}

TEST_F(VectorTest, TestEquality) {
    EXPECT_FALSE(vDouble1 == vInt1);
    EXPECT_FALSE(vDouble2 == vInt2);

    Vector flippedDouble = vDouble1;
    flippedDouble[1] = -1*vDouble1[1];

    EXPECT_FALSE(flippedDouble == vDouble1);

    Vector vDouble(1.0, 2.0, 3.0);
    VectorInt vInt(1,2,3);

    EXPECT_TRUE(vDouble == vInt);
}

TEST_F(VectorTest, TestComparison) {
    EXPECT_TRUE(vDouble1 > vInt1);
    EXPECT_TRUE(vDouble1 > vDouble2);
    EXPECT_TRUE(vDouble2 < vInt2);

    // We do this trying to reduce numerical accuricy
    Vector similarDouble(1000000000001.0,1000000000002.0,1000000000003.0);
    similarDouble -= Vector(1000000000000.0, 1000000000000.0, 1000000000000.0);

    VectorInt similarInt(1,2,3);

    EXPECT_FALSE(similarDouble < similarInt);
    EXPECT_FALSE(similarDouble > similarInt);
}

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc,argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}
