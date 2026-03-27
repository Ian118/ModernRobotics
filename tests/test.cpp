#include <iostream>
#include <Eigen/Dense>
#include "ModernRobotics"
#include "gtest/gtest.h"

#define M_PI 3.14159265358979323846 /* pi */

TEST(MRTest, VecToSO3Test)
{
	Eigen::Vector3d vec(1, 2, 3);
	Eigen::Matrix3d result(3, 3);
	result << 0, -3, 2, 3, 0, -1, -2, 1, 0;
	EXPECT_EQ(result, mr::VecToso3(vec));
}

TEST(MRTest, JacobianSpaceTest)
{
	Eigen::Matrix<double, 6, 3> s_list;
	s_list << 0, 0, 0,		//
		0, 1, -1,			//
		1, 0, 0,			//
		0, -0.0711, 0.0711, //
		0, 0, 0,			//
		0, 0, -0.2795;		//
	Eigen::Vector3d theta(3);
	theta << 1.0472, 1.0472, 1.0472;
	Eigen::Matrix<double, 6, 3> result;
	result << 0, -0.866, 0.866, //
		0, 0.5, -0.5,			//
		1, 0, 0,				//
		0, -0.0355, -0.0855,	//
		0, -0.0615, -0.1481,	//
		0, 0, -0.1398;			//
	Eigen::Matrix<double, 6, 3> tmp_result = mr::JacobianSpace(s_list, theta);
	// std::cout << tmp_result << std::endl;
	ASSERT_TRUE(mr::JacobianSpace(s_list, theta).isApprox(result, 4));
}

TEST(MRTest, JacobianBodyTest)
{
	Eigen::Matrix<double, 6, 3> b_list;
	b_list << 0, 0, 0,		//
		0, 1, -1,			//
		1, 0, 0,			//
		0.0425, 0, 0,		//
		0.5515, 0, 0,		//
		0, -0.5515, 0.2720; //
	Eigen::Vector3d theta;
	theta << 0, 0, 1.5708;
	Eigen::Matrix<double, 6, 3> result;
	result << 1, 0, 0,			  //
		0, 1, -1,				  //
		0, 0, 0,				  //
		0, -0.2795, 0,			  //
		0.2795, 0, 0,			  //
		-0.0425, -0.2720, 0.2720; //
	Eigen::Matrix<double, 6, 3> tmp_result = mr::JacobianBody(b_list, theta);
	// std::cout << tmp_result << std::endl;
	ASSERT_TRUE(mr::JacobianBody(b_list, theta).isApprox(result, 4));
}

TEST(MRTest, adTest)
{
	Eigen::Matrix<double, 6, 1> V;
	V << 1, 2, 3, 4, 5, 6;

	Eigen::Matrix<double, 6, 6> result;
	result << 0, -3, 2, 0, 0, 0, //
		3, 0, -1, 0, 0, 0,		 //
		-2, 1, 0, 0, 0, 0,		 //
		0, -6, 5, 0, -3, 2,		 //
		6, 0, -4, 3, 0, -1,		 //
		-5, 4, 0, -2, 1, 0;		 //

	ASSERT_TRUE(mr::ad(V).isApprox(result, 4));
}

TEST(MRTest, TransInvTest)
{
	Eigen::Isometry3d input;
	input.setIdentity();
	input.linear() << 1, 0, 0, //
		0, 0, -1,			   //
		0, 1, 0;			   //
	input.translation() << 0, 0, 3;
	Eigen::Isometry3d result;
	result.setIdentity();
	result.linear() << 1, 0, 0, //
		0, 0, 1,				//
		0, -1, 0;				//
	result.translation() << 0, -3, 0;

	auto inv = mr::TransInv(input);
	ASSERT_TRUE(inv.isApprox(result, 4));
}

TEST(MRTest, RotInvTest)
{
	Eigen::Matrix3d input;
	input << 0, 0, 1, //
		1, 0, 0,	  //
		0, 1, 0;	  //
	Eigen::Matrix3d result;
	result << 0, 1, 0, //
		0, 0, 1,	   //
		1, 0, 0;	   //

	auto inv = mr::RotInv(input);
	ASSERT_TRUE(inv.isApprox(result, 4));
}

TEST(MRTest, ScrewToAxisTest)
{
	Eigen::Vector3d q, s;
	q << 3, 0, 1;
	s << 0, 0, 1;
	double h = 2;

	Eigen::Matrix<double, 6, 1> axis = mr::ScrewToAxis(q, s, h);
	Eigen::Matrix<double, 6, 1> result;
	result << 0, 0, 1, 0, -3, 2; //

	ASSERT_TRUE(axis.isApprox(result, 4));
}

TEST(MRTest, FKInBodyTest)
{
	Eigen::Isometry3d M;
	M.setIdentity();
	M.linear() << -1, 0, 0, //
		0, 1, 0,			//
		0, 0, -1;			//
	M.translation() << 0, 6, 2;
	Eigen::Matrix<double, 6, 3> Blist;
	Blist << 0, 0, 0, //
		0, 0, 0,	  //
		-1, 0, 1,	  //
		2, 0, 0,	  //
		0, 1, 0,	  //
		0, 0, 0.1;	  //
	Eigen::Vector3d thetaList;
	thetaList << M_PI / 2.0, 3, M_PI;

	Eigen::Isometry3d result;
	result.setIdentity();
	result.linear() << 0, 1, 0, //
		1, 0, 0,				//
		0, 0, -1;				//
	result.translation() << -5, 4, 1.68584073;
	Eigen::Isometry3d FKCal = mr::FKinBody(M, Blist, thetaList);

	ASSERT_TRUE(FKCal.isApprox(result, 4));
}

TEST(MRTest, FKInSpaceTest)
{
	Eigen::Isometry3d M;
	M.setIdentity();
	M.linear() << -1, 0, 0, //
		0, 1, 0,			//
		0, 0, -1;			//
	M.translation() << 0, 6, 2;
	Eigen::Matrix<double, 6, 3> Slist;
	Slist << 0, 0, 0, //
		0, 0, 0,	  //
		1, 0, -1,	  //
		4, 0, -6,	  //
		0, 1, 0,	  //
		0, 0, -0.1;	  //
	Eigen::Vector3d thetaList;
	thetaList << M_PI / 2.0, 3, M_PI;

	Eigen::Isometry3d result;
	result.setIdentity();
	result.linear() << 0, 1, 0, //
		1, 0, 0,				//
		0, 0, -1;				//
	result.translation() << -5, 4, 1.68584073;
	Eigen::Isometry3d FKCal = mr::FKinBody(M, Slist, thetaList);

	ASSERT_TRUE(FKCal.isApprox(result, 4));
}

TEST(MRTest, AxisAng6Test)
{
	Eigen::Matrix<double, 6, 1> input;
	Eigen::Matrix<double, 7, 1> result;
	input << 1.0, 0.0, 0.0, 1.0, 2.0, 3.0;
	result << 1.0, 0.0, 0.0, 1.0, 2.0, 3.0, 1.0;

	Eigen::Matrix<double, 7, 1> output = mr::AxisAng6(input);
	ASSERT_TRUE(output.isApprox(result, 4));
}

TEST(MRTest, MatrixLog6Test)
{
	Eigen::Isometry3d Tinput;
	Eigen::Matrix4d result;
	Tinput.setIdentity();
	Tinput.linear() << 1, 0, 0, //
		0, 0, -1,				//
		0, 1, 0;				//
	Tinput.translation() << 0, 0, 3;

	result << 0, 0, 0, 0,			   //
		0, 0, -1.57079633, 2.35619449, //
		0, 1.57079633, 0, 2.35619449,  //
		0, 0, 0, 0;					   //

	Eigen::Matrix4d Toutput = mr::MatrixLog6(Tinput);
	ASSERT_TRUE(Toutput.isApprox(result, 4));
}

TEST(MRTest, DistanceToSO3Test)
{
	Eigen::Matrix3d input;
	double result = 0.088353;
	input << 1.0, 0.0, 0.0, //
		0.0, 0.1, -0.95,	//
		0.0, 1.0, 0.1;		//
	EXPECT_NEAR(result, mr::DistanceToSO3(input), 3);
}

TEST(MRTest, DistanceToSE3Test)
{
	Eigen::Matrix4d input;
	double result = 0.134931;
	input << 1.0, 0.0, 0.0, 1.2, //
		0.0, 0.1, -0.95, 1.5,	 //
		0.0, 1.0, 0.1, -0.9,	 //
		0.0, 0.0, 0.1, 0.98;	 //
	EXPECT_NEAR(result, mr::DistanceToSE3(input), 3);
}

TEST(MRTest, TestIfSO3Test)
{
	Eigen::Matrix3d input;
	bool result = false;
	input << 1.0, 0.0, 0.0, //
		0.0, 0.1, -0.95,	//
		0.0, 1.0, 0.1;		//
	ASSERT_EQ(result, mr::TestIfSO3(input));
}

TEST(MRTest, TestIfSE3Test)
{
	Eigen::Matrix4d input;
	bool result = false;
	input << 1.0, 0.0, 0.0, 1.2, //
		0.0, 0.1, -0.95, 1.5,	 //
		0.0, 1.0, 0.1, -0.9,	 //
		0.0, 0.0, 0.1, 0.98;	 //
	ASSERT_EQ(result, mr::TestIfSE3(input));
}

TEST(MRTest, IKinBodyTest)
{
	Eigen::Matrix<double, 3, 6> BlistT;
	BlistT << 0, 0, -1, 2, 0, 0, //
		0, 0, 0, 0, 1, 0,		 //
		0, 0, 1, 0, 0, 0.1;		 //
	Eigen::Matrix<double, 6, 3> Blist = BlistT.transpose();
	Eigen::Isometry3d M;
	M.setIdentity();
	M.linear() << -1, 0, 0, //
		0, 1, 0,			//
		0, 0, -1;			//
	M.translation() << 0, 6, 2;
	Eigen::Isometry3d T;
	T.setIdentity();
	T.linear() << 0, 1, 0, //
		1, 0, 0,		   //
		0, 0, -1;		   //
	T.translation() << -5, 4, 1.6858;
	Eigen::Vector3d thetalist;
	thetalist << 1.5, 2.5, 3;
	double eomg = 0.01;
	double ev = 0.001;
	bool b_result = true;
	Eigen::Vector3d theta_result;
	theta_result << 1.57073819, 2.999667, 3.14153913;
	bool iRet = mr::IKinBody(Blist, M, T, thetalist, eomg, ev);
	ASSERT_EQ(b_result, iRet);
	ASSERT_TRUE(thetalist.isApprox(theta_result, 4));
}

TEST(MRTest, IKinSpaceTest)
{
	Eigen::Matrix<double, 3, 6> SlistT;
	SlistT << 0, 0, 1, 4, 0, 0, //
		0, 0, 0, 0, 1, 0,		//
		0, 0, -1, -6, 0, -0.1;	//
	Eigen::Matrix<double, 6, 3> Slist = SlistT.transpose();
	Eigen::Isometry3d M;
	M.setIdentity();
	M.linear() << -1, 0, 0, //
		0, 1, 0,			//
		0, 0, -1;			//
	M.translation() << 0, 6, 2;
	Eigen::Isometry3d T;
	T.setIdentity();
	T.linear() << 0, 1, 0, //
		1, 0, 0,		   //
		0, 0, -1;		   //
	T.translation() << -5, 4, 1.6858;
	Eigen::Vector3d thetalist;
	thetalist << 1.5, 2.5, 3;
	double eomg = 0.01;
	double ev = 0.001;
	bool b_result = true;
	Eigen::Vector3d theta_result;
	theta_result << 1.57073783, 2.99966384, 3.1415342;
	bool iRet = mr::IKinSpace(Slist, M, T, thetalist, eomg, ev);
	ASSERT_EQ(b_result, iRet);
	ASSERT_TRUE(thetalist.isApprox(theta_result, 4));
}

TEST(MRTest, AdjointTest)
{
	Eigen::Isometry3d T;
	T.setIdentity();
	T.linear() << 1, 0, 0, //
		0, 0, -1,		   //
		0, 1, 0;		   //
	T.translation() << 0, 0, 3;
	Eigen::Matrix<double, 6, 6> result;
	result << 1, 0, 0, 0, 0, 0, //
		0, 0, -1, 0, 0, 0,		//
		0, 1, 0, 0, 0, 0,		//
		0, 0, 3, 1, 0, 0,		//
		3, 0, 0, 0, 0, -1,		//
		0, 0, 0, 0, 1, 0;		//

	ASSERT_TRUE(mr::Adjoint(T).isApprox(result, 4));
}

TEST(MRTest, InverseDynamicsTest)
{
	Eigen::Vector3d thetalist;
	thetalist << 0.1, 0.1, 0.1;
	Eigen::Vector3d dthetalist;
	dthetalist << 0.1, 0.2, 0.3;
	Eigen::Vector3d ddthetalist;
	ddthetalist << 2, 1.5, 1;
	Eigen::Vector3d g;
	g << 0, 0, -9.8;
	Eigen::Matrix<double, 6, 1> Ftip;
	Ftip << 1, 1, 1, 1, 1, 1;

	Eigen::Isometry3d M01;
	M01.setIdentity();
	M01.translation() << 0, 0, 0.089159;
	Eigen::Isometry3d M12;
	M12.setIdentity();
	M12.linear() << 0, 0, 1, //
		0, 1, 0,			 //
		-1, 0, 0;			 //
	M12.translation() << 0.28, 0.13585, 0;
	Eigen::Isometry3d M23;
	M23.setIdentity();
	M23.translation() << 0, -0.1197, 0.395;
	Eigen::Isometry3d M34;
	M34.setIdentity();
	M34.translation() << 0, 0, 0.14225;

	std::array<Eigen::Isometry3d, 4> Mlist{M01, M12, M23, M34};

	Eigen::Matrix<double, 6, 1> G1;
	G1 << 0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7;
	Eigen::Matrix<double, 6, 1> G2;
	G2 << 0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393;
	Eigen::Matrix<double, 6, 1> G3;
	G3 << 0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275;

	std::array<Eigen::Matrix<double, 6, 6>, 3> Glist{G1.asDiagonal(), G2.asDiagonal(), G3.asDiagonal()};

	Eigen::Matrix<double, 3, 6> SlistT;
	SlistT << 1, 0, 1, 0, 1, 0,	   //
		0, 1, 0, -0.089, 0, 0,	   //
		0, 1, 0, -0.089, 0, 0.425; //
	Eigen::Matrix<double, 6, 3> Slist = SlistT.transpose();

	Eigen::Vector3d taulist = mr::InverseDynamics<3>(thetalist, dthetalist, ddthetalist, g,
													 Ftip, Mlist, Glist, Slist);

	Eigen::Vector3d result;
	result << 74.6962, -33.0677, -3.23057;

	ASSERT_TRUE(taulist.isApprox(result, 4));
}

TEST(MRTest, GravityForcesTest)
{
	Eigen::Vector3d thetalist;
	thetalist << 0.1, 0.1, 0.1;
	Eigen::Vector3d g;
	g << 0, 0, -9.8;

	std::array<Eigen::Isometry3d, 4> Mlist;
	std::array<Eigen::Matrix<double, 6, 6>, 3> Glist;

	Eigen::Isometry3d M01;
	M01.setIdentity();
	M01.translation() << 0, 0, 0.089159;
	Eigen::Isometry3d M12;
	M12.setIdentity();
	M12.linear() << 0, 0, 1, //
		0, 1, 0,			 //
		-1, 0, 0;			 //
	M12.translation() << 0.28, 0.13585, 0;
	Eigen::Isometry3d M23;
	M23.setIdentity();
	M23.translation() << 0, -0.1197, 0.395;
	Eigen::Isometry3d M34;
	M34.setIdentity();
	M34.translation() << 0, 0, 0.14225;

	Mlist = {M01, M12, M23, M34};

	Eigen::Matrix<double, 6, 1> G1;
	G1 << 0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7;
	Eigen::Matrix<double, 6, 1> G2;
	G2 << 0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393;
	Eigen::Matrix<double, 6, 1> G3;
	G3 << 0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275;

	Glist = {G1.asDiagonal(), G2.asDiagonal(), G3.asDiagonal()};

	Eigen::Matrix<double, 3, 6> SlistT;
	SlistT << 1, 0, 1, 0, 1, 0,	   //
		0, 1, 0, -0.089, 0, 0,	   //
		0, 1, 0, -0.089, 0, 0.425; //
	Eigen::Matrix<double, 6, 3> Slist = SlistT.transpose();

	Eigen::Vector3d grav = mr::GravityForces<3>(thetalist, g, Mlist, Glist, Slist);

	Eigen::Vector3d result;
	result << 28.4033, -37.6409, -5.4416;

	ASSERT_TRUE(grav.isApprox(result, 4));
}

TEST(MRTest, MassMatrixTest)
{
	Eigen::Vector3d thetalist;
	thetalist << 0.1, 0.1, 0.1;

	std::array<Eigen::Isometry3d, 4> Mlist;
	std::array<Eigen::Matrix<double, 6, 6>, 3> Glist;

	Eigen::Isometry3d M01;
	M01.setIdentity();
	M01.translation() << 0, 0, 0.089159;
	Eigen::Isometry3d M12;
	M12.setIdentity();
	M12.linear() << 0, 0, 1, //
		0, 1, 0,			 //
		-1, 0, 0;			 //
	M12.translation() << 0.28, 0.13585, 0;
	Eigen::Isometry3d M23;
	M23.setIdentity();
	M23.translation() << 0, -0.1197, 0.395;
	Eigen::Isometry3d M34;
	M34.setIdentity();
	M34.translation() << 0, 0, 0.14225;

	Mlist = {M01, M12, M23, M34};

	Eigen::Matrix<double, 6, 1> G1;
	G1 << 0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7;
	Eigen::Matrix<double, 6, 1> G2;
	G2 << 0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393;
	Eigen::Matrix<double, 6, 1> G3;
	G3 << 0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275;

	Glist = {G1.asDiagonal(), G2.asDiagonal(), G3.asDiagonal()};

	Eigen::Matrix<double, 3, 6> SlistT;
	SlistT << 1, 0, 1, 0, 1, 0,	   //
		0, 1, 0, -0.089, 0, 0,	   //
		0, 1, 0, -0.089, 0, 0.425; //
	Eigen::Matrix<double, 6, 3> Slist = SlistT.transpose();

	Eigen::Matrix3d M = mr::MassMatrix<3>(thetalist, Mlist, Glist, Slist);

	Eigen::Matrix3d result;
	result << 22.5433, -0.3071, -0.0072, //
		-0.3071, 1.9685, 0.4322,		 //
		-0.0072, 0.4322, 0.1916;		 //

	ASSERT_TRUE(M.isApprox(result, 4));
}

TEST(MRTest, VelQuadraticForcesTest)
{
	Eigen::Vector3d thetalist;
	thetalist << 0.1, 0.1, 0.1;
	Eigen::Vector3d dthetalist;
	dthetalist << 0.1, 0.2, 0.3;

	std::array<Eigen::Isometry3d, 4> Mlist;
	std::array<Eigen::Matrix<double, 6, 6>, 3> Glist;

	Eigen::Isometry3d M01;
	M01.setIdentity();
	M01.translation() << 0, 0, 0.089159;
	Eigen::Isometry3d M12;
	M12.setIdentity();
	M12.linear() << 0, 0, 1, //
		0, 1, 0,			 //
		-1, 0, 0;			 //
	M12.translation() << 0.28, 0.13585, 0;
	Eigen::Isometry3d M23;
	M23.setIdentity();
	M23.translation() << 0, -0.1197, 0.395;
	Eigen::Isometry3d M34;
	M34.setIdentity();
	M34.translation() << 0, 0, 0.14225;

	Mlist = {M01, M12, M23, M34};

	Eigen::Matrix<double, 6, 1> G1;
	G1 << 0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7;
	Eigen::Matrix<double, 6, 1> G2;
	G2 << 0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393;
	Eigen::Matrix<double, 6, 1> G3;
	G3 << 0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275;

	Glist = {G1.asDiagonal(), G2.asDiagonal(), G3.asDiagonal()};

	Eigen::Matrix<double, 3, 6> SlistT;
	SlistT << 1, 0, 1, 0, 1, 0,	   //
		0, 1, 0, -0.089, 0, 0,	   //
		0, 1, 0, -0.089, 0, 0.425; //
	Eigen::Matrix<double, 6, 3> Slist = SlistT.transpose();

	Eigen::Vector3d c = mr::VelQuadraticForces<3>(thetalist, dthetalist, Mlist, Glist, Slist);

	Eigen::Vector3d result;
	result << 0.2645, -0.0551, -0.0069;

	ASSERT_TRUE(c.isApprox(result, 4));
}

TEST(MRTest, EndEffectorForcesTest)
{
	Eigen::Vector3d thetalist;
	thetalist << 0.1, 0.1, 0.1;
	Eigen::Matrix<double, 6, 1> Ftip;
	Ftip << 1, 1, 1, 1, 1, 1;

	std::array<Eigen::Isometry3d, 4> Mlist;
	std::array<Eigen::Matrix<double, 6, 6>, 3> Glist;

	Eigen::Isometry3d M01;
	M01.setIdentity();
	M01.translation() << 0, 0, 0.089159;
	Eigen::Isometry3d M12;
	M12.setIdentity();
	M12.linear() << 0, 0, 1, //
		0, 1, 0,			 //
		-1, 0, 0;			 //
	M12.translation() << 0.28, 0.13585, 0;
	Eigen::Isometry3d M23;
	M23.setIdentity();
	M23.translation() << 0, -0.1197, 0.395;
	Eigen::Isometry3d M34;
	M34.setIdentity();
	M34.translation() << 0, 0, 0.14225;

	Mlist = {M01, M12, M23, M34};

	Eigen::Matrix<double, 6, 1> G1;
	G1 << 0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7;
	Eigen::Matrix<double, 6, 1> G2;
	G2 << 0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393;
	Eigen::Matrix<double, 6, 1> G3;
	G3 << 0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275;

	Glist = {G1.asDiagonal(), G2.asDiagonal(), G3.asDiagonal()};

	Eigen::Matrix<double, 3, 6> SlistT;
	SlistT << 1, 0, 1, 0, 1, 0,	   //
		0, 1, 0, -0.089, 0, 0,	   //
		0, 1, 0, -0.089, 0, 0.425; //
	Eigen::Matrix<double, 6, 3> Slist = SlistT.transpose();

	Eigen::Vector3d JTFtip = mr::EndEffectorForces<3>(thetalist, Ftip, Mlist, Glist, Slist);

	Eigen::Vector3d result;
	result << 1.4095, 1.8577, 1.3924;

	ASSERT_TRUE(JTFtip.isApprox(result, 4));
}

TEST(MRTest, ForwardDynamicsTest)
{
	Eigen::Vector3d thetalist;
	thetalist << 0.1, 0.1, 0.1;
	Eigen::Vector3d dthetalist;
	dthetalist << 0.1, 0.2, 0.3;
	Eigen::Vector3d taulist;
	taulist << 0.5, 0.6, 0.7;
	Eigen::Vector3d g;
	g << 0, 0, -9.8;
	Eigen::Matrix<double, 6, 1> Ftip;
	Ftip << 1, 1, 1, 1, 1, 1;

	std::array<Eigen::Isometry3d, 4> Mlist;
	std::array<Eigen::Matrix<double, 6, 6>, 3> Glist;

	Eigen::Isometry3d M01;
	M01.setIdentity();
	M01.translation() << 0, 0, 0.089159;
	Eigen::Isometry3d M12;
	M12.setIdentity();
	M12.linear() << 0, 0, 1, //
		0, 1, 0,			 //
		-1, 0, 0;			 //
	M12.translation() << 0.28, 0.13585, 0;
	Eigen::Isometry3d M23;
	M23.setIdentity();
	M23.translation() << 0, -0.1197, 0.395;
	Eigen::Isometry3d M34;
	M34.setIdentity();
	M34.translation() << 0, 0, 0.14225;

	Mlist = {M01, M12, M23, M34};

	Eigen::Matrix<double, 6, 1> G1;
	G1 << 0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7;
	Eigen::Matrix<double, 6, 1> G2;
	G2 << 0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393;
	Eigen::Matrix<double, 6, 1> G3;
	G3 << 0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275;

	Glist = {G1.asDiagonal(), G2.asDiagonal(), G3.asDiagonal()};

	Eigen::Matrix<double, 3, 6> SlistT;
	SlistT << 1, 0, 1, 0, 1, 0,	   //
		0, 1, 0, -0.089, 0, 0,	   //
		0, 1, 0, -0.089, 0, 0.425; //
	Eigen::Matrix<double, 6, 3> Slist = SlistT.transpose();

	Eigen::Vector3d ddthetalist = mr::ForwardDynamics<3>(thetalist, dthetalist, taulist, g,
														 Ftip, Mlist, Glist, Slist);

	Eigen::Vector3d result;
	result << -0.9739, 25.5847, -32.9150;

	ASSERT_TRUE(ddthetalist.isApprox(result, 4));
}

TEST(MRTest, EulerStepTest)
{
	Eigen::Vector3d thetalist;
	thetalist << 0.1, 0.1, 0.1;
	Eigen::Vector3d dthetalist;
	dthetalist << 0.1, 0.2, 0.3;
	Eigen::Vector3d ddthetalist;
	ddthetalist << 2, 1.5, 1;
	double dt = 0.1;

	mr::EulerStep(thetalist, dthetalist, ddthetalist, dt);

	Eigen::Vector3d result_thetalistNext;
	result_thetalistNext << 0.11, 0.12, 0.13;
	Eigen::Vector3d result_dthetalistNext;
	result_dthetalistNext << 0.3, 0.35, 0.4;

	ASSERT_TRUE(thetalist.isApprox(result_thetalistNext, 4));
	ASSERT_TRUE(dthetalist.isApprox(result_dthetalistNext, 4));
}

TEST(MRTest, ComputedTorqueTest)
{
	Eigen::Vector3d thetalist;
	thetalist << 0.1, 0.1, 0.1;
	Eigen::Vector3d dthetalist;
	dthetalist << 0.1, 0.2, 0.3;
	Eigen::Vector3d eint;
	eint << 0.2, 0.2, 0.2;
	Eigen::Vector3d g;
	g << 0, 0, -9.8;

	std::array<Eigen::Isometry3d, 4> Mlist;
	std::array<Eigen::Matrix<double, 6, 6>, 3> Glist;

	Eigen::Isometry3d M01;
	M01.setIdentity();
	M01.translation() << 0, 0, 0.089159;
	Eigen::Isometry3d M12;
	M12.setIdentity();
	M12.linear() << 0, 0, 1, //
		0, 1, 0,			 //
		-1, 0, 0;			 //
	M12.translation() << 0.28, 0.13585, 0;
	Eigen::Isometry3d M23;
	M23.setIdentity();
	M23.translation() << 0, -0.1197, 0.395;
	Eigen::Isometry3d M34;
	M34.setIdentity();
	M34.translation() << 0, 0, 0.14225;

	Mlist = {M01, M12, M23, M34};

	Eigen::Matrix<double, 6, 1> G1;
	G1 << 0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7;
	Eigen::Matrix<double, 6, 1> G2;
	G2 << 0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393;
	Eigen::Matrix<double, 6, 1> G3;
	G3 << 0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275;

	Glist = {G1.asDiagonal(), G2.asDiagonal(), G3.asDiagonal()};

	Eigen::Matrix<double, 3, 6> SlistT;
	SlistT << 1, 0, 1, 0, 1, 0,	   //
		0, 1, 0, -0.089, 0, 0,	   //
		0, 1, 0, -0.089, 0, 0.425; //
	Eigen::Matrix<double, 6, 3> Slist = SlistT.transpose();

	Eigen::Vector3d thetalistd;
	thetalistd << 1.0, 1.0, 1.0;
	Eigen::Vector3d dthetalistd;
	dthetalistd << 2, 1.2, 2;
	Eigen::Vector3d ddthetalistd;
	ddthetalistd << 0.1, 0.1, 0.1;
	double Kp = 1.3;
	double Ki = 1.2;
	double Kd = 1.1;

	Eigen::Vector3d taulist = mr::ComputedTorque<3>(thetalist, dthetalist, eint, g,
													Mlist, Glist, Slist, thetalistd, dthetalistd, ddthetalistd, Kp, Ki, Kd);

	Eigen::Vector3d result;
	result << 133.00525246, -29.94223324, -3.03276856;

	ASSERT_TRUE(taulist.isApprox(result, 4));
}

TEST(MRTest, CubicTimeScalingTest)
{
	double Tf = 2.0;
	double t = 0.6;
	double result = 0.216;

	EXPECT_NEAR(result, mr::CubicTimeScaling(Tf, t), 3);
}

TEST(MRTest, QuinticTimeScalingTest)
{
	double Tf = 2.0;
	double t = 0.6;
	double result = 0.16308;

	EXPECT_NEAR(result, mr::QuinticTimeScaling(Tf, t), 3);
}

TEST(MRTest, JointTrajectoryTest)
{
	constexpr int dof = 8;
	Eigen::Matrix<double, dof, 1> thetastart;
	thetastart << 1, 0, 0, 1, 1, 0.2, 0, 1;
	Eigen::Matrix<double, dof, 1> thetaend;
	thetaend << 1.2, 0.5, 0.6, 1.1, 2, 2, 0.9, 1;
	double Tf = 4.0;
	constexpr int N = 6;
	int method = 3;

	Eigen::Matrix<double, N, dof> result;
	result << 1, 0, 0, 1, 1, 0.2, 0, 1,							 //
		1.0208, 0.052, 0.0624, 1.0104, 1.104, 0.3872, 0.0936, 1, //
		1.0704, 0.176, 0.2112, 1.0352, 1.352, 0.8336, 0.3168, 1, //
		1.1296, 0.324, 0.3888, 1.0648, 1.648, 1.3664, 0.5832, 1, //
		1.1792, 0.448, 0.5376, 1.0896, 1.896, 1.8128, 0.8064, 1, //
		1.2, 0.5, 0.6, 1.1, 2, 2, 0.9, 1;						 //

	Eigen::Matrix<double, N, dof> traj = mr::JointTrajectory<dof>(thetastart, thetaend, Tf, N, method);
	ASSERT_TRUE(traj.isApprox(result, 4));
}

TEST(MRTest, ScrewTrajectoryTest)
{
	Eigen::Isometry3d Xstart;
	Xstart.setIdentity();
	Xstart.translation() << 1, 0, 1;
	Eigen::Isometry3d Xend;
	Xend.setIdentity();
	Xend.linear() << 0, 0, 1,
		1, 0, 0,
		0, 1, 0;
	Xend.translation() << 0.1, 0, 4.1;
	double Tf = 5.0;
	int N = 4;
	int method = 3;

	std::vector<Eigen::Isometry3d> result(N);
	result[0] = Xstart;
	Eigen::Isometry3d X12;
	X12.setIdentity();
	X12.linear() << 0.904, -0.25, 0.346, //
		0.346, 0.904, -0.25,			 //
		-0.25, 0.346, 0.904;			 //
	X12.translation() << 0.441, 0.529, 1.601;
	Eigen::Isometry3d X23;
	X23.setIdentity();
	X23.linear() << 0.346, -0.25, 0.904, //
		0.904, 0.346, -0.25,			 //
		-0.25, 0.904, 0.346;			 //
	X23.translation() << -0.117, 0.473, 3.274;
	result[1] = X12;
	result[2] = X23;
	result[3] = Xend;

	std::vector<Eigen::Isometry3d> traj = mr::ScrewTrajectory(Xstart, Xend, Tf, N, method);

	for (int i = 0; i < N; ++i)
	{
		ASSERT_TRUE(traj[i].isApprox(result[i], 4));
	}
}

TEST(MRTest, CartesianTrajectoryTest)
{
	Eigen::Isometry3d Xstart;
	Xstart.setIdentity();
	Xstart.translation() << 1, 0, 1;
	Eigen::Isometry3d Xend;
	Xend.setIdentity();
	Xend.linear() << 0, 0, 1,
		1, 0, 0,
		0, 1, 0;
	Xend.translation() << 0.1, 0, 4.1;
	double Tf = 5.0;
	int N = 4;
	int method = 5;

	std::vector<Eigen::Isometry3d> result(N);
	result[0] = Xstart;
	Eigen::Isometry3d X12;
	X12.setIdentity();
	X12.linear() << 0.937, -0.214, 0.277, //
		0.277, 0.937, -0.214,			  //
		-0.214, 0.277, 0.937;			  //
	X12.translation() << 0.811, 0, 1.651;
	Eigen::Isometry3d X23;
	X23.setIdentity();
	X23.linear() << 0.277, -0.214, 0.937, //
		0.937, 0.277, -0.214,			  //
		-0.214, 0.937, 0.277;			  //
	X23.translation() << 0.289, 0, 3.449;
	result[1] = X12;
	result[2] = X23;
	result[3] = Xend;

	std::vector<Eigen::Isometry3d> traj = mr::CartesianTrajectory(Xstart, Xend, Tf, N, method);

	for (int i = 0; i < N; ++i)
	{
		ASSERT_TRUE(traj[i].isApprox(result[i], 4));
	}
}

TEST(MRTest, InverseDynamicsTrajectoryTest)
{
	constexpr int dof = 3;
	Eigen::Vector3d thetastart;
	thetastart << 0, 0, 0;
	Eigen::Vector3d thetaend;
	thetaend << M_PI / 2, M_PI / 2, M_PI / 2;
	double Tf = 3.0;
	constexpr int N = 1000;
	int method = 5;

	Eigen::Matrix<double, N, dof> traj = mr::JointTrajectory(thetastart, thetaend, Tf, N, method);
	Eigen::Matrix<double, N, dof> thetamat = traj;
	Eigen::Matrix<double, N, dof> dthetamat = Eigen::Matrix<double, N, dof>::Zero();
	Eigen::Matrix<double, N, dof> ddthetamat = Eigen::Matrix<double, N, dof>::Zero();
	double dt = Tf / (N - 1.0);
	for (int i = 0; i < N - 1; ++i)
	{
		dthetamat.row(i + 1) = (thetamat.row(i + 1) - thetamat.row(i)) / dt;
		ddthetamat.row(i + 1) = (dthetamat.row(i + 1) - dthetamat.row(i)) / dt;
	}
	Eigen::Vector3d g;
	g << 0, 0, -9.8;
	Eigen::Matrix<double, N, 6> Ftipmat = Eigen::Matrix<double, N, 6>::Zero();

	std::array<Eigen::Isometry3d, 4> Mlist;
	std::array<Eigen::Matrix<double, 6, 6>, 3> Glist;
	Eigen::Isometry3d M01;
	M01.setIdentity();
	M01.translation() << 0, 0, 0.089159;
	Eigen::Isometry3d M12;
	M12.setIdentity();
	M12.linear() << 0, 0, 1,
		0, 1, 0,
		-1, 0, 0;
	M12.translation() << 0.28, 0.13585, 0;
	Eigen::Isometry3d M23;
	M23.setIdentity();
	M23.translation() << 0, -0.1197, 0.395;
	Eigen::Isometry3d M34;
	M34.setIdentity();
	M34.translation() << 0, 0, 0.14225;
	Mlist = {M01, M12, M23, M34};

	Eigen::Matrix<double, 6, 1> G1;
	G1 << 0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7;
	Eigen::Matrix<double, 6, 1> G2;
	G2 << 0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393;
	Eigen::Matrix<double, 6, 1> G3;
	G3 << 0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275;
	Glist = {G1.asDiagonal(), G2.asDiagonal(), G3.asDiagonal()};

	Eigen::Matrix<double, 3, 6> SlistT;
	SlistT << 1, 0, 1, 0, 1, 0,
		0, 1, 0, -0.089, 0, 0,
		0, 1, 0, -0.089, 0, 0.425;
	Eigen::Matrix<double, 6, 3> Slist = SlistT.transpose();

	constexpr int numTest = 3;
	Eigen::Matrix<double, numTest, dof> result;
	Eigen::Vector3d tau_timestep_beg;
	tau_timestep_beg << 13.22970794, -36.262108, -4.181341;
	Eigen::Vector3d tau_timestep_mid;
	tau_timestep_mid << 115.55863434, -22.05129215, 1.00916115;
	Eigen::Vector3d tau_timestep_end;
	tau_timestep_end << 81.12700926, -23.20753925, 2.48432708;
	result << tau_timestep_beg.transpose(),
		tau_timestep_mid.transpose(),
		tau_timestep_end.transpose();

	Eigen::Matrix<double, N, dof> taumat = mr::InverseDynamicsTrajectory<dof, N>(thetamat, dthetamat, ddthetamat, g, Ftipmat, Mlist, Glist, Slist);
	Eigen::Matrix<double, numTest, dof> taumat_timestep;
	taumat_timestep << taumat.row(0),
		taumat.row(int(N / 2) - 1),
		taumat.row(N - 1);
	ASSERT_TRUE(taumat_timestep.isApprox(result, 4));
}

TEST(MRTest, ForwardDynamicsTrajectoryTest)
{
	Eigen::Vector3d thetalist;
	thetalist << 0.1, 0.1, 0.1;
	Eigen::Vector3d dthetalist;
	dthetalist << 0.1, 0.2, 0.3;
	constexpr int N = 10;
	constexpr int dof = 3;
	Eigen::Matrix<double, N, 3> taumat;
	taumat << 3.63, -6.58, -5.57, //
		3.74, -5.55, -5.5,		  //
		4.31, -0.68, -5.19,		  //
		5.18, 5.63, -4.31,		  //
		5.85, 8.17, -2.59,		  //
		5.78, 2.79, -1.7,		  //
		4.99, -5.3, -1.19,		  //
		4.08, -9.41, 0.07,		  //
		3.56, -10.1, 0.97,		  //
		3.49, -9.41, 1.23;		  //
	Eigen::Vector3d g;
	g << 0, 0, -9.8;
	Eigen::Matrix<double, N, 6> Ftipmat = Eigen::Matrix<double, N, 6>::Zero();

	std::array<Eigen::Isometry3d, 4> Mlist;
	std::array<Eigen::Matrix<double, 6, 6>, 3> Glist;
	Eigen::Isometry3d M01;
	M01.setIdentity();
	M01.translation() << 0, 0, 0.089159;
	Eigen::Isometry3d M12;
	M12.setIdentity();
	M12.linear() << 0, 0, 1,
		0, 1, 0,
		-1, 0, 0;
	M12.translation() << 0.28, 0.13585, 0;
	Eigen::Isometry3d M23;
	M23.setIdentity();
	M23.translation() << 0, -0.1197, 0.395;
	Eigen::Isometry3d M34;
	M34.setIdentity();
	M34.translation() << 0, 0, 0.14225;
	Mlist = {M01, M12, M23, M34};

	Eigen::Matrix<double, 6, 1> G1;
	G1 << 0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7;
	Eigen::Matrix<double, 6, 1> G2;
	G2 << 0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393;
	Eigen::Matrix<double, 6, 1> G3;
	G3 << 0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275;
	Glist = {G1.asDiagonal(), G2.asDiagonal(), G3.asDiagonal()};

	Eigen::Matrix<double, 3, 6> SlistT;
	SlistT << 1, 0, 1, 0, 1, 0,
		0, 1, 0, -0.089, 0, 0,
		0, 1, 0, -0.089, 0, 0.425;
	Eigen::Matrix<double, 6, 3> Slist = SlistT.transpose();
	double dt = 0.1;
	int intRes = 8;

	Eigen::Matrix<double, N, dof> result_thetamat;
	Eigen::Matrix<double, N, dof> result_dthetamat;
	result_thetamat << 0.1, 0.1, 0.1,		   //
		0.10643138, 0.2625997, -0.22664947,	   //
		0.10197954, 0.71581297, -1.22521632,   //
		0.0801044, 1.33930884, -2.28074132,	   //
		0.0282165, 2.11957376, -3.07544297,	   //
		-0.07123855, 2.87726666, -3.83289684,  //
		-0.20136466, 3.397858, -4.83821609,	   //
		-0.32380092, 3.73338535, -5.98695747,  //
		-0.41523262, 3.85883317, -7.01130559,  //
		-0.4638099, 3.63178793, -7.63190052;   //
	result_dthetamat << 0.1, 0.2, 0.3,		   //
		0.01212502, 3.42975773, -7.74792602,   //
		-0.13052771, 5.55997471, -11.22722784, //
		-0.35521041, 7.11775879, -9.18173035,  //
		-0.77358795, 8.17307573, -7.05744594,  //
		-1.2350231, 6.35907497, -8.99784746,   //
		-1.31426299, 4.07685875, -11.18480509, //
		-1.06794821, 2.49227786, -11.69748583, //
		-0.70264871, -0.55925705, -8.16067131, //
		-0.1455669, -4.57149985, -3.43135114;  //

	auto traj = mr::ForwardDynamicsTrajectory<dof>(thetalist, dthetalist, taumat, g, Ftipmat, Mlist, Glist, Slist, dt, intRes);
	Eigen::Matrix<double, N, dof> traj_theta = traj.first;
	Eigen::Matrix<double, N, dof> traj_dtheta = traj.second;

	ASSERT_TRUE(traj_theta.isApprox(result_thetamat, 4));
	ASSERT_TRUE(traj_dtheta.isApprox(result_dthetamat, 4));
}

TEST(MRTest, SimulateControlTest)
{
	Eigen::Vector3d thetalist;
	thetalist << 0.1, 0.1, 0.1;
	Eigen::Vector3d dthetalist;
	dthetalist << 0.1, 0.2, 0.3;
	Eigen::Vector3d g;
	g << 0, 0, -9.8;

	std::array<Eigen::Isometry3d, 4> Mlist;
	std::array<Eigen::Matrix<double, 6, 6>, 3> Glist;
	Eigen::Isometry3d M01;
	M01.setIdentity();
	M01.translation() << 0, 0, 0.089159;
	Eigen::Isometry3d M12;
	M12.setIdentity();
	M12.linear() << 0, 0, 1,
		0, 1, 0,
		-1, 0, 0;
	M12.translation() << 0.28, 0.13585, 0;
	Eigen::Isometry3d M23;
	M23.setIdentity();
	M23.translation() << 0, -0.1197, 0.395;
	Eigen::Isometry3d M34;
	M34.setIdentity();
	M34.translation() << 0, 0, 0.14225;
	Mlist = {M01, M12, M23, M34};

	Eigen::Matrix<double, 6, 1> G1;
	G1 << 0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7;
	Eigen::Matrix<double, 6, 1> G2;
	G2 << 0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393;
	Eigen::Matrix<double, 6, 1> G3;
	G3 << 0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275;
	Glist = {G1.asDiagonal(), G2.asDiagonal(), G3.asDiagonal()};

	Eigen::Matrix<double, 3, 6> SlistT;
	SlistT << 1, 0, 1, 0, 1, 0,
		0, 1, 0, -0.089, 0, 0,
		0, 1, 0, -0.089, 0, 0.425;
	Eigen::Matrix<double, 6, 3> Slist = SlistT.transpose();
	Eigen::Vector3d thetaend;
	thetaend << M_PI / 2, M_PI / 2, M_PI / 2;
	constexpr double Tf = 1.0;
	constexpr int N = int(1.0 * Tf / 0.01);
	double dt = Tf / (N - 1.0);
	int method = 5;

	Eigen::MatrixXd traj = mr::JointTrajectory(thetalist, thetaend, Tf, N, method);
	Eigen::MatrixXd thetamatd = traj;
	Eigen::MatrixXd dthetamatd = Eigen::MatrixXd::Zero(N, 3);
	Eigen::MatrixXd ddthetamatd = Eigen::MatrixXd::Zero(N, 3);
	for (int i = 0; i < N - 1; ++i)
	{
		dthetamatd.row(i + 1) = (thetamatd.row(i + 1) - thetamatd.row(i)) / dt;
		ddthetamatd.row(i + 1) = (dthetamatd.row(i + 1) - dthetamatd.row(i)) / dt;
	}

	Eigen::Vector3d gtilde;
	gtilde << 0.8, 0.2, -8.8;

	std::array<Eigen::Isometry3d, 4> Mtildelist;
	std::array<Eigen::Matrix<double, 6, 6>, 3> Gtildelist;
	Eigen::Isometry3d Mhat01;
	Mhat01.setIdentity();
	Mhat01.translation() << 0, 0, 0.1;
	Eigen::Isometry3d Mhat12;
	Mhat12.setIdentity();
	Mhat12.linear() << 0, 0, 1,
		0, 1, 0,
		-1, 0, 0;
	Mhat12.translation() << 0.3, 0.2, 0;
	Eigen::Isometry3d Mhat23;
	Mhat23.setIdentity();
	Mhat23.translation() << 0, -0.2, 0.4;
	Eigen::Isometry3d Mhat34;
	Mhat34.setIdentity();
	Mhat34.translation() << 0, 0, 0.2;
	Mtildelist = {Mhat01, Mhat12, Mhat23, Mhat34};

	Eigen::Matrix<double, 6, 1> Ghat1;
	Ghat1 << 0.1, 0.1, 0.1, 4, 4, 4;
	Eigen::Matrix<double, 6, 1> Ghat2;
	Ghat2 << 0.3, 0.3, 0.1, 9, 9, 9;
	Eigen::Matrix<double, 6, 1> Ghat3;
	Ghat3 << 0.1, 0.1, 0.1, 3, 3, 3;
	Gtildelist = {Ghat1.asDiagonal(), Ghat2.asDiagonal(), Ghat3.asDiagonal()};

	Eigen::MatrixXd Ftipmat = Eigen::MatrixXd::Ones(N, 6);
	double Kp = 20.0;
	double Ki = 10.0;
	double Kd = 18.0;
	int intRes = 8;

	constexpr int numTest = 3; // test 0, N/2-1, N-1 indices of results
	Eigen::Matrix<double, numTest, 3> result_taumat;
	Eigen::Matrix<double, numTest, 3> result_thetamat;

	Eigen::Vector3d tau_timestep_beg;
	tau_timestep_beg << -14.2640765, -54.06797429, -11.265448;
	Eigen::Vector3d tau_timestep_mid;
	tau_timestep_mid << 31.98269367, 9.89625811, 1.47810165;
	Eigen::Vector3d tau_timestep_end;
	tau_timestep_end << 57.04391384, 4.75360586, -1.66561523;
	result_taumat << tau_timestep_beg.transpose(),
		tau_timestep_mid.transpose(),
		tau_timestep_end.transpose();

	Eigen::Vector3d theta_timestep_beg;
	theta_timestep_beg << 0.10092029, 0.10190511, 0.10160667;
	Eigen::Vector3d theta_timestep_mid;
	theta_timestep_mid << 0.85794085, 1.55124503, 2.80130978;
	Eigen::Vector3d theta_timestep_end;
	theta_timestep_end << 1.56344023, 3.07994906, 4.52269971;
	result_thetamat << theta_timestep_beg.transpose(),
		theta_timestep_mid.transpose(),
		theta_timestep_end.transpose();

	auto controlTraj = mr::SimulateControl<3>(thetalist, dthetalist, g, Ftipmat, Mlist, Glist, Slist, thetamatd, dthetamatd,
											  ddthetamatd, gtilde, Mtildelist, Gtildelist, Kp, Ki, Kd, dt, intRes);
	Eigen::Matrix<double, Eigen::Dynamic, 3> traj_tau = controlTraj.first;
	Eigen::Matrix<double, Eigen::Dynamic, 3> traj_theta = controlTraj.second;
	Eigen::Matrix<double, numTest, 3> traj_tau_timestep;
	traj_tau_timestep << traj_tau.row(0),
		traj_tau.row(int(N / 2) - 1),
		traj_tau.row(N - 1);
	Eigen::Matrix<double, numTest, 3> traj_theta_timestep;
	traj_theta_timestep << traj_theta.row(0),
		traj_theta.row(int(N / 2) - 1),
		traj_theta.row(N - 1);

	ASSERT_TRUE(traj_tau_timestep.isApprox(result_taumat, 4));
	ASSERT_TRUE(traj_theta_timestep.isApprox(result_thetamat, 4));
}