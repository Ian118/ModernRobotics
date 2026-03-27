#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

/*
 * ModernRobotics
 * Adapted from modern_robotics.py provided by modernrobotics.org
 * Provides useful Jacobian and frame representation functions
 */

namespace mr {

/*
 * Function: Find if the value is negligible enough to consider 0
 * Inputs: value to be checked as a double
 * Returns: Boolean of true-ignore or false-can't ignore
 */
inline bool NearZero(const double val) { return (std::abs(val) < .000001); }

/*
 * Function: Returns a normalized version of the input vector
 * Input: Eigen::MatrixXd
 * Output: Eigen::MatrixXd
 * Note: MatrixXd is used instead of VectorXd for the case of row vectors
 * 		Requires a copy
 *		Useful because of the MatrixXd casting
 */
template <int a, int b>
inline Eigen::Matrix<double, a, b>
Normalize(const Eigen::Matrix<double, a, b> &V) {
  return V.normalized();
}

/*
 * Function: Returns the skew symmetric matrix representation of an angular
 * velocity vector Input: Eigen::Vector3d 3x1 angular velocity vector Returns:
 * Eigen::MatrixXd 3x3 skew symmetric matrix
 */
inline Eigen::Matrix3d VecToso3(const Eigen::Vector3d &omg) {
  Eigen::Matrix3d so3mat;
  so3mat << 0, -omg(2), omg(1), omg(2), 0, -omg(0), -omg(1), omg(0), 0;
  return so3mat;
}

/*
 * Function: Returns angular velocity vector represented by the skew symmetric
 * matrix Inputs: Eigen::MatrixXd 3x3 skew symmetric matrix Returns:
 * Eigen::Vector3d 3x1 angular velocity
 */
inline Eigen::Vector3d so3ToVec(const Eigen::Matrix3d &so3mat) {
  return Eigen::Vector3d{so3mat(2, 1), so3mat(0, 2), so3mat(1, 0)};
}

/*
 * Function: Calculate the 6x6 matrix [adV] of the given 6-vector
 * Input: Eigen::VectorXd (6x1)
 * Output: Eigen::MatrixXd (6x6)
 * Note: Can be used to calculate the Lie bracket [V1, V2] = [adV1]V2
 */
Eigen::Matrix<double, 6, 6> ad(Eigen::Vector<double, 6> V) {
  Eigen::Matrix3d omgmat = VecToso3(V.head<3>());

  Eigen::Matrix<double, 6, 6> result;
  result.topLeftCorner<3, 3>() = omgmat;
  result.topRightCorner<3, 3>() = Eigen::Matrix3d::Zero();
  result.bottomLeftCorner<3, 3>() = VecToso3(V.tail<3>());
  result.bottomRightCorner<3, 3>() = omgmat;
  return result;
}

/*
 * Function: Tranlates an exponential rotation into it's individual components
 * Inputs: Exponential rotation (rotation matrix in terms of a rotation axis
 *				and the angle of rotation)
 * Returns: The axis and angle of rotation as [x, y, z, theta]
 */
inline Eigen::AngleAxisd AxisAng3(const Eigen::Vector3d &expc3) {
  return Eigen::AngleAxisd{expc3.norm(), expc3.normalized()};
}

/*
 * Function: Translates an exponential rotation into a rotation matrix
 * Inputs: exponenential representation of a rotation
 * Returns: Rotation matrix
 */
Eigen::Matrix3d MatrixExp3(const Eigen::Matrix3d &so3mat) {
  Eigen::Vector3d omgtheta = so3ToVec(so3mat);
  Eigen::Matrix3d m_ret = Eigen::Matrix3d::Identity();
  double theta = omgtheta.norm();

  if (NearZero(theta)) {
    return m_ret;
  } else {
    Eigen::Matrix3d omgmat = so3mat * (1 / theta);
    return m_ret + std::sin(theta) * omgmat +
           ((1 - std::cos(theta)) * (omgmat * omgmat));
  }
}

/* Function: Computes the matrix logarithm of a rotation matrix
 * Inputs: Rotation matrix
 * Returns: matrix logarithm of a rotation
 */
Eigen::Matrix3d MatrixLog3(const Eigen::Matrix3d &R) {
  double acosinput = (R.trace() - 1) / 2.0;
  Eigen::Matrix3d m_ret = Eigen::Matrix3d::Zero(3, 3);
  if (acosinput >= 1)
    return m_ret;
  else if (acosinput <= -1) {
    Eigen::Vector3d omg;
    if (!NearZero(1 + R(2, 2)))
      omg = (1.0 / std::sqrt(2 * (1 + R(2, 2)))) *
            Eigen::Vector3d(R(0, 2), R(1, 2), 1 + R(2, 2));
    else if (!NearZero(1 + R(1, 1)))
      omg = (1.0 / std::sqrt(2 * (1 + R(1, 1)))) *
            Eigen::Vector3d(R(0, 1), 1 + R(1, 1), R(2, 1));
    else
      omg = (1.0 / std::sqrt(2 * (1 + R(0, 0)))) *
            Eigen::Vector3d(1 + R(0, 0), R(1, 0), R(2, 0));
    m_ret = VecToso3(M_PI * omg);
    return m_ret;
  } else {
    double theta = std::acos(acosinput);
    m_ret = theta / 2.0 / sin(theta) * (R - R.transpose());
    return m_ret;
  }
}

/*
 * Function: Combines a rotation matrix and position vector into a single
 * 				Special Euclidian Group (SE3) homogeneous
 * transformation matrix Inputs: Rotation Matrix (R), Position Vector (p)
 * Returns: Matrix of T = [ [R, p],
 *						    [0, 1] ]
 */
inline Eigen::Isometry3d RpToTrans(const Eigen::Matrix3d &R,
                                   const Eigen::Vector3d &p) {
  Eigen::Isometry3d T;
  T.linear() = R;
  T.translation() = p;
  return T;
}

/*
 * Function: Separates the rotation matrix and position vector from
 *				the transfomation matrix representation
 * Inputs: Homogeneous transformation matrix
 * Returns: std::vector of [rotation matrix, position vector]
 */
inline std::pair<Eigen::Matrix3d, Eigen::Vector3d>
TransToRp(const Eigen::Isometry3d &T) {
  return {T.linear(), T.translation()};
}

/*
 * Function: Translates a spatial velocity vector into a transformation matrix
 * Inputs: Spatial velocity vector [angular velocity, linear velocity]
 * Returns: Transformation matrix
 */
inline Eigen::Matrix4d VecTose3(const Eigen::Vector<double, 6> &V) {
  // Separate angular (exponential representation) and linear velocities and
  // fill in values to the appropriate parts of the transformation matrix
  Eigen::Matrix4d m_ret;
  m_ret << VecToso3(V.head<3>()), V.tail<3>(), 0, 0, 0, 0;

  return m_ret;
}

/* Function: Translates a transformation matrix into a spatial velocity vector
 * Inputs: Transformation matrix
 * Returns: Spatial velocity vector [angular velocity, linear velocity]
 */
inline Eigen::Vector<double, 6> se3ToVec(const Eigen::Matrix4d &T) {
  return {T(2, 1), T(0, 2), T(1, 0), T(0, 3), T(1, 3), T(2, 3)};
}
/*
 * Function: Provides the adjoint representation of a transformation matrix
 *			 Used to change the frame of reference for spatial
 * velocity vectors Inputs: 4x4 Transformation matrix SE(3) Returns: 6x6 Adjoint
 * Representation of the matrix
 */
inline Eigen::Matrix<double, 6, 6> Adjoint(const Eigen::Isometry3d &T) {
  auto rp = TransToRp(T);
  Eigen::Matrix<double, 6, 6> ad_ret;
  ad_ret << rp.first, Eigen::Matrix3d::Zero(3, 3),
      VecToso3(rp.second) * rp.first, rp.first;
  return ad_ret;
}

/*
 * Function: Rotation expanded for screw axis
 * Inputs: se3 matrix representation of exponential coordinates (transformation
 * matrix) Returns: 6x6 Matrix representing the rotation
 */
Eigen::Isometry3d MatrixExp6(const Eigen::Matrix4d &se3mat) {
  // Extract the angular velocity vector from the transformation matrix
  Eigen::Matrix3d se3mat_cut = se3mat.topLeftCorner<3, 3>();
  Eigen::Vector3d omgtheta = so3ToVec(se3mat_cut);
  double theta = omgtheta.norm();

  // If negligible rotation
  if (NearZero(theta)) {
    return Eigen::Isometry3d{
        Eigen::Translation3d{se3mat.topRightCorner<3, 1>()}};
  }
  // If not negligible, MR page 105
  else {
    Eigen::Matrix3d omgmat = se3mat_cut / theta;
    Eigen::Matrix3d expExpand = Eigen::Matrix3d::Identity() * theta +
                                (1 - std::cos(theta)) * omgmat +
                                ((theta - std::sin(theta)) * (omgmat * omgmat));
    Eigen::Vector3d linear(se3mat(0, 3), se3mat(1, 3), se3mat(2, 3));
    Eigen::Vector3d GThetaV = (expExpand * linear) / theta;
    return RpToTrans(MatrixExp3(se3mat_cut), GThetaV);
  }
}

/*
 * Function: Computes the matrix logarithm of a homogeneous transformation
 * matrix Inputs: R: Transformation matrix in SE3 Returns: The matrix logarithm
 * of R
 */
Eigen::Matrix4d MatrixLog6(const Eigen::Isometry3d &T) {
  Eigen::Matrix4d m_ret{Eigen::Matrix4d::Zero()};
  Eigen::Matrix3d omgmat = MatrixLog3(T.rotation());

  if (NearZero(omgmat.norm())) {
    m_ret.topLeftCorner<3, 1>() = T.translation();
    return m_ret;
  }

  double theta = std::acos((T.rotation().trace() - 1) / 2.0);
  m_ret.topLeftCorner<3, 3>() = omgmat;
  m_ret.topRightCorner<3, 1>() =
      (Eigen::Matrix3d::Identity() - omgmat / 2.0 +
       (1.0 / theta - 1.0 / std::tan(theta / 2.0) / 2) * omgmat * omgmat /
           theta) *
      T.translation();
  return m_ret;
}

/*
 * Function: Compute end effector frame (used for current spatial position
 * calculation) Inputs: Home configuration (position and orientation) of
 * end-effector The joint screw axes in the space frame when the manipulator is
 * at the home position A list of joint coordinates. Returns: Transfomation
 * matrix representing the end-effector frame when the joints are at the
 * specified coordinates Notes: FK means Forward Kinematics
 */
template <int njoints>
Eigen::Isometry3d FKinSpace(const Eigen::Isometry3d &M,
                            const Eigen::Matrix<double, 6, njoints> &Slist,
                            const Eigen::Vector<double, njoints> &thetaList) {
  Eigen::Isometry3d T = M;
  for (int i = (njoints - 1); i > -1; i--) {
    T = MatrixExp6(VecTose3(Slist.col(i) * thetaList(i))) * T;
  }
  return T;
}

/*
 * Function: Compute end effector frame (used for current body position
 * calculation) Inputs: Home configuration (position and orientation) of
 * end-effector The joint screw axes in the body frame when the manipulator is
 * at the home position A list of joint coordinates. Returns: Transfomation
 * matrix representing the end-effector frame when the joints are at the
 * specified coordinates Notes: FK means Forward Kinematics
 */
template <int njoints>
Eigen::Isometry3d FKinBody(const Eigen::Isometry3d &M,
                           const Eigen::Matrix<double, 6, njoints> &Blist,
                           const Eigen::Vector<double, njoints> &thetaList) {
  Eigen::Isometry3d T = M;
  for (int i = 0; i < njoints; i++) {
    T = T * MatrixExp6(VecTose3(Blist.col(i) * thetaList(i)));
  }
  return T;
}

/*
 * Function: Gives the space Jacobian
 * Inputs: Screw axis in home position, joint configuration
 * Returns: 6xn Spatial Jacobian
 */
template <int njoints>
Eigen::Matrix<double, 6, njoints>
JacobianSpace(const Eigen::Matrix<double, 6, njoints> &Slist,
              const Eigen::Vector<double, njoints> &thetaList) {
  Eigen::Matrix<double, 6, njoints> Js = Slist;
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  Eigen::Vector<double, 6> sListTemp;
  for (int i = 1; i < njoints; i++) {
    sListTemp << Slist.col(i - 1) * thetaList(i - 1);
    T = T * MatrixExp6(VecTose3(sListTemp));
    Js.col(i) = Adjoint(T) * Slist.col(i);
  }
  return Js;
}

/*
 * Function: Gives the body Jacobian
 * Inputs: Screw axis in BODY position, joint configuration
 * Returns: 6xn Bobdy Jacobian
 */
template <int njoints>
Eigen::Matrix<double, 6, njoints>
JacobianBody(const Eigen::Matrix<double, 6, njoints> &Blist,
             const Eigen::Vector<double, njoints> &thetaList) {
  Eigen::Matrix<double, 6, njoints> Jb = Blist;
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  Eigen::Vector<double, 6> bListTemp(Blist.col(0).size());
  for (int i = thetaList.size() - 2; i >= 0; i--) {
    bListTemp << Blist.col(i + 1) * thetaList(i + 1);
    T = T * MatrixExp6(VecTose3(-1 * bListTemp));
    Jb.col(i) = Adjoint(T) * Blist.col(i);
  }
  return Jb;
}

/*
 * Inverts a homogeneous transformation matrix
 * Inputs: A homogeneous transformation Matrix T
 * Returns: The inverse of T
 */
inline Eigen::Isometry3d TransInv(const Eigen::Isometry3d &transform) {
  return transform.inverse();
}

/*
 * Inverts a rotation matrix
 * Inputs: A rotation matrix  R
 * Returns: The inverse of R
 */
inline Eigen::Matrix3d RotInv(const Eigen::Matrix3d &rotMatrix) {
  return rotMatrix.transpose();
}

/*
 * Takes a parametric description of a screw axis and converts it to a
 * normalized screw axis
 * Inputs:
 * q: A point lying on the screw axis
 * s: A unit vector in the direction of the screw axis
 * h: The pitch of the screw axis
 * Returns: A normalized screw axis described by the inputs
 */
inline Eigen::Vector<double, 6> ScrewToAxis(const Eigen::Vector3d &q,
                                            const Eigen::Vector3d &s,
                                            const double &h) {
  Eigen::Vector<double, 6> axis;
  axis << s, q.cross(s) + h * s;
  return axis;
}

/*
 * Function: Translates a 6-vector of exponential coordinates into screw
 * axis-angle form
 * Inputs:
 * expc6: A 6-vector of exponential coordinates for rigid-body motion
          S*theta
 * Returns: The corresponding normalized screw axis S; The distance theta
 traveled
 * along/about S in form [S, theta]
 */
Eigen::Vector<double, 7> AxisAng6(const Eigen::Vector<double, 6> &expc6) {
  Eigen::Vector<double, 7> v_ret;
  double theta = Eigen::Vector3d(expc6(0), expc6(1), expc6(2)).norm();
  if (NearZero(theta))
    theta = Eigen::Vector3d(expc6(3), expc6(4), expc6(5)).norm();
  v_ret << expc6 / theta, theta;
  return v_ret;
}

/*
 * Function: Returns projection of one matrix into SO(3)
 * Inputs:
 * M:		A matrix near SO(3) to project to SO(3)
 * Returns: The closest matrix R that is in SO(3)
 * Projects a matrix mat to the closest matrix in SO(3) using singular-value
 * decomposition (see
 * http://hades.mech.northwestern.edu/index.php/Modern_Robotics_Linear_Algebra_Review).
 * This function is only appropriate for matrices close to SO(3).
 */
Eigen::Matrix3d ProjectToSO3(const Eigen::Matrix3d &M) {
  Eigen::JacobiSVD<Eigen::Matrix3d, Eigen::ComputeFullU | Eigen::ComputeFullV>
      svd(M);
  Eigen::Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
  if (R.determinant() < 0)
    // In this case the result may be far from M; reverse sign of 3rd column
    R.col(2) *= -1;
  return R;
}

/*
 * Function: Returns projection of one matrix into SE(3)
 * Inputs:
 * M:		A 4x4 matrix near SE(3) to project to SE(3)
 * Returns: The closest matrix T that is in SE(3)
 * Projects a matrix mat to the closest matrix in SO(3) using singular-value
 * decomposition (see
 * http://hades.mech.northwestern.edu/index.php/Modern_Robotics_Linear_Algebra_Review).
 * This function is only appropriate for matrices close to SE(3).
 */
inline Eigen::Isometry3d ProjectToSE3(const Eigen::Matrix4d &M) {
  return RpToTrans(ProjectToSO3(M.topLeftCorner<3, 3>()),
                   M.topRightCorner<3, 1>());
}

/*
 * Function: Returns the Frobenius norm to describe the distance of M from the
 * SO(3) manifold Inputs: M: A 3x3 matrix Outputs: the distance from mat to the
 * SO(3) manifold using the following method: If det(M) <= 0, return a large
 * number. If det(M) > 0, return norm(M^T*M - I).
 */
inline double DistanceToSO3(const Eigen::Matrix3d &M) {
  if (M.determinant() > 0)
    return (M.transpose() * M - Eigen::Matrix3d::Identity()).norm();
  else
    return 1.0e9;
}

/*
 * Function: Returns the Frobenius norm to describe the distance of mat from the
 * SE(3) manifold Inputs: T: A 4x4 matrix Outputs: the distance from T to the
 * SE(3) manifold using the following method: Compute the determinant of matR,
 * the top 3x3 submatrix of T. If det(matR) <= 0, return a large number. If
 * det(matR) > 0, replace the top 3x3 submatrix of mat with matR^T*matR, and set
 * the first three entries of the fourth column of mat to zero. Then return
 * norm(T - I).
 */
double DistanceToSE3(const Eigen::Matrix4d &T) {
  Eigen::Matrix3d matR = T.topLeftCorner<3, 3>();
  if (matR.determinant() > 0) {
    Eigen::Matrix4d m_ret;
    m_ret << matR.transpose() * matR, Eigen::Vector3d::Zero(3), T.row(3);
    m_ret = m_ret - Eigen::Matrix4d::Identity();
    return m_ret.norm();
  } else
    return 1.0e9;
}

/*
 * Function: Returns true if M is close to or on the manifold SO(3)
 * Inputs:
 * M: A 3x3 matrix
 * Outputs:
 *	 true if M is very close to or in SO(3), false otherwise
 */
inline bool TestIfSO3(const Eigen::Matrix3d &M) {
  return std::abs(DistanceToSO3(M)) < 1e-3;
}

/*
 * Function: Returns true if T is close to or on the manifold SE(3)
 * Inputs:
 * M: A 4x4 matrix
 * Outputs:
 *	 true if T is very close to or in SE(3), false otherwise
 */
inline bool TestIfSE3(const Eigen::Matrix4d &T) {
  return std::abs(DistanceToSE3(T)) < 1e-3;
}

/*
 * Function: Computes inverse kinematics in the body frame for an open chain
 * robot Inputs: Blist: The joint screw axes in the end-effector frame when the
 *         manipulator is at the home position, in the format of a
 *         matrix with axes as the columns
 *	M: The home configuration of the end-effector
 *	T: The desired end-effector configuration Tsd
 *	thetalist[in][out]: An initial guess and result output of joint angles
 * that are close to satisfying Tsd emog: A small positive tolerance on the
 * end-effector orientation error. The returned joint angles must give an
 * end-effector orientation error less than eomg ev: A small positive tolerance
 * on the end-effector linear position error. The returned joint angles must
 * give an end-effector position error less than ev Outputs: success: A logical
 * value where TRUE means that the function found a solution and FALSE means
 * that it ran through the set number of maximum iterations without finding a
 * solution within the tolerances eomg and ev. thetalist[in][out]: Joint angles
 * that achieve T within the specified tolerances,
 */
template <int njoints>
bool IKinBody(const Eigen::Matrix<double, 6, njoints> &Blist,
              const Eigen::Isometry3d &M, const Eigen::Isometry3d &T,
              Eigen::Vector<double, njoints> &thetalist, double eomg,
              double ev) {
  int i = 0;
  int maxiterations = 20;
  Eigen::Isometry3d Tfk = FKinBody<njoints>(M, Blist, thetalist);
  Eigen::Isometry3d Tdiff = TransInv(Tfk) * T;
  Eigen::Vector<double, 6> Vb = se3ToVec(MatrixLog6(Tdiff));
  Eigen::Vector3d angular{Vb.head<3>()};
  Eigen::Vector3d linear{Vb.tail<3>()};

  bool err = angular.norm() > eomg || linear.norm() > ev;
  Eigen::Matrix<double, 6, Eigen::Dynamic> Jb;
  while (err && i < maxiterations) {
    Jb = JacobianBody(Blist, thetalist);
    thetalist +=
        Jb.bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>().solve(Vb);
    i += 1;
    // iterate
    Tfk = FKinBody<njoints>(M, Blist, thetalist);
    Tdiff = TransInv(Tfk) * T;
    Vb = se3ToVec(MatrixLog6(Tdiff));
    angular = Eigen::Vector3d{Vb.head<3>()};
    linear = Eigen::Vector3d{Vb.tail<3>()};
    err = angular.norm() > eomg || linear.norm() > ev;
  }
  return !err;
}

/*
 * Function: Computes inverse kinematics in the space frame for an open chain
 * robot Inputs: Slist: The joint screw axes in the space frame when the
 *         manipulator is at the home position, in the format of a
 *         matrix with axes as the columns
 *	M: The home configuration of the end-effector
 *	T: The desired end-effector configuration Tsd
 *	thetalist[in][out]: An initial guess and result output of joint angles
 * that are close to satisfying Tsd emog: A small positive tolerance on the
 * end-effector orientation error. The returned joint angles must give an
 * end-effector orientation error less than eomg ev: A small positive tolerance
 * on the end-effector linear position error. The returned joint angles must
 * give an end-effector position error less than ev Outputs: success: A logical
 * value where TRUE means that the function found a solution and FALSE means
 * that it ran through the set number of maximum iterations without finding a
 * solution within the tolerances eomg and ev. thetalist[in][out]: Joint angles
 * that achieve T within the specified tolerances,
 */
template <int njoints>
bool IKinSpace(const Eigen::Matrix<double, 6, njoints> &Slist,
               const Eigen::Isometry3d &M, const Eigen::Isometry3d &T,
               Eigen::Vector<double, njoints> &thetalist, double eomg,
               double ev) {
  int i = 0;
  int maxiterations = 20;
  Eigen::Isometry3d Tfk = FKinSpace<njoints>(M, Slist, thetalist);
  Eigen::Isometry3d Tdiff = TransInv(Tfk) * T;
  Eigen::Vector<double, 6> Vs = Adjoint(Tfk) * se3ToVec(MatrixLog6(Tdiff));
  Eigen::Vector3d angular{Vs.head<3>()};
  Eigen::Vector3d linear{Vs.tail<3>()};

  bool err = angular.norm() > eomg || linear.norm() > ev;
  Eigen::MatrixXd Js;
  while (err && i < maxiterations) {
    Js = JacobianSpace(Slist, thetalist);
    thetalist +=
        Js.bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>().solve(Vs);
    i += 1;
    // iterate
    Tfk = FKinSpace<njoints>(M, Slist, thetalist);
    Tdiff = TransInv(Tfk) * T;
    Vs = Adjoint(Tfk) * se3ToVec(MatrixLog6(Tdiff));
    angular = Eigen::Vector3d{Vs.head<3>()};
    linear = Eigen::Vector3d{Vs.tail<3>()};
    err = angular.norm() > eomg || linear.norm() > ev;
  }
  return !err;
}

/*
 * Function: This function uses forward-backward Newton-Euler iterations to
 * solve the equation: taulist = Mlist(thetalist) * ddthetalist + c(thetalist,
 * dthetalist) ...
 *           + g(thetalist) + Jtr(thetalist) * Ftip
 * Inputs:
 *  thetalist: n-vector of joint variables
 *  dthetalist: n-vector of joint rates
 *  ddthetalist: n-vector of joint accelerations
 *  g: Gravity vector g
 *  Ftip: Spatial force applied by the end-effector expressed in frame {n+1}
 *  Mlist: List of link frames {i} relative to {i-1} at the home position
 *  Glist: Spatial inertia matrices Gi of the links
 *  Slist: Screw axes Si of the joints in a space frame, in the format
 *         of a matrix with the screw axes as the columns.
 *
 * Outputs:
 *  taulist: The n-vector of required joint forces/torques
 *
 */
template <int njoints>
Eigen::Vector<double, njoints>
InverseDynamics(const Eigen::Vector<double, njoints> &thetalist,
                const Eigen::Vector<double, njoints> &dthetalist,
                const Eigen::Vector<double, njoints> &ddthetalist,
                const Eigen::Vector3d &g, const Eigen::Vector<double, 6> &Ftip,
                const std::array<Eigen::Isometry3d, njoints + 1> &Mlist,
                const std::array<Eigen::Matrix<double, 6, 6>, njoints> &Glist,
                const Eigen::Matrix<double, 6, njoints> &Slist) {
  Eigen::Isometry3d Mi{Eigen::Isometry3d::Identity()};
  Eigen::Matrix<double, 6, njoints> Ai{
      Eigen::Matrix<double, 6, njoints>::Zero()};
  Eigen::Matrix<double, 6, njoints + 1> Vi{
      Eigen::Matrix<double, 6, njoints + 1>::Zero()}; // velocity
  Eigen::Matrix<double, 6, njoints + 1> Vdi{
      Eigen::Matrix<double, 6, njoints + 1>::Zero()}; // acceleration
  Vdi.col(0) << Eigen::Vector3d::Zero(), -g;
  std::array<Eigen::Matrix<double, 6, 6>, njoints + 1> AdTi;
  AdTi[njoints] = Adjoint(TransInv(Mlist[njoints]));
  Eigen::Vector<double, 6> Fi = Ftip;

  Eigen::Vector<double, njoints> taulist;

  // forward pass
  for (int i = 0; i < njoints; i++) {
    Mi = Mi * Mlist[i];
    Ai.col(i) = Adjoint(TransInv(Mi)) * Slist.col(i);

    AdTi[i] = Adjoint(MatrixExp6(VecTose3(Ai.col(i) * -thetalist(i))) *
                      TransInv(Mlist[i]));

    Vi.col(i + 1) = AdTi[i] * Vi.col(i) + Ai.col(i) * dthetalist(i);
    Vdi.col(i + 1) = AdTi[i] * Vdi.col(i) + Ai.col(i) * ddthetalist(i) +
                     ad(Vi.col(i + 1)) * Ai.col(i) * dthetalist(i);
  }

  // backward pass
  for (int i = njoints - 1; i >= 0; i--) {
    Fi = AdTi[i + 1].transpose() * Fi + Glist[i] * Vdi.col(i + 1) -
         ad(Vi.col(i + 1)).transpose() * (Glist[i] * Vi.col(i + 1));
    taulist(i) = Fi.transpose() * Ai.col(i);
  }
  return taulist;
}

/*
 * Function: This function calls InverseDynamics with Ftip = 0, dthetalist = 0,
 * and ddthetalist = 0. The purpose is to calculate one important term in the
 * dynamics equation Inputs: thetalist: n-vector of joint variables g: Gravity
 * vector g Mlist: List of link frames {i} relative to {i-1} at the home
 * position Glist: Spatial inertia matrices Gi of the links Slist: Screw axes Si
 * of the joints in a space frame, in the format of a matrix with the screw axes
 * as the columns.
 *
 * Outputs:
 *  grav: The 3-vector showing the effect force of gravity to the dynamics
 *
 */
template <int njoints>
inline Eigen::Vector<double, njoints>
GravityForces(const Eigen::Vector<double, njoints> &thetalist,
              const Eigen::Vector3d &g,
              const std::array<Eigen::Isometry3d, njoints + 1> &Mlist,
              const std::array<Eigen::Matrix<double, 6, 6>, njoints> &Glist,
              const Eigen::Matrix<double, 6, njoints> &Slist) {
  return InverseDynamics<njoints>(
      thetalist, Eigen::Vector<double, njoints>::Zero(),
      Eigen::Vector<double, njoints>::Zero(), g,
      Eigen::Vector<double, 6>::Zero(), Mlist, Glist, Slist);
}

/*
 * Function: This function calls InverseDynamics n times, each time passing a
 * ddthetalist vector with a single element equal to one and all other
 * inputs set to zero. Each call of InverseDynamics generates a single
 * column, and these columns are assembled to create the inertia matrix.
 *
 * Inputs:
 *  thetalist: n-vector of joint variables
 *  Mlist: List of link frames {i} relative to {i-1} at the home position
 *  Glist: Spatial inertia matrices Gi of the links
 *  Slist: Screw axes Si of the joints in a space frame, in the format
 *         of a matrix with the screw axes as the columns.
 *
 * Outputs:
 *  M: The numerical inertia matrix M(thetalist) of an n-joint serial
 *     chain at the given configuration thetalist.
 */
template <int njoints>
Eigen::Matrix<double, njoints, njoints>
MassMatrix(const Eigen::Vector<double, njoints> &thetalist,
           const std::array<Eigen::Isometry3d, njoints + 1> &Mlist,
           const std::array<Eigen::Matrix<double, 6, 6>, njoints> &Glist,
           const Eigen::Matrix<double, 6, njoints> &Slist) {
  auto dummylist{Eigen::Vector<double, njoints>::Zero()};
  auto dummyg{Eigen::Vector3d::Zero()};
  auto dummyforce{Eigen::Vector<double, 6>::Zero()};
  Eigen::Matrix<double, njoints, njoints> M{
      Eigen::Matrix<double, njoints, njoints>::Zero()};
  Eigen::Vector<double, njoints> ddthetalist;
  for (int i = 0; i < njoints; i++) {
    ddthetalist.setZero();
    ddthetalist(i) = 1;
    M.col(i) =
        InverseDynamics<njoints>(thetalist, dummylist, ddthetalist, dummyg,
                                 dummyforce, Mlist, Glist, Slist);
  }
  return M;
}

/*
 * Function: This function calls InverseDynamics with g = 0, Ftip = 0, and
 * ddthetalist = 0.
 *
 * Inputs:
 *  thetalist: n-vector of joint variables
 *  dthetalist: A list of joint rates
 *  Mlist: List of link frames {i} relative to {i-1} at the home position
 *  Glist: Spatial inertia matrices Gi of the links
 *  Slist: Screw axes Si of the joints in a space frame, in the format
 *         of a matrix with the screw axes as the columns.
 *
 * Outputs:
 *  c: The vector c(thetalist,dthetalist) of Coriolis and centripetal
 *     terms for a given thetalist and dthetalist.
 */
template <int njoints>
inline Eigen::Vector<double, njoints> VelQuadraticForces(
    const Eigen::Vector<double, njoints> &thetalist,
    const Eigen::Vector<double, njoints> &dthetalist,
    const std::array<Eigen::Isometry3d, njoints + 1> &Mlist,
    const std::array<Eigen::Matrix<double, 6, 6>, njoints> &Glist,
    const Eigen::Matrix<double, 6, njoints> &Slist) {
  return InverseDynamics<njoints>(
      thetalist, dthetalist, Eigen::Vector<double, njoints>::Zero(),
      Eigen::Vector3d::Zero(), Eigen::Vector<double, 6>::Zero(), Mlist, Glist,
      Slist);
}

/*
 * Function: This function calls InverseDynamics with g = 0, dthetalist = 0, and
 * ddthetalist = 0.
 *
 * Inputs:
 *  thetalist: n-vector of joint variables
 *  Ftip: Spatial force applied by the end-effector expressed in frame {n+1}
 *  Mlist: List of link frames {i} relative to {i-1} at the home position
 *  Glist: Spatial inertia matrices Gi of the links
 *  Slist: Screw axes Si of the joints in a space frame, in the format
 *         of a matrix with the screw axes as the columns.
 *
 * Outputs:
 *  JTFtip: The joint forces and torques required only to create the
 *     end-effector force Ftip.
 */
template <int njoints>
inline Eigen::Vector<double, njoints>
EndEffectorForces(const Eigen::Vector<double, njoints> &thetalist,
                  const Eigen::Vector<double, 6> &Ftip,
                  const std::array<Eigen::Isometry3d, njoints + 1> &Mlist,
                  const std::array<Eigen::Matrix<double, 6, 6>, njoints> &Glist,
                  const Eigen::Matrix<double, 6, njoints> &Slist) {
  return InverseDynamics<njoints>(
      thetalist, Eigen::Vector<double, njoints>::Zero(),
      Eigen::Vector<double, njoints>::Zero(), Eigen::Vector3d::Zero(), Ftip,
      Mlist, Glist, Slist);
}

/*
 * Function: This function computes ddthetalist by solving:
 * Mlist(thetalist) * ddthetalist = taulist - c(thetalist,dthetalist)
 *                                  - g(thetalist) - Jtr(thetalist) * Ftip
 * Inputs:
 *  thetalist: n-vector of joint variables
 *  dthetalist: n-vector of joint rates
 *  taulist: An n-vector of joint forces/torques
 *  g: Gravity vector g
 *  Ftip: Spatial force applied by the end-effector expressed in frame {n+1}
 *  Mlist: List of link frames {i} relative to {i-1} at the home position
 *  Glist: Spatial inertia matrices Gi of the links
 *  Slist: Screw axes Si of the joints in a space frame, in the format
 *         of a matrix with the screw axes as the columns.
 *
 * Outputs:
 *  ddthetalist: The resulting joint accelerations
 *
 */
template <int njoints>
Eigen::Vector<double, njoints>
ForwardDynamics(const Eigen::Vector<double, njoints> &thetalist,
                const Eigen::Vector<double, njoints> &dthetalist,
                const Eigen::Vector<double, njoints> &taulist,
                const Eigen::Vector3d &g, const Eigen::Vector<double, 6> &Ftip,
                const std::array<Eigen::Isometry3d, njoints + 1> &Mlist,
                const std::array<Eigen::Matrix<double, 6, 6>, njoints> &Glist,
                const Eigen::Matrix<double, 6, njoints> &Slist) {
  Eigen::Vector<double, njoints> totalForce =
      taulist -
      VelQuadraticForces<njoints>(thetalist, dthetalist, Mlist, Glist, Slist) -
      GravityForces<njoints>(thetalist, g, Mlist, Glist, Slist) -
      EndEffectorForces<njoints>(thetalist, Ftip, Mlist, Glist, Slist);

  Eigen::Matrix<double, njoints, njoints> M =
      MassMatrix<njoints>(thetalist, Mlist, Glist, Slist);

  // Use LDLT since M is positive definite
  Eigen::Vector<double, njoints> ddthetalist = M.ldlt().solve(totalForce);

  return ddthetalist;
}

/*
 * Function: Compute the joint angles and velocities at the next timestep using
    first order Euler integration
 * Inputs:
 *  thetalist[in]: n-vector of joint variables
 *  dthetalist[in]: n-vector of joint rates
 *	ddthetalist: n-vector of joint accelerations
 *  dt: The timestep delta t
 *
 * Outputs:
 *  thetalist[out]: Vector of joint variables after dt from first order Euler
 integration
 *  dthetalist[out]: Vector of joint rates after dt from first order Euler
 integration
 */
template <int njoints>
inline void EulerStep(Eigen::Vector<double, njoints> &thetalist,
                      Eigen::Vector<double, njoints> &dthetalist,
                      const Eigen::Vector<double, njoints> &ddthetalist,
                      const double &dt) {
  thetalist += dthetalist * dt;
  dthetalist += ddthetalist * dt;
  return;
}

/*
 * Function: Compute the joint forces/torques required to move the serial chain
 * along the given trajectory using inverse dynamics Inputs: thetamat: An N x n
 * matrix of robot joint variables (N: no. of trajecoty time step points; n: no.
 * of robot joints dthetamat: An N x n matrix of robot joint velocities
 *  ddthetamat: An N x n matrix of robot joint accelerations
 *	g: Gravity vector g
 *	Ftipmat: An N x 6 matrix of spatial forces applied by the end-effector
 * (if there are no tip forces the user should input a zero matrix) Mlist: List
 * of link frames {i} relative to {i-1} at the home position Glist: Spatial
 * inertia matrices Gi of the links Slist: Screw axes Si of the joints in a
 * space frame, in the format of a matrix with the screw axes as the columns.
 *
 * Outputs:
 *  taumat: The N x n matrix of joint forces/torques for the specified
 * trajectory, where each of the N rows is the vector of joint forces/torques at
 * each time step
 */
template <int njoints, int npoints>
Eigen::Matrix<double, npoints, njoints> InverseDynamicsTrajectory(
    const Eigen::Matrix<double, npoints, njoints> &thetamat,
    const Eigen::Matrix<double, npoints, njoints> &dthetamat,
    const Eigen::Matrix<double, npoints, njoints> &ddthetamat,
    const Eigen::Vector3d &g, const Eigen::Matrix<double, npoints, 6> &Ftipmat,
    const std::array<Eigen::Isometry3d, njoints + 1> &Mlist,
    const std::array<Eigen::Matrix<double, 6, 6>, njoints> &Glist,
    const Eigen::Matrix<double, 6, njoints> &Slist) {
  Eigen::Matrix<double, njoints, npoints> thetamatT = thetamat.transpose();
  Eigen::Matrix<double, njoints, npoints> dthetamatT = dthetamat.transpose();
  Eigen::Matrix<double, njoints, npoints> ddthetamatT = ddthetamat.transpose();
  Eigen::Matrix<double, 6, npoints> FtipmatT = Ftipmat.transpose();

  Eigen::Matrix<double, njoints, npoints> taumatT;
  for (int i = 0; i < npoints; ++i) {
    taumatT.col(i) = InverseDynamics<njoints>(
        thetamatT.col(i), dthetamatT.col(i), ddthetamatT.col(i), g,
        FtipmatT.col(i), Mlist, Glist, Slist);
  }
  return taumatT.transpose();
}

/*
 * Function: Compute the motion of a serial chain given an open-loop history of
 * joint forces/torques Inputs: thetalist: n-vector of initial joint variables
 *  dthetalist: n-vector of initial joint rates
 *  taumat: An N x n matrix of joint forces/torques, where each row is is the
 * joint effort at any time step g: Gravity vector g Ftipmat: An N x 6 matrix of
 * spatial forces applied by the end-effector (if there are no tip forces the
 * user should input a zero matrix) Mlist: List of link frames {i} relative to
 * {i-1} at the home position Glist: Spatial inertia matrices Gi of the links
 *  Slist: Screw axes Si of the joints in a space frame, in the format
 *         of a matrix with the screw axes as the columns.
 *	dt: The timestep between consecutive joint forces/torques
 *	intRes: Integration resolution is the number of times integration
 * (Euler) takes places between each time step. Must be an integer value greater
 * than or equal to 1
 *
 * Outputs: std::pair of [thetamat, dthetamat]
 *  thetamat: The N x n matrix of joint angles resulting from the specified
 * joint forces/torques dthetamat: The N x n matrix of joint velocities
 */
template <int njoints>
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> ForwardDynamicsTrajectory(
    const Eigen::Vector<double, njoints> &thetalist,
    const Eigen::Vector<double, njoints> &dthetalist,
    const Eigen::MatrixXd &taumat, const Eigen::Vector3d &g,
    const Eigen::MatrixXd &Ftipmat,
    const std::array<Eigen::Isometry3d, njoints + 1> &Mlist,
    const std::array<Eigen::Matrix<double, 6, 6>, njoints> &Glist,
    const Eigen::Matrix<double, 6, njoints> &Slist, const double &dt,
    const int &intRes) {
  double npoints = taumat.rows();
  Eigen::MatrixXd FtipmatT = Ftipmat.transpose();
  Eigen::MatrixXd thetamatT = Eigen::MatrixXd::Zero(njoints, npoints);
  Eigen::MatrixXd dthetamatT = Eigen::MatrixXd::Zero(njoints, npoints);
  thetamatT.col(0) = thetalist;
  dthetamatT.col(0) = dthetalist;
  Eigen::Vector<double, njoints> thetacurrent = thetalist;
  Eigen::Vector<double, njoints> dthetacurrent = dthetalist;
  Eigen::Vector<double, njoints> ddthetalist;
  for (int i = 0; i < npoints - 1; ++i) {
    for (int j = 0; j < intRes; ++j) {
      ddthetalist =
          ForwardDynamics<njoints>(thetacurrent, dthetacurrent, taumat.row(i),
                                   g, FtipmatT.col(i), Mlist, Glist, Slist);
      EulerStep<njoints>(thetacurrent, dthetacurrent, ddthetalist, dt / intRes);
    }
    thetamatT.col(i + 1) = thetacurrent;
    dthetamatT.col(i + 1) = dthetacurrent;
  }
  return {thetamatT.transpose(), dthetamatT.transpose()};
}

/*
 * Function: Compute the joint control torques at a particular time instant
 * Inputs:
 *  thetalist: n-vector of joint variables
 *  dthetalist: n-vector of joint rates
 *	eint: n-vector of the time-integral of joint errors
 *	g: Gravity vector g
 *  Mlist: List of link frames {i} relative to {i-1} at the home position
 *  Glist: Spatial inertia matrices Gi of the links
 *  Slist: Screw axes Si of the joints in a space frame, in the format
 *         of a matrix with the screw axes as the columns.
 *  thetalistd: n-vector of reference joint variables
 *  dthetalistd: n-vector of reference joint rates
 *  ddthetalistd: n-vector of reference joint accelerations
 *	Kp: The feedback proportional gain (identical for each joint)
 *	Ki: The feedback integral gain (identical for each joint)
 *	Kd: The feedback derivative gain (identical for each joint)
 *
 * Outputs:
 *  tau_computed: The vector of joint forces/torques computed by the feedback
 *				  linearizing controller at the current instant
 */
template <int njoints>
Eigen::Vector<double, njoints>
ComputedTorque(const Eigen::Vector<double, njoints> &thetalist,
               const Eigen::Vector<double, njoints> &dthetalist,
               const Eigen::Vector<double, njoints> &eint,
               const Eigen::Vector3d &g,
               const std::array<Eigen::Isometry3d, njoints + 1> &Mlist,
               const std::array<Eigen::Matrix<double, 6, 6>, njoints> &Glist,
               const Eigen::Matrix<double, 6, njoints> &Slist,
               const Eigen::Vector<double, njoints> &thetalistd,
               const Eigen::Vector<double, njoints> &dthetalistd,
               const Eigen::Vector<double, njoints> &ddthetalistd,
               const double &Kp, const double &Ki, const double &Kd) {
  Eigen::Vector<double, njoints> e = thetalistd - thetalist; // position err
  Eigen::Vector<double, njoints> tau_feedforward =
      MassMatrix<njoints>(thetalist, Mlist, Glist, Slist) *
      (Kp * e + Ki * (eint + e) + Kd * (dthetalistd - dthetalist));

  Eigen::Vector<double, njoints> tau_inversedyn = InverseDynamics<njoints>(
      thetalist, dthetalist, ddthetalistd, g, Eigen::Vector<double, 6>::Zero(),
      Mlist, Glist, Slist);

  return tau_feedforward + tau_inversedyn;
}

/*
 * Function: Compute s(t) for a cubic time scaling
 * Inputs:
 *  Tf: Total time of the motion in seconds from rest to rest
 *  t: The current time t satisfying 0 < t < Tf
 *
 * Outputs:
 *  st: The path parameter corresponding to a third-order
 *      polynomial motion that begins and ends at zero velocity
 */
inline double CubicTimeScaling(double Tf, double t) {
  double timeratio = 1.0 * t / Tf;
  return 3 * pow(timeratio, 2) - 2 * pow(timeratio, 3);
}

/*
 * Function: Compute s(t) for a quintic time scaling
 * Inputs:
 *  Tf: Total time of the motion in seconds from rest to rest
 *  t: The current time t satisfying 0 < t < Tf
 *
 * Outputs:
 *  st: The path parameter corresponding to a fifth-order
 *      polynomial motion that begins and ends at zero velocity
 *	    and zero acceleration
 */
inline double QuinticTimeScaling(double Tf, double t) {
  double timeratio = 1.0 * t / Tf;
  return 10 * pow(timeratio, 3) - 15 * pow(timeratio, 4) +
         6 * pow(timeratio, 5);
}

/*
 * Function: Compute a straight-line trajectory in joint space
 * Inputs:
 *  thetastart: The initial joint variables
 *  thetaend: The final joint variables
 *  Tf: Total time of the motion in seconds from rest to rest
 *	N: The number of points N > 1 (Start and stop) in the discrete
 *     representation of the trajectory
 *  method: The time-scaling method, where 3 indicates cubic (third-
 *          order polynomial) time scaling and 5 indicates quintic
 *          (fifth-order polynomial) time scaling
 *
 * Outputs:
 *  traj: A trajectory as an N x n matrix, where each row is an n-vector
 *        of joint variables at an instant in time. The first row is
 *        thetastart and the Nth row is thetaend . The elapsed time
 *        between each row is Tf / (N - 1)
 */
template <int njoints>
Eigen::MatrixXd
JointTrajectory(const Eigen::Vector<double, njoints> &thetastart,
                const Eigen::Vector<double, njoints> &thetaend, double Tf,
                int N, int method) {
  double timegap = Tf / (N - 1);
  Eigen::Matrix<double, njoints, Eigen::Dynamic> trajT =
      Eigen::Matrix<double, njoints, Eigen::Dynamic>::Zero(njoints, N);
  double st;
  for (int i = 0; i < N; ++i) {
    if (method == 3)
      st = CubicTimeScaling(Tf, timegap * i);
    else
      st = QuinticTimeScaling(Tf, timegap * i);
    trajT.col(i) = st * thetaend + (1 - st) * thetastart;
  }
  return trajT.transpose();
}

/*
 * Function: Compute a trajectory as a list of N SE(3) matrices corresponding to
 *			 the screw motion about a space screw axis
 * Inputs:
 *  Xstart: The initial end-effector configuration
 *  Xend: The final end-effector configuration
 *  Tf: Total time of the motion in seconds from rest to rest
 *	N: The number of points N > 1 (Start and stop) in the discrete
 *     representation of the trajectory
 *  method: The time-scaling method, where 3 indicates cubic (third-
 *          order polynomial) time scaling and 5 indicates quintic
 *          (fifth-order polynomial) time scaling
 *
 * Outputs:
 *  traj: The discretized trajectory as a list of N matrices in SE(3)
 *        separated in time by Tf/(N-1). The first in the list is Xstart
 *        and the Nth is Xend
 */
std::vector<Eigen::Isometry3d> ScrewTrajectory(const Eigen::Isometry3d &Xstart,
                                               const Eigen::Isometry3d &Xend,
                                               double Tf, int N, int method) {
  double timegap = Tf / (N - 1);
  std::vector<Eigen::Isometry3d> traj;
  double st;
  for (int i = 0; i < N; ++i) {
    if (method == 3)
      st = CubicTimeScaling(Tf, timegap * i);
    else
      st = QuinticTimeScaling(Tf, timegap * i);
    traj.push_back(Xstart *
                   MatrixExp6(MatrixLog6(TransInv(Xstart) * Xend) * st));
  }
  return traj;
}

/*
 * Function: Compute a trajectory as a list of N SE(3) matrices corresponding to
 *			 the origin of the end-effector frame following a
 * straight line Inputs: Xstart: The initial end-effector configuration Xend:
 * The final end-effector configuration Tf: Total time of the motion in seconds
 * from rest to rest N: The number of points N > 1 (Start and stop) in the
 * discrete representation of the trajectory method: The time-scaling method,
 * where 3 indicates cubic (third- order polynomial) time scaling and 5
 * indicates quintic (fifth-order polynomial) time scaling
 *
 * Outputs:
 *  traj: The discretized trajectory as a list of N matrices in SE(3)
 *        separated in time by Tf/(N-1). The first in the list is Xstart
 *        and the Nth is Xend
 * Notes:
 *	This function is similar to ScrewTrajectory, except the origin of the
 *  end-effector frame follows a straight line, decoupled from the rotational
 *  motion.
 */
std::vector<Eigen::Isometry3d>
CartesianTrajectory(const Eigen::Isometry3d &Xstart,
                    const Eigen::Isometry3d &Xend, double Tf, int N,
                    int method) {
  double timegap = Tf / (N - 1);
  std::vector<Eigen::Isometry3d> traj(N);
  double st;
  for (int i = 0; i < N; ++i) {
    if (method == 3)
      st = CubicTimeScaling(Tf, timegap * i);
    else
      st = QuinticTimeScaling(Tf, timegap * i);
    traj.at(i).linear() =
        Xstart.linear() *
        MatrixExp3(MatrixLog3(Xstart.linear().transpose() * Xend.linear()) *
                   st);
    traj.at(i).translation() =
        st * Xstart.translation() + (1 - st) * Xend.translation();
  }
  return traj;
}

/*
 * Function: Compute the motion of a serial chain given an open-loop history of
 * joint forces/torques Inputs: thetalist: n-vector of initial joint variables
 *  dthetalist: n-vector of initial joint rates
 *	g: Gravity vector g
 *	Ftipmat: An N x 6 matrix of spatial forces applied by the end-effector
 * (if there are no tip forces the user should input a zero matrix) Mlist: List
 * of link frames {i} relative to {i-1} at the home position Glist: Spatial
 * inertia matrices Gi of the links Slist: Screw axes Si of the joints in a
 * space frame, in the format of a matrix with the screw axes as the columns.
 *  thetamatd: An Nxn matrix of desired joint variables from the reference
 * trajectory dthetamatd: An Nxn matrix of desired joint velocities ddthetamatd:
 * An Nxn matrix of desired joint accelerations gtilde: The gravity vector based
 * on the model of the actual robot (actual values given above) Mtildelist: The
 * link frame locations based on the model of the actual robot (actual values
 * given above) Gtildelist: The link spatial inertias based on the model of the
 * actual robot (actual values given above) Kp: The feedback proportional gain
 * (identical for each joint) Ki: The feedback integral gain (identical for each
 * joint) Kd: The feedback derivative gain (identical for each joint) dt: The
 * timestep between points on the reference trajectory intRes: Integration
 * resolution is the number of times integration (Euler) takes places between
 * each time step. Must be an integer value greater than or equal to 1
 *
 * Outputs: std::pair of [taumat, thetamat]
 *  taumat: An Nxn matrix of the controllers commanded joint forces/ torques,
 * where each row of n forces/torques corresponds to a single time instant
 *  thetamat: The N x n matrix of actual joint angles
 */
template <int njoints>
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> SimulateControl(
    const Eigen::Vector<double, njoints> &thetalist,
    const Eigen::Vector<double, njoints> &dthetalist, const Eigen::Vector3d &g,
    const Eigen::MatrixXd &Ftipmat,
    const std::array<Eigen::Isometry3d, njoints + 1> &Mlist,
    const std::array<Eigen::Matrix<double, 6, 6>, njoints> &Glist,
    const Eigen::Matrix<double, 6, njoints> &Slist,
    const Eigen::MatrixXd &thetamatd, const Eigen::MatrixXd &dthetamatd,
    const Eigen::MatrixXd &ddthetamatd, const Eigen::Vector3d &gtilde,
    const std::array<Eigen::Isometry3d, njoints + 1> &Mtildelist,
    const std::array<Eigen::Matrix<double, 6, 6>, njoints> &Gtildelist,
    double Kp, double Ki, double Kd, double dt, int intRes) {
  Eigen::Matrix<double, 6, Eigen::Dynamic> FtipmatT = Ftipmat.transpose();
  Eigen::MatrixXd thetamatdT = thetamatd.transpose();
  Eigen::MatrixXd dthetamatdT = dthetamatd.transpose();
  Eigen::MatrixXd ddthetamatdT = ddthetamatd.transpose();

  int npoints = thetamatd.rows();
  Eigen::Vector<double, njoints> thetacurrent = thetalist;
  Eigen::Vector<double, njoints> dthetacurrent = dthetalist;
  Eigen::Vector<double, njoints> eint = Eigen::Vector<double, njoints>::Zero();
  Eigen::MatrixXd taumatT = Eigen::MatrixXd::Zero(njoints, npoints);
  Eigen::MatrixXd thetamatT = Eigen::MatrixXd::Zero(njoints, npoints);
  Eigen::Vector<double, njoints> taulist;
  Eigen::Vector<double, njoints> ddthetalist;
  for (int i = 0; i < njoints; ++i) {
    taulist = ComputedTorque<njoints>(thetacurrent, dthetacurrent, eint, gtilde,
                                      Mtildelist, Gtildelist, Slist,
                                      thetamatdT.col(i), dthetamatdT.col(i),
                                      ddthetamatdT.col(i), Kp, Ki, Kd);
    for (int j = 0; j < intRes; ++j) {
      ddthetalist =
          ForwardDynamics<njoints>(thetacurrent, dthetacurrent, taulist, g,
                                   FtipmatT.col(i), Mlist, Glist, Slist);
      EulerStep(thetacurrent, dthetacurrent, ddthetalist, dt / intRes);
    }
    taumatT.col(i) = taulist;
    thetamatT.col(i) = thetacurrent;
    eint += dt * (thetamatdT.col(i) - thetacurrent);
  }
  return {taumatT.transpose(), thetamatT.transpose()};
}

} // namespace mr
