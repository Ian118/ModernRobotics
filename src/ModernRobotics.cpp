#include "ModernRobotics"

/*
 * modernRobotics.cpp
 * Adapted from modern_robotics.py provided by modernrobotics.org
 * Provides useful Jacobian and frame representation functions
 */
#include <cmath>

#define M_PI 3.14159265358979323846 /* pi */

namespace mr {

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

/* Function: Translates an exponential rotation into a rotation matrix
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

/* Function: Translates a spatial velocity vector into a transformation matrix
 * Inputs: Spatial velocity vector [angular velocity, linear velocity]
 * Returns: Transformation matrix
 */
Eigen::Matrix4d VecTose3(const Eigen::Vector<double, 6> &V) {
  // Separate angular (exponential representation) and linear velocities and
  // fill in values to the appropriate parts of the transformation matrix
  Eigen::Matrix4d m_ret;
  m_ret << VecToso3(V.head<3>()), V.tail<3>(), 0, 0, 0, 0;

  return m_ret;
}


/* Function: Provides the adjoint representation of a transformation matrix
 *			 Used to change the frame of reference for spatial
 * velocity vectors Inputs: 4x4 Transformation matrix SE(3) Returns: 6x6 Adjoint
 * Representation of the matrix
 */
Eigen::Matrix<double, 6, 6> Adjoint(const Eigen::Isometry3d &T) {
  auto rp = TransToRp(T);
  Eigen::Matrix<double, 6, 6> ad_ret;
  ad_ret << rp.first, Eigen::Matrix3d::Zero(3, 3),
      VecToso3(rp.second) * rp.first, rp.first;
  return ad_ret;
}

/* Function: Rotation expanded for screw axis
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

/* Function: Compute end effector frame (used for current spatial position
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

/* Function: Gives the space Jacobian
 * Inputs: Screw axis in home position, joint configuration
 * Returns: 6xn Spatial Jacobian
 */
template <int njoints>
Eigen::Matrix<double, 6, njoints>
JacobianSpace(const Eigen::Matrix<double, 6, njoints> &Slist,
              const Eigen::Vector<double, njoints> &thetaList) {
  Eigen::Matrix<double, 6, njoints> Js = Slist;
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  Eigen::Vector<double, njoints> sListTemp;
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
  Eigen::Vector<double, njoints> bListTemp(Blist.col(0).size());
  for (int i = thetaList.size() - 2; i >= 0; i--) {
    bListTemp << Blist.col(i + 1) * thetaList(i + 1);
    T = T * MatrixExp6(VecTose3(-1 * bListTemp));
    Jb.col(i) = Adjoint(T) * Blist.col(i);
  }
  return Jb;
}

std::pair<Eigen::Vector<double, 6>, double>
AxisAng6(const Eigen::Vector<double, 6> &expc6) {
  Eigen::Vector<double, 6> v_ret;
  double theta = Eigen::Vector3d(expc6(0), expc6(1), expc6(2)).norm();
  if (NearZero(theta))
    theta = Eigen::Vector3d(expc6(3), expc6(4), expc6(5)).norm();
  v_ret << expc6 / theta;
  return std::make_pair(v_ret, theta);
}

Eigen::Matrix3d ProjectToSO3(const Eigen::Matrix3d &M) {
  Eigen::JacobiSVD<Eigen::Matrix3d, Eigen::ComputeFullU | Eigen::ComputeFullV>
      svd(M);
  Eigen::Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
  if (R.determinant() < 0)
    // In this case the result may be far from M; reverse sign of 3rd column
    R.col(2) *= -1;
  return R;
}

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

template <int njoints>
bool IKinSpace(const Eigen::Matrix<double, 6, njoints> &Slist,
               const Eigen::Isometry3d &M, const Eigen::Isometry3d &T,
               Eigen::Vector<double, njoints> &thetalist, double eomg,
               double ev) {
  int i = 0;
  int maxiterations = 20;
  Eigen::Isometry3d Tfk = FKinSpace(M, Slist, thetalist);
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
    Tfk = FKinSpace(M, Slist, thetalist);
    Tdiff = TransInv(Tfk) * T;
    Vs = Adjoint(Tfk) * se3ToVec(MatrixLog6(Tdiff));
    angular = Eigen::Vector3d{Vs.head<3>()};
    linear = Eigen::Vector3d{Vs.tail<3>()};
    err = angular.norm() > eomg || linear.norm() > ev;
  }
  return !err;
}

template <int njoints>
bool IKinBody(const Eigen::Matrix<double, 6, njoints> &Blist,
              const Eigen::Isometry3d &M, const Eigen::Isometry3d &T,
              Eigen::Vector<double, njoints> &thetalist, double eomg,
              double ev) {
  int i = 0;
  int maxiterations = 20;
  Eigen::Isometry3d Tfk = FKinBody(M, Blist, thetalist);
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
    Tfk = FKinBody(M, Blist, thetalist);
    Tdiff = TransInv(Tfk) * T;
    Vb = se3ToVec(MatrixLog6(Tdiff));
    angular = Eigen::Vector3d{Vb.head<3>()};
    linear = Eigen::Vector3d{Vb.tail<3>()};
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
                const std::array<Eigen::Isometry3d, njoints> &Mlist,
                const std::array<Eigen::Matrix<double, 6, 6>, njoints> &Glist,
                const Eigen::Matrix<double, 6, njoints> &Slist) {
  constexpr int n1 = njoints == -1 ? -1 : njoints + 1;
  Eigen::Isometry3d Mi{Eigen::Isometry3d::Identity()};
  Eigen::Matrix<double, 6, njoints> Ai{
      Eigen::Matrix<double, 6, njoints>::Zero()};
  Eigen::Matrix<double, 6, n1> Vi{
      Eigen::Matrix<double, 6, n1>::Zero()}; // velocity
  Eigen::Matrix<double, 6, n1> Vdi{
      Eigen::Matrix<double, 6, n1>::Zero()}; // acceleration
  Vdi.bottomLeftCorner() = -g;
  std::array<Eigen::Matrix<double, 6, 6>, n1> AdTi;
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
           const std::array<Eigen::Isometry3d, njoints> &Mlist,
           const std::array<Eigen::Matrix<double, 6, 6>, njoints> &Glist,
           const Eigen::Matrix<double, 6, njoints> &Slist) {
  auto dummylist{Eigen::Vector<double, njoints>::Zero()};
  auto dummyg{Eigen::Vector3d::Zero()};
  auto dummyforce{Eigen::Vector<double, 6>::Zero()};
  auto M{Eigen::Matrix<double, njoints, njoints>::Zero()};
  Eigen::Vector<double, njoints> ddthetalist;
  for (int i = 0; i < njoints; i++) {
    ddthetalist.setZero();
    ddthetalist(i) = 1;
    M.col(i) = InverseDynamics(thetalist, dummylist, ddthetalist, dummyg,
                               dummyforce, Mlist, Glist, Slist);
  }
  return M;
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
                const std::array<Eigen::Isometry3d, njoints> &Mlist,
                const std::array<Eigen::Matrix<double, 6, 6>, njoints> &Glist,
                const Eigen::Matrix<double, 6, njoints> &Slist) {

  Eigen::Vector<double, njoints> totalForce =
      taulist - VelQuadraticForces(thetalist, dthetalist, Mlist, Glist, Slist) -
      GravityForces(thetalist, g, Mlist, Glist, Slist) -
      EndEffectorForces(thetalist, Ftip, Mlist, Glist, Slist);

  Eigen::Matrix<double, njoints, njoints> M =
      MassMatrix(thetalist, Mlist, Glist, Slist);

  // Use LDLT since M is positive definite
  Eigen::Vector<double, njoints> ddthetalist = M.ldlt().solve(totalForce);

  return ddthetalist;
}

template <int n, int N>
Eigen::Matrix<double, N, n> InverseDynamicsTrajectory(
    const Eigen::Matrix<double, N, n> &thetamat,
    const Eigen::Matrix<double, N, n> &dthetamat,
    const Eigen::Matrix<double, N, n> &ddthetamat, const Eigen::Vector3d &g,
    const Eigen::Matrix<double, N, 6> &Ftipmat,
    const std::array<Eigen::Isometry3d, n> &Mlist,
    const std::array<Eigen::Matrix<double, 6, 6>, n> &Glist,
    const Eigen::Matrix<double, 6, n> &Slist) {
  Eigen::Matrix<double, n, N> thetamatT = thetamat.transpose();
  Eigen::Matrix<double, n, N> dthetamatT = dthetamat.transpose();
  Eigen::Matrix<double, n, N> ddthetamatT = ddthetamat.transpose();
  Eigen::Matrix<double, 6, N> FtipmatT = Ftipmat.transpose();

  Eigen::Matrix<double, n, N> taumatT{Eigen::Matrix<double, n, N>::Zero()};
  for (int i = 0; i < N; ++i) {
    taumatT.col(i) =
        InverseDynamics(thetamatT.col(i), dthetamatT.col(i), ddthetamatT.col(i),
                        g, FtipmatT.col(i), Mlist, Glist, Slist);
  }
  return taumatT.transpose();
}

template <int n, int N>
std::pair<Eigen::Matrix<double, N, n>, Eigen::Matrix<double, N, n>>
ForwardDynamicsTrajectory(
    const Eigen::Vector<double, n> &thetalist,
    const Eigen::Vector<double, n> &dthetalist,
    const Eigen::Matrix<double, N, n> &taumat, const Eigen::Vector3d &g,
    const Eigen::Matrix<double, N, 6> &Ftipmat,
    const std::array<Eigen::Isometry3d, n> &Mlist,
    const std::array<Eigen::Matrix<double, 6, 6>, n> &Glist,
    const Eigen::Matrix<double, 6, n> &Slist, const double &dt,
    const int &intRes) {
  Eigen::Matrix<double, n, N> taumatT = taumat.transpose();
  Eigen::Matrix<double, 6, N> FtipmatT = Ftipmat.transpose();
  Eigen::Matrix<double, n, N> thetamatT = Eigen::Matrix<double, n, N>::Zero();
  Eigen::Matrix<double, n, N> dthetamatT = Eigen::Matrix<double, n, N>::Zero();
  thetamatT.col(0) = thetalist;
  dthetamatT.col(0) = dthetalist;
  Eigen::Vector<double, n> thetacurrent = thetalist;
  Eigen::Vector<double, n> dthetacurrent = dthetalist;
  Eigen::Vector<double, n> ddthetalist;
  for (int i = 0; i < N - 1; ++i) {
    for (int j = 0; j < intRes; ++j) {
      ddthetalist = ForwardDynamics(thetacurrent, dthetacurrent, taumatT.col(i),
                                    g, FtipmatT.col(i), Mlist, Glist, Slist);
      EulerStep(thetacurrent, dthetacurrent, ddthetalist, dt / intRes);
    }
    thetamatT.col(i + 1) = thetacurrent;
    dthetamatT.col(i + 1) = dthetacurrent;
  }
  return {thetamatT.transpose(), dthetamatT.transpose()};
}

template <int njoints>
Eigen::Vector<double, njoints>
ComputedTorque(const Eigen::Vector<double, njoints> &thetalist,
               const Eigen::Vector<double, njoints> &dthetalist,
               const Eigen::Vector<double, njoints> &eint,
               const Eigen::Vector3d &g,
               const std::array<Eigen::Isometry3d, njoints> &Mlist,
               const std::array<Eigen::Matrix<double, 6, 6>, njoints> &Glist,
               const Eigen::Matrix<double, 6, njoints> &Slist,
               const Eigen::Vector<double, njoints> &thetalistd,
               const Eigen::Vector<double, njoints> &dthetalistd,
               const Eigen::Vector<double, njoints> &ddthetalistd,
               const double &Kp, const double &Ki, const double &Kd) {

  Eigen::Vector<double, njoints> e = thetalistd - thetalist; // position err
  Eigen::Vector<double, njoints> tau_feedforward =
      MassMatrix(thetalist, Mlist, Glist, Slist) *
      (Kp * e + Ki * (eint + e) + Kd * (dthetalistd - dthetalist));

  Eigen::Vector<double, njoints> tau_inversedyn =
      InverseDynamics(thetalist, dthetalist, ddthetalistd, g,
                      Eigen::Vector<double, 6>::Zero(), Mlist, Glist, Slist);

  return tau_feedforward + tau_inversedyn;
}

// double CubicTimeScaling(double Tf, double t) {
//   double timeratio = 1.0 * t / Tf;
//   double st = 3 * pow(timeratio, 2) - 2 * pow(timeratio, 3);
//   return st;
// }

// double QuinticTimeScaling(double Tf, double t) {
//   double timeratio = 1.0 * t / Tf;
//   double st =
//       10 * pow(timeratio, 3) - 15 * pow(timeratio, 4) + 6 * pow(timeratio,
//       5);
//   return st;
// }

// Eigen::MatrixXd JointTrajectory(const Eigen::VectorXd &thetastart,
//                                 const Eigen::VectorXd &thetaend, double Tf,
//                                 int N, int method) {
//   double timegap = Tf / (N - 1);
//   Eigen::MatrixXd trajT = Eigen::MatrixXd::Zero(thetastart.size(), N);
//   double st;
//   for (int i = 0; i < N; ++i) {
//     if (method == 3)
//       st = CubicTimeScaling(Tf, timegap * i);
//     else
//       st = QuinticTimeScaling(Tf, timegap * i);
//     trajT.col(i) = st * thetaend + (1 - st) * thetastart;
//   }
//   Eigen::MatrixXd traj = trajT.transpose();
//   return traj;
// }
// std::vector<Eigen::MatrixXd> ScrewTrajectory(const Eigen::MatrixXd &Xstart,
//                                              const Eigen::MatrixXd &Xend,
//                                              double Tf, int N, int method) {
//   double timegap = Tf / (N - 1);
//   std::vector<Eigen::MatrixXd> traj(N);
//   double st;
//   for (int i = 0; i < N; ++i) {
//     if (method == 3)
//       st = CubicTimeScaling(Tf, timegap * i);
//     else
//       st = QuinticTimeScaling(Tf, timegap * i);
//     Eigen::MatrixXd Ttemp = MatrixLog6(TransInv(Xstart) * Xend);
//     traj.at(i) = Xstart * MatrixExp6(Ttemp * st);
//   }
//   return traj;
// }

// std::vector<Eigen::MatrixXd> CartesianTrajectory(const Eigen::MatrixXd
// &Xstart,
//                                                  const Eigen::MatrixXd &Xend,
//                                                  double Tf, int N, int
//                                                  method) {
//   double timegap = Tf / (N - 1);
//   std::vector<Eigen::MatrixXd> traj(N);
//   auto Rpstart = TransToRp(Xstart);
//   auto Rpend = TransToRp(Xend);
//   Eigen::Matrix3d Rstart = Rpstart.first;
//   Eigen::Vector3d pstart = Rpstart.second;
//   Eigen::Matrix3d Rend = Rpend.first;
//   Eigen::Vector3d pend = Rpend.second;
//   double st;
//   for (int i = 0; i < N; ++i) {
//     if (method == 3)
//       st = CubicTimeScaling(Tf, timegap * i);
//     else
//       st = QuinticTimeScaling(Tf, timegap * i);
//     Eigen::Matrix3d Ri =
//         Rstart * MatrixExp3(MatrixLog3(Rstart.transpose() * Rend) * st);
//     Eigen::Vector3d pi = st * pend + (1 - st) * pstart;
//     Eigen::MatrixXd traji(4, 4);
//     traji << Ri, pi, 0, 0, 0, 1;
//     traj.at(i) = traji;
//   }
//   return traj;
// }
// std::vector<Eigen::MatrixXd> SimulateControl(
//     const Eigen::VectorXd &thetalist, const Eigen::VectorXd &dthetalist,
//     const Eigen::Vector3d &g, const Eigen::MatrixXd &Ftipmat,
//     const std::vector<Eigen::Matrix4d> &Mlist,
//     const std::vector<Eigen::Matrix<double, 6, 6>> &Glist,
//     const Eigen::Matrix<double, 6, Eigen::Dynamic> &Slist,
//     const Eigen::MatrixXd &thetamatd, const Eigen::MatrixXd &dthetamatd,
//     const Eigen::MatrixXd &ddthetamatd, const Eigen::Vector3d &gtilde,
//     const std::vector<Eigen::Matrix4d> &Mtildelist,
//     const std::vector<Eigen::Matrix<double, 6, 6>> &Gtildelist, double Kp,
//     double Ki, double Kd, double dt, int intRes) {
//   Eigen::MatrixXd FtipmatT = Ftipmat.transpose();
//   Eigen::MatrixXd thetamatdT = thetamatd.transpose();
//   Eigen::MatrixXd dthetamatdT = dthetamatd.transpose();
//   Eigen::MatrixXd ddthetamatdT = ddthetamatd.transpose();
//   int m = thetamatdT.rows();
//   int n = thetamatdT.cols();
//   Eigen::VectorXd thetacurrent = thetalist;
//   Eigen::VectorXd dthetacurrent = dthetalist;
//   Eigen::VectorXd eint = Eigen::VectorXd::Zero(m);
//   Eigen::MatrixXd taumatT = Eigen::MatrixXd::Zero(m, n);
//   Eigen::MatrixXd thetamatT = Eigen::MatrixXd::Zero(m, n);
//   Eigen::VectorXd taulist;
//   Eigen::VectorXd ddthetalist;
//   for (int i = 0; i < n; ++i) {
//     taulist =
//         ComputedTorque(thetacurrent, dthetacurrent, eint, gtilde, Mtildelist,
//                        Gtildelist, Slist, thetamatdT.col(i),
//                        dthetamatdT.col(i), ddthetamatdT.col(i), Kp, Ki, Kd);
//     for (int j = 0; j < intRes; ++j) {
//       ddthetalist = ForwardDynamics(thetacurrent, dthetacurrent, taulist, g,
//                                     FtipmatT.col(i), Mlist, Glist, Slist);
//       EulerStep(thetacurrent, dthetacurrent, ddthetalist, dt / intRes);
//     }
//     taumatT.col(i) = taulist;
//     thetamatT.col(i) = thetacurrent;
//     eint += dt * (thetamatdT.col(i) - thetacurrent);
//   }
//   std::vector<Eigen::MatrixXd> ControlTauTraj_ret;
//   ControlTauTraj_ret.push_back(taumatT.transpose());
//   ControlTauTraj_ret.push_back(thetamatT.transpose());
//   return ControlTauTraj_ret;
// }
} // namespace mr
