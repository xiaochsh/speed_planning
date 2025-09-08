#pragma once
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <OsqpEigen/OsqpEigen.h>

namespace mpc {
class KineBicycleMpc {
 public:
  struct Options {
    double eps_abs = 1e-3;       // 绝对精度要求，控制解的精度
    double eps_rel = 1e-3;       // 相对精度要求, 控制解的精度
    double eps_prim_inf = 1e-3;  // 原始可行性判断阈值
    double eps_dual_inf = 1e-3;  // 对偶可行性判断阈值
    double alpha = 1.6;          // ADMM松弛因子，通常在[1.0, 1.8]之间
    double rho = 0.1;            // ADMM惩罚参数，影响收敛速度和数值稳定性
    double sigma = 1e-6;         // H矩阵的正则化参数，防止H矩阵奇异
    int max_iter = 10000;        // 最大迭代次数
    bool verbose = false;        // 是否输出求解过程信息
    bool warm_start = true;      // 是否使用初始解进行热启动
    bool polish = false;         // 是否进行抛光步骤以提高解的精度
  };

  struct Result {
    std::vector<Eigen::VectorXd> x_pred;
    std::vector<Eigen::VectorXd> u_pred;
    bool feasible = true;
    std::string message;
  };

  KineBicycleMpc(int N_, double Ts_) : N(N_), Ts(Ts_) { u_prev.setZero(nu); }

  void setWeights(double wx, double wy, double wpsi, double wv, double wxe, double wye,
                  double wpsie, double wve, double wa, double wdelta, double dwa, double dwdelta) {
    Q.setZero(4, 4);
    Qf.setZero(4, 4);
    R.setZero(2, 2);
    Rdelta.setZero(2, 2);

    Q.diagonal() << wx, wy, wpsi, wv;
    Qf.diagonal() << wxe, wye, wpsie, wve;
    R.diagonal() << wa, wdelta;
    Rdelta.diagonal() << dwa, dwdelta;
  }

  void setBounds(double a_min, double a_max, double delta_min, double delta_max) {
    a_min_ = a_min;
    a_max_ = a_max;
    delta_min_ = delta_min;
    delta_max_ = delta_max;
  }

  Result solve(const std::vector<Eigen::VectorXd>& x_ref, const std::vector<Eigen::VectorXd>& u_ref,
               const Eigen::Vector4d& x0) {
    Result out;
    out.feasible = true;
    out.message = "";

    if ((int)x_ref.size() != N + 1 || (int)u_ref.size() != N || (int)x0.rows() != nx) {
      out.feasible = false;
      out.message = "Input vectors size mismatch N";
      return out;
    }

    int nX = nx * N;
    int nU = nu * N;

    // ========================
    // 1) 线性化并离散化模型
    // ========================
    std::vector<Eigen::MatrixXd> Ad(N), Bd(N);
    std::vector<Eigen::VectorXd> cd(N);
    for (int k = 0; k < N; ++k) {
      Eigen::MatrixXd Ac(nx, nx), Bc(nx, nu);
      Eigen::VectorXd cc(nx);
      compute_Ac_Bc_cc(x_ref[k], u_ref[k], Ac, Bc, cc);
      Ad[k] = Eigen::MatrixXd::Identity(nx, nx) + Ac * Ts;
      Bd[k] = Bc * Ts;
      cd[k] = cc * Ts;
    }

    // ========================
    // 2) 构造大矩阵 bigA, bigB, bigC
    // ========================
    Eigen::MatrixXd bigA = Eigen::MatrixXd::Zero(nX, nx);
    Eigen::MatrixXd bigB = Eigen::MatrixXd::Zero(nX, nU);
    Eigen::VectorXd bigC = Eigen::VectorXd::Zero(nX);

    Eigen::MatrixXd Aprod = Eigen::MatrixXd::Identity(nx, nx);
    for (int i = 0; i < N; ++i) {
      Aprod = Ad[i] * Aprod;
      bigA.block(i * nx, 0, nx, nx) = Aprod;

      Eigen::MatrixXd prod = Eigen::MatrixXd::Identity(nx, nx);
      for (int j = i; j >= 0; --j) {
        if (j < i) prod = Ad[j + 1] * prod;
        bigB.block(i * nx, j * nu, nx, nu) = prod * Bd[j];
      }

      Eigen::VectorXd csum = Eigen::VectorXd::Zero(nx);
      Eigen::MatrixXd prod2 = Eigen::MatrixXd::Identity(nx, nx);
      for (int j = i; j >= 0; --j) {
        if (j < i) prod2 = Ad[j + 1] * prod2;
        csum += prod2 * cd[j];
      }
      bigC.segment(i * nx, nx) = csum;
    }

    // ========================
    // 3) ΔU -> U 转换矩阵
    // ========================
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(nU, nU);
    Eigen::VectorXd U0 = Eigen::VectorXd::Zero(nU);
    for (int row = 0; row < N; ++row) {
      for (int col = 0; col <= row; ++col) {
        M.block(row * nu, col * nu, nu, nu) = Eigen::MatrixXd::Identity(nu, nu);
      }
      U0.segment(row * nu, nu) = u_prev;
    }

    // ========================
    // 4) 权重矩阵
    // ========================
    Eigen::MatrixXd Qblk = Eigen::MatrixXd::Zero(nX, nX);
    for (int i = 0; i < N - 1; ++i) Qblk.block(i * nx, i * nx, nx, nx) = Q;
    Qblk.block((N - 1) * nx, (N - 1) * nx, nx, nx) = Qf;

    Eigen::MatrixXd Rblk = Eigen::MatrixXd::Zero(nU, nU);
    for (int i = 0; i < N; ++i) Rblk.block(i * nu, i * nu, nu, nu) = R;

    Eigen::MatrixXd Rdblk = Eigen::MatrixXd::Zero(nU, nU);
    for (int i = 0; i < N; ++i) Rdblk.block(i * nu, i * nu, nu, nu) = Rdelta;

    // ========================
    // 5) e 向量
    // ========================
    Eigen::VectorXd e = bigA * x0 + bigB * U0 + bigC;

    // ========================
    // 6) 构造代价函数 H, f
    // ========================
    Eigen::VectorXd Xref = Eigen::VectorXd::Zero(nX);
    Eigen::VectorXd Uref = Eigen::VectorXd::Zero(nU);
    for (int i = 0; i < N; ++i) {
      Xref.segment(i * nx, nx) = x_ref[i];
      if (i < N) Uref.segment(i * nu, nu) = u_ref[i];
    }

    Eigen::MatrixXd H = 2.0 * (M.transpose() * bigB.transpose() * Qblk * bigB * M +
                               M.transpose() * Rblk * M + Rdblk);
    Eigen::VectorXd f = 2.0 * (M.transpose() * bigB.transpose() * Qblk * (e - Xref) +
                               M.transpose() * Rblk * (U0 - Uref));

    // 确保对称：取 (H + H.transpose()) / 2
    Eigen::MatrixXd H_sym = 0.5 * (H + H.transpose());
    Eigen::SparseMatrix<double> H_sp = H_sym.sparseView();

    // ========================
    // 7) 构造约束 (这里只放 ΔU 约束)
    // ========================
    Eigen::VectorXd U_min(nu * N), U_max(nu * N);
    for (int i = 0; i < N; ++i) {
      U_min(i * nu + 0) = a_min_;
      U_max(i * nu + 0) = a_max_;
      U_min(i * nu + 1) = delta_min_;
      U_max(i * nu + 1) = delta_max_;
    }

    Eigen::SparseMatrix<double> A_cons(nu * N, nu * N);
    A_cons = M.sparseView();
    Eigen::VectorXd l_cons = U_min - U0;
    Eigen::VectorXd u_cons = U_max - U0;

    // ========================
    // 8) 调用 OSQP
    // ========================
    OsqpEigen::Solver solver;
    solver.settings()->setVerbosity(verbose_);
    solver.settings()->setWarmStart(warm_start_);
    solver.settings()->setAbsoluteTolerance(eps_abs_);
    solver.settings()->setRelativeTolerance(eps_rel_);
    solver.settings()->setMaxIteration(max_iter_);
    solver.data()->setNumberOfVariables(nU);
    solver.data()->setNumberOfConstraints(nU);

    solver.data()->setHessianMatrix(H_sp);
    solver.data()->setGradient(f);
    solver.data()->setLinearConstraintsMatrix(A_cons);
    solver.data()->setLowerBound(l_cons);
    solver.data()->setUpperBound(u_cons);

    if (!solver.initSolver()) {
      out.feasible = false;
      out.message = "initSolver failed";
      return out;
    }
    if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) {
      out.feasible = false;
      out.message = "solveProblem failed";
      return out;
    }

    Eigen::VectorXd dU_opt = solver.getSolution();
    Eigen::VectorXd U_opt = M * dU_opt + U0;
    u_prev = U_opt.segment((N - 1) * nu, nu);  // 保存最后一步作为下次 U0

    // ========================
    // 9) 恢复预测轨迹
    // ========================
    Eigen::VectorXd X_pred = bigA * x0 + bigB * U_opt + bigC;

    out.x_pred.resize(N);
    out.u_pred.resize(N);
    for (int i = 0; i < N; ++i) {
      out.x_pred[i] = X_pred.segment(i * nu, nu);
      out.u_pred[i] = U_opt.segment(i * nu, nu);
    }

    return out;
  }

  void setOptions(const Options& opt) {
    eps_abs_ = opt.eps_abs;
    eps_rel_ = opt.eps_rel;
    eps_prim_inf_ = opt.eps_prim_inf;
    eps_dual_inf_ = opt.eps_dual_inf;
    alpha_ = opt.alpha;
    rho_ = opt.rho;
    sigma_ = opt.sigma;
    max_iter_ = opt.max_iter;
    verbose_ = opt.verbose;
    warm_start_ = opt.warm_start;
    polish_ = opt.polish;
  }

  // 协助函数（你需实现）
  void compute_Ac_Bc_cc(const Eigen::VectorXd& xr, const Eigen::VectorXd& ur, Eigen::MatrixXd& A_c,
                        Eigen::MatrixXd& B_c, Eigen::VectorXd& c_c) {
    // 用你的解析偏导公式实现
    // 例如 A_c as in previous derivation etc.
    A_c.setZero(4, 4);
    A_c(0, 2) = -xr(3) * std::sin(xr(2));
    A_c(0, 3) = std::cos(xr(2));
    A_c(1, 2) = xr(3) * std::cos(xr(2));
    A_c(1, 3) = std::sin(xr(2));
    A_c(2, 3) = std::tan(ur(1)) / L;

    B_c.setZero(4, 2);
    B_c(2, 1) = xr(3) / std::pow(std::cos(ur(1)), 2) / L;
    B_c(3, 1) = 1.0;

    c_c.setZero(4, 1);
    Eigen::VectorXd f0(4);
    f0 << xr(3) * std::cos(xr(2)), xr(3) * std::sin(xr(2)), xr(3) * std::tan(ur(1)) / L, ur(0);
    c_c = f0 - A_c * xr - B_c * ur;
  }

 private:
  int N;
  int nx = 4;
  int nu = 2;
  double L = 2.7;
  double Ts = 0.02;
  Eigen::VectorXd u_prev;

  // 权重
  Eigen::MatrixXd Q;
  Eigen::MatrixXd Qf;
  Eigen::MatrixXd R;
  Eigen::MatrixXd Rdelta;

  // 约束边界
  double delta_min_;
  double delta_max_;
  double a_min_;
  double a_max_;

  // 求解器参数
  double eps_abs_ = 1e-3;
  double eps_rel_ = 1e-3;
  double eps_prim_inf_ = 1e-3;
  double eps_dual_inf_ = 1e-3;
  double alpha_ = 1.6;
  double rho_ = 0.1;
  double sigma_ = 1e-6;
  int max_iter_ = 10000;
  bool verbose_ = false;
  bool warm_start_ = true;
  bool polish_ = false;
};
}  // namespace mpc