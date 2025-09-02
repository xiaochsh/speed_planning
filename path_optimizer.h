#pragma once
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <OsqpEigen/OsqpEigen.h>

namespace pjso {
class PathOptimizer {
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
    std::vector<double> x;
    std::vector<double> dx;
    std::vector<double> ddx;
    std::vector<double> dddx;
    std::vector<double> s;
    bool feasible = true;
    std::string message;
  };

  PathOptimizer(int N, double ds) : N_(N), ds_(ds) {}

  void setWeights(double wx, double wxe, double wdx, double wdxe, double wddx, double wddxe,
                  double wdddx, double wobs) {
    wx_ = wx;
    wxe_ = wxe;
    wdx_ = wdx;
    wdxe_ = wdxe;
    wddx_ = wddx;
    wddxe_ = wddxe;
    wdddx_ = wdddx;
    wobs_ = wobs;
  }

  void setBounds(double x_min, double x_max, double dx_min, double dx_max, double ddx_min,
                 double ddx_max, double dddx_min, double dddx_max, std::vector<double> x_max_kappa) {
    x_min_ = x_min;
    x_max_ = x_max;
    dx_min_ = dx_min;
    dx_max_ = dx_max;
    ddx_min_ = ddx_min;
    ddx_max_ = ddx_max;
    dddx_min_ = dddx_min;
    dddx_max_ = dddx_max;
    x_max_kappa_ = x_max_kappa;
  }

  Result solve(const std::vector<double>& x0, const std::vector<double>& dx0,
               const std::vector<double>& ddx0, const std::vector<double>& refx) {
    Result out;
    out.feasible = true;
    out.message = "";

    if ((int)x0.size() != N_ || (int)dx0.size() != N_ || (int)ddx0.size() != N_ ||
        (int)refx.size() != N_) {
      out.feasible = false;
      out.message = "Input vectors size mismatch N";
      return out;
    }

    int variables = 3 * N_;
    int constraints = 3 + 3 * N_ + 3 * (N_ - 1);

    Eigen::SparseMatrix<double> P(variables, variables);
    Eigen::VectorXd q(variables);
    Eigen::SparseMatrix<double> A(constraints, variables);
    Eigen::VectorXd l(constraints), u(constraints);

    // --- 构建成本矩阵 P ---
    P.setZero();
    for (int i = 0; i < N_ - 1; i++) P.coeffRef(i, i) += wx_ + wobs_;
    P.coeffRef(N_ - 1, N_ - 1) += wxe_ + wobs_;

    for (int i = 0; i < N_ - 1; i++) P.coeffRef(N_ + i, N_ + i) += wdx_;
    P.coeffRef(2 * N_ - 1, 2 * N_ - 1) += wdxe_;

    for (int i = 0; i < N_ - 1; i++) {
      P.coeffRef(2 * N_ + i, 2 * N_ + i) += wddx_;
      P.coeffRef(2 * N_ + i, 2 * N_ + i) += wdddx_ / (ds_ * ds_);
      P.coeffRef(2 * N_ + i + 1, 2 * N_ + i + 1) += wdddx_ / (ds_ * ds_);
      P.coeffRef(2 * N_ + i, 2 * N_ + i + 1) -= wdddx_ / (ds_ * ds_);
      P.coeffRef(2 * N_ + i + 1, 2 * N_ + i) -= wdddx_ / (ds_ * ds_);
    }
    P.coeffRef(3 * N_ - 1, 3 * N_ - 1) += wddxe_;

    // --- 构建线性成本向量 q ---
    q.setZero();
    for (int i = 0; i < N_; i++) q(i) = -wobs_ * refx[i];

    // --- 构建约束矩阵 A 和上下界 l/u ---
    A.setZero();
    l.setZero();
    u.setZero();

    int constraint_idx = 0;

    // 初始状态约束
    A.insert(constraint_idx, 0) = 1.0;
    l(constraint_idx) = x0[0];
    u(constraint_idx) = x0[0];
    constraint_idx++;
    A.insert(constraint_idx, N_) = 1.0;
    l(constraint_idx) = dx0[0];
    u(constraint_idx) = dx0[0];
    constraint_idx++;
    A.insert(constraint_idx, 2 * N_) = 1.0;
    l(constraint_idx) = ddx0[0];
    u(constraint_idx) = ddx0[0];
    constraint_idx++;

    // 状态边界约束
    for (int i = 0; i < N_; i++) {
      A.insert(constraint_idx, i) = 1.0;
      l(constraint_idx) = x_min_;
      u(constraint_idx) = std::min(x_max_, x_max_kappa_[i]);
      constraint_idx++;
      A.insert(constraint_idx, N_ + i) = 1.0;
      l(constraint_idx) = dx_min_;
      u(constraint_idx) = dx_max_;
      constraint_idx++;
      A.insert(constraint_idx, 2 * N_ + i) = 1.0;
      l(constraint_idx) = ddx_min_;
      u(constraint_idx) = ddx_max_;
      constraint_idx++;
    }

    // 加速度动力学约束
    for (int i = 0; i < N_ - 1; i++) {
      A.insert(constraint_idx, 2 * N_ + i) = -1.0;
      A.insert(constraint_idx, 2 * N_ + i + 1) = 1.0;
      l(constraint_idx) = dddx_min_ * ds_;
      u(constraint_idx) = dddx_max_ * ds_;
      constraint_idx++;
    }

    // 速度动力学约束
    for (int i = 0; i < N_ - 1; i++) {
      A.insert(constraint_idx, N_ + i) = -1.0;
      A.insert(constraint_idx, N_ + i + 1) = 1.0;
      A.insert(constraint_idx, 2 * N_ + i) = -0.5 * ds_;
      A.insert(constraint_idx, 2 * N_ + i + 1) = -0.5 * ds_;
      l(constraint_idx) = 0.0;
      u(constraint_idx) = 0.0;
      constraint_idx++;
    }

    // 位置动力学约束
    for (int i = 0; i < N_ - 1; i++) {
      A.insert(constraint_idx, i) = -1.0;
      A.insert(constraint_idx, i + 1) = 1.0;
      A.insert(constraint_idx, N_ + i) = -ds_;
      A.insert(constraint_idx, 2 * N_ + i) = -ds_ * ds_ / 3.0;
      A.insert(constraint_idx, 2 * N_ + i + 1) = -ds_ * ds_ / 6.0;
      l(constraint_idx) = 0.0;
      u(constraint_idx) = 0.0;
      constraint_idx++;
    }

    // --- 调用OSQP求解 ---
    OsqpEigen::Solver solver;
    solver.settings()->setVerbosity(verbose_);
    solver.settings()->setWarmStart(warm_start_);
    solver.settings()->setAbsoluteTolerance(eps_abs_);
    solver.settings()->setRelativeTolerance(eps_rel_);
    solver.settings()->setPrimalInfeasibilityTolerance(eps_prim_inf_);
    solver.settings()->setDualInfeasibilityTolerance(eps_dual_inf_);
    solver.settings()->setAlpha(alpha_);
    solver.settings()->setRho(rho_);
    solver.settings()->setSigma(sigma_);
    solver.settings()->setMaxIteration(max_iter_);
    solver.settings()->setPolish(polish_);

    solver.data()->setNumberOfVariables(variables);
    solver.data()->setNumberOfConstraints(constraints);

    if (!solver.data()->setHessianMatrix(P)) {
      out.feasible = false;
      out.message = "setHessianMatrix failed";
      return out;
    }
    if (!solver.data()->setGradient(q)) {
      out.feasible = false;
      out.message = "setGradient failed";
      return out;
    }
    if (!solver.data()->setLinearConstraintsMatrix(A)) {
      out.feasible = false;
      out.message = "setLinearConstraintsMatrix failed";
      return out;
    }
    if (!solver.data()->setLowerBound(l)) {
      out.feasible = false;
      out.message = "setLowerBound failed";
      return out;
    }
    if (!solver.data()->setUpperBound(u)) {
      out.feasible = false;
      out.message = "setUpperBound failed";
      return out;
    }

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

    Eigen::VectorXd sol = solver.getSolution();

    out.s.resize(N_);
    for (int i = 0; i < N_; i++) out.s[i] = i * ds_;
    out.x.assign(sol.segment(0, N_).data(), sol.segment(0, N_).data() + N_);
    out.dx.assign(sol.segment(N_, N_).data(), sol.segment(N_, N_).data() + N_);
    out.ddx.assign(sol.segment(2 * N_, N_).data(), sol.segment(2 * N_, N_).data() + N_);
    out.dddx.resize(N_);
    for (int i = 0; i < N_ - 1; i++) out.dddx[i] = (out.ddx[i + 1] - out.ddx[i]) / ds_;
    out.dddx[N_ - 1] = out.dddx[N_ - 2];

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

 private:
  int N_;
  double ds_ = 0.1;

  // 权重
  double wx_ = 1.0, wxe_ = 100000.0;
  double wdx_ = 1.0, wdxe_ = 100000.0;
  double wddx_ = 0.1, wddxe_ = 100000.0, wdddx_ = 1e-4;
  double wobs_ = 100000.0;

  // 约束边界
  double x_min_ = -10.0, x_max_ = 10.0;
  double dx_min_ = -50.0, dx_max_ = 50.0;
  double ddx_min_ = -100.0, ddx_max_ = 100.0;
  double dddx_min_ = -1000.0, dddx_max_ = 1000.0;
  std::vector<double> x_max_kappa_;

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
}  // namespace pjso