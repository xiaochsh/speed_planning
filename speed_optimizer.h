#pragma once
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <OsqpEigen/OsqpEigen.h>

namespace pjso {
class TrajectoryOptimizer {
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
    std::vector<double> s;
    std::vector<double> v;
    std::vector<double> a;
    std::vector<double> jerk;
    std::vector<double> t;
    bool feasible = true;
    std::string message;
  };

  TrajectoryOptimizer(int N, double dt) : N_(N), dt_(dt) {}

  void setWeights(double ws, double wse, double wv, double wve, double wa, double wae, double wj) {
    ws_ = ws;
    wse_ = wse;
    wv_ = wv;
    wve_ = wve;
    wa_ = wa;
    wae_ = wae;
    wj_ = wj;
  }

  void setBounds(double s_min, double s_max, double v_min, double v_max, double a_min, double a_max,
                 double j_min, double j_max) {
    s_min_ = s_min;
    s_max_ = s_max;
    v_min_ = v_min;
    v_max_ = v_max;
    a_min_ = a_min;
    a_max_ = a_max;
    j_min_ = j_min;
    j_max_ = j_max;
  }

  Result solve(const std::vector<double>& x0, const std::vector<double>& v0,
               const std::vector<double>& a0) {
    Result out;
    out.feasible = true;
    out.message = "";

    if ((int)x0.size() != N_ || (int)v0.size() != N_ || (int)a0.size() != N_) {
      out.feasible = false;
      out.message = "Input vectors size mismatch N";
      return out;
    }

    int variables = 3 * N_;
    int constraints = 3 + 3 * N_ + (N_ - 1) + 2 * (N_ - 1);

    Eigen::SparseMatrix<double> P(variables, variables);
    Eigen::VectorXd q(variables);
    Eigen::SparseMatrix<double> A(constraints, variables);
    Eigen::VectorXd l(constraints), u(constraints);

    // --- 构建成本矩阵 P ---
    P.setZero();
    for (int i = 0; i < N_ - 1; i++) P.coeffRef(i, i) += ws_;
    P.coeffRef(N_ - 1, N_ - 1) += wse_;

    for (int i = 0; i < N_ - 1; i++) P.coeffRef(N_ + i, N_ + i) += wv_;
    P.coeffRef(2 * N_ - 1, 2 * N_ - 1) += wve_;

    for (int i = 0; i < N_; i++) {
      P.coeffRef(2 * N_ + i, 2 * N_ + i) += wa_;
      if (i < N_ - 1) {
        P.coeffRef(2 * N_ + i, 2 * N_ + i) += wj_ / (dt_ * dt_);
        P.coeffRef(2 * N_ + i + 1, 2 * N_ + i + 1) += wj_ / (dt_ * dt_);
        P.coeffRef(2 * N_ + i, 2 * N_ + i + 1) -= wj_ / (dt_ * dt_);
        P.coeffRef(2 * N_ + i + 1, 2 * N_ + i) -= wj_ / (dt_ * dt_);
      }
    }
    P.coeffRef(3 * N_ - 1, 3 * N_ - 1) += wae_;

    // --- 构建线性成本向量 q ---
    q.setZero();
    for (int i = 0; i < N_; i++) {
      q(i) = -ws_ * x0[i];
      q(N_ + i) = -wv_ * v0[i];
      q(2 * N_ + i) = -wa_ * a0[i];
    }
    q(N_ - 1) = -wse_ * x0[N_ - 1];
    q(2 * N_ - 1) = -wve_ * v0[N_ - 1];
    q(3 * N_ - 1) = -wae_ * a0[N_ - 1];

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
    l(constraint_idx) = v0[0];
    u(constraint_idx) = v0[0];
    constraint_idx++;
    A.insert(constraint_idx, 2 * N_) = 1.0;
    l(constraint_idx) = a0[0];
    u(constraint_idx) = a0[0];
    constraint_idx++;

    // 状态边界约束
    for (int i = 0; i < N_; i++) {
      A.insert(constraint_idx, i) = 1.0;
      l(constraint_idx) = s_min_;
      u(constraint_idx) = s_max_;
      constraint_idx++;
      A.insert(constraint_idx, N_ + i) = 1.0;
      l(constraint_idx) = v_min_;
      u(constraint_idx) = v_max_;
      constraint_idx++;
      A.insert(constraint_idx, 2 * N_ + i) = 1.0;
      l(constraint_idx) = a_min_;
      u(constraint_idx) = a_max_;
      constraint_idx++;
    }

    // Jerk约束
    for (int i = 0; i < N_ - 1; i++) {
      A.insert(constraint_idx, 2 * N_ + i) = -1.0 / dt_;
      A.insert(constraint_idx, 2 * N_ + i + 1) = 1.0 / dt_;
      l(constraint_idx) = j_min_;
      u(constraint_idx) = j_max_;
      constraint_idx++;
    }

    // 速度动力学约束
    for (int i = 0; i < N_ - 1; i++) {
      A.insert(constraint_idx, N_ + i) = -1.0;
      A.insert(constraint_idx, N_ + i + 1) = 1.0;
      A.insert(constraint_idx, 2 * N_ + i) = -0.5 * dt_;
      A.insert(constraint_idx, 2 * N_ + i + 1) = -0.5 * dt_;
      l(constraint_idx) = 0.0;
      u(constraint_idx) = 0.0;
      constraint_idx++;
    }

    // 位置动力学约束
    for (int i = 0; i < N_ - 1; i++) {
      A.insert(constraint_idx, i) = -1.0;
      A.insert(constraint_idx, i + 1) = 1.0;
      A.insert(constraint_idx, N_ + i) = -dt_;
      A.insert(constraint_idx, 2 * N_ + i) = -dt_ * dt_ / 3.0;
      A.insert(constraint_idx, 2 * N_ + i + 1) = -dt_ * dt_ / 6.0;
      l(constraint_idx) = 0.0;
      u(constraint_idx) = 0.0;
      constraint_idx++;
    }

    // --- 调用OSQP求解 ---
    OsqpEigen::Solver solver;
    solver.settings()->setVerbosity(verbose_);
    solver.settings()->setWarmStart(true);

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

    out.t.resize(N_);
    for (int i = 0; i < N_; i++) out.t[i] = i * dt_;
    out.s.assign(sol.segment(0, N_).data(), sol.segment(0, N_).data() + N_);
    out.v.assign(sol.segment(N_, N_).data(), sol.segment(N_, N_).data() + N_);
    out.a.assign(sol.segment(2 * N_, N_).data(), sol.segment(2 * N_, N_).data() + N_);
    out.jerk.resize(N_);
    for (int i = 0; i < N_ - 1; i++) out.jerk[i] = (out.a[i + 1] - out.a[i]) / dt_;
    out.jerk[N_ - 1] = out.jerk[N_ - 2];

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
  double dt_ = 0.1;

  // 权重
  double ws_ = 1.0, wse_ = 100000.0;
  double wv_ = 1.0, wve_ = 100000.0;
  double wa_ = 0.1, wae_ = 100000.0, wj_ = 1e-4;

  // 约束边界
  double s_min_ = 0.0, s_max_ = 10000.0;
  double v_min_ = -50.0, v_max_ = 50.0;
  double a_min_ = -5.0, a_max_ = 2.5;
  double j_min_ = -5.0, j_max_ = 5.0;

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