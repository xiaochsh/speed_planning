#pragma once
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <OsqpEigen/OsqpEigen.h>

namespace fem {
class FemSmoother {
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
    std::vector<double> y;
    bool feasible = true;
    std::string message;
  };

  FemSmoother(int N) : N_(N) {}

  void setWeights(double wr, double wl, double ws) {
    wr_ = wr;
    wl_ = wl;
    ws_ = ws;
  }

  void setBounds(double bounds) {
    bounds_ = bounds;
  }

  Result solve(const std::vector<double>& refx, const std::vector<double>& refy) {
    Result out;
    out.feasible = true;
    out.message = "";

    if ((int)refx.size() != N_ || (int)refy.size() != N_) {
      out.feasible = false;
      out.message = "Input vectors size mismatch N";
      return out;
    }

    int variables = 2 * N_;
    int constraints = 2 * N_;

    Eigen::SparseMatrix<double> P(variables, variables);
    Eigen::VectorXd q(variables);
    Eigen::SparseMatrix<double> A(constraints, variables);
    Eigen::VectorXd l(constraints), u(constraints);
    Eigen::VectorXd primal_warm_start(variables);
    
    // --- 构建成本矩阵 P ---
    P.setZero();

    // 几何形状代价
    for (int i = 0; i < N_; i++) {
      P.coeffRef(2 * i, 2 * i) += wr_;
      P.coeffRef(2 * i + 1, 2 * i + 1) += wr_;
    }

    // 长度代价
    P.coeffRef(0, 0) += wl_;
    P.coeffRef(1, 1) += wl_;
    P.coeffRef(2 * N_ - 2, 2 * N_ - 2) += wl_;
    P.coeffRef(2 * N_ - 1, 2 * N_ - 1) += wl_;
    for (int i = 1; i < N_ - 1; i++) {
      P.coeffRef(2 * i, 2 * i) += 2.0 * wl_;
      P.coeffRef(2 * i + 1, 2 * i + 1) += 2.0 * wl_;
    }
    for (int i = 0; i < N_ - 1; i++) {
      P.coeffRef(2 * i, 2 * (i + 1)) -= wl_;
      P.coeffRef(2 * i + 1, 2 * (i + 1) + 1) -= wl_;
      P.coeffRef(2 * (i + 1), 2 * i) -= wl_;
      P.coeffRef(2 * (i + 1) + 1, 2 * i + 1) -= wl_;
    }

    // 平滑代价（曲率）
    for (int i = 0; i < N_ - 2; i++) {
      P.coeffRef(2 * i, 2 * i) += ws_;
      P.coeffRef(2 * i + 1, 2 * i + 1) += ws_;
      P.coeffRef(2 * (i + 1), 2 * (i + 1)) += 4.0 * ws_;
      P.coeffRef(2 * (i + 1) + 1, 2 * (i + 1) + 1) += 4.0 * ws_;
      P.coeffRef(2 * (i + 2), 2 * (i + 2)) += ws_;
      P.coeffRef(2 * (i + 2) + 1, 2 * (i + 2) + 1) += ws_;

      P.coeffRef(2 * i, 2 * (i + 1)) -= 2.0 * ws_;
      P.coeffRef(2 * i + 1, 2 * (i + 1) + 1) -= 2.0 * ws_;
      P.coeffRef(2 * (i + 1), 2 * i) -= 2.0 * ws_;
      P.coeffRef(2 * (i + 1) + 1, 2 * i + 1) -= 2.0 * ws_;

      P.coeffRef(2 * i, 2 * (i + 2)) += ws_;
      P.coeffRef(2 * i + 1, 2 * (i + 2) + 1) += ws_;
      P.coeffRef(2 * (i + 2), 2 * i) += ws_;
      P.coeffRef(2 * (i + 2) + 1, 2 * i + 1) += ws_;

      P.coeffRef(2 * (i + 1), 2 * (i + 2)) -= 2.0 * ws_;
      P.coeffRef(2 * (i + 1) + 1, 2 * (i + 2) + 1) -= 2.0 * ws_;
      P.coeffRef(2 * (i + 2), 2 * (i + 1)) -= 2.0 * ws_;
      P.coeffRef(2 * (i + 2) + 1, 2 * (i + 1) + 1) -= 2.0 * ws_;
    }

    // --- 构建线性成本向量 q ---
    q.setZero();
    for (int i = 0; i < N_; i++) {
      q(2 * i) = -wr_ * refx[i];
      q(2 * i + 1) = -wr_ * refy[i];
    }

    // --- 构建约束矩阵 A 和上下界 l/u ---
    A.setZero();
    l.setZero();
    u.setZero();

    for (int i = 0; i < N_; i++) {
      A.coeffRef(2 * i, 2 * i) = 1.0;
      A.coeffRef(2 * i + 1, 2 * i + 1) = 1.0;
      l(2 * i) = refx[i] - bounds_;
      l(2 * i + 1) = refy[i] - bounds_;
      u(2 * i) = refx[i] + bounds_;
      u(2 * i + 1) = refy[i] + bounds_;
    }

    for (int i = 0; i < N_; i++) {
      primal_warm_start[2 * i] = refx[i];
      primal_warm_start[2 * i + 1] = refy[i];
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

    if (!solver.setPrimalVariable(primal_warm_start)) {
      out.feasible = false;
      out.message = "setPrimalVariable failed";
      return out;
    }

    if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) {
      out.feasible = false;
      out.message = "solveProblem failed";
      return out;
    }

    Eigen::VectorXd sol = solver.getSolution();

    out.x.resize(N_);
    out.y.resize(N_);
    for (int i = 0; i < N_; i++) {
      out.x[i] = sol[2 * i];
      out.y[i] = sol[2 * i + 1];
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

 private:
  int N_;

  // 权重
  double wr_ = 100.0;
  double wl_ = 10.0;
  double ws_ = 10.0;

  // 约束边界
  double bounds_ = 0.5;

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
}  // fem