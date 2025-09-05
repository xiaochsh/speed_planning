#pragma once
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <OsqpEigen/OsqpEigen.h>

namespace fem {
class NlpFemSmoother {
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
    int sqp_pen_max_iter = 10;
    int sqp_sub_max_iter = 100;
    double sqp_ctol = 1e-3;
    double sqp_ftol = 1e-4;
  };

  struct Result {
    std::vector<double> x;
    std::vector<double> y;
    bool feasible = true;
    std::string message;
  };

  NlpFemSmoother(int N, double ds) : N_(N), ds_(ds) {}

  void setWeights(double wr, double wl, double ws, double wk) {
    wr_ = wr;
    wl_ = wl;
    ws_ = ws;
    wk_ = wk;
  }

  void setBounds(double bounds, double max_cur) {
    bounds_ = bounds;
    max_cur_ = max_cur;
  }

  Result solve(const std::vector<double>& refx, const std::vector<double>& refy) {
    Result opt;
    opt.feasible = true;
    opt.message = "";

    if ((int)refx.size() != N_ || (int)refy.size() != N_) {
      opt.feasible = false;
      opt.message = "Input vectors size mismatch N";
      return opt;
    }

    int variables = 2 * N_ + N_ - 2;
    int constraints = 2 * N_ + 2* (N_ - 2);

    Eigen::SparseMatrix<double> P(variables, variables);
    Eigen::VectorXd q(variables);
    Eigen::SparseMatrix<double> A(constraints, variables);
    Eigen::VectorXd l(constraints), u(constraints);
    Eigen::VectorXd primal_warm_start(variables);

    // --- 计算热启动原始解 ---
    SetPrimalWarmStart(refx, refy, primal_warm_start);

    // --- 构建成本矩阵 P ---
    CalculateKernel(P);

    // --- 构建线性成本向量 q ---
    CalculateOffset(refx, refy, q);

    // --- 构建约束矩阵 A 和上下界 l/u ---
    CalculateAffineConstraint(refx, refy, A, l, u);

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
      opt.feasible = false;
      opt.message = "setHessianMatrix failed";
      return opt;
    }
    if (!solver.data()->setGradient(q)) {
      opt.feasible = false;
      opt.message = "setGradient failed";
      return opt;
    }
    if (!solver.data()->setLinearConstraintsMatrix(A)) {
      opt.feasible = false;
      opt.message = "setLinearConstraintsMatrix failed";
      return opt;
    }
    if (!solver.data()->setLowerBound(l)) {
      opt.feasible = false;
      opt.message = "setLowerBound failed";
      return opt;
    }
    if (!solver.data()->setUpperBound(u)) {
      opt.feasible = false;
      opt.message = "setUpperBound failed";
      return opt;
    }

    if (!solver.initSolver()) {
      opt.feasible = false;
      opt.message = "initSolver failed";
      return opt;
    }

    if (!solver.setPrimalVariable(primal_warm_start)) {
      opt.feasible = false;
      opt.message = "setPrimalVariable failed";
      return opt;
    }

    if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) {
      opt.feasible = false;
      opt.message = "solveProblem failed";
      return opt;
    }

    Eigen::VectorXd sol = solver.getSolution();

    opt.x.resize(N_);
    opt.y.resize(N_);
    slack_.resize(N_ - 2);
    for (int i = 0; i < N_; i++) {
      opt.x[i] = sol[2 * i];
      opt.y[i] = sol[2 * i + 1];
    }
    for (int i = 0; i < N_ - 2; i++) {
      slack_[i] = sol[2 * N_ + i];
    }

    // Sequential solution
    int pen_itr = 0;
    double ctol = 0.0;
    double origin_wk = wk_;
    double last_fvalue = solver.getObjValue();

    while (pen_itr < sqp_pen_max_iter_) {
      int sub_itr = 1;
      bool fconverged = false;

      while (sub_itr < sqp_sub_max_iter_) {
        SetPrimalWarmStart(opt.x, opt.y, primal_warm_start);
        CalculateOffset(opt.x, opt.y, q);
        CalculateAffineConstraint(opt.x, opt.y, A, l, u);

        solver.updateGradient(q);
        solver.updateLinearConstraintsMatrix(A);
        solver.updateBounds(l, u);

        if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) {
          opt.feasible = false;
          opt.message = "iteration at " + std::to_string(sub_itr) +
                        ", solving fails with max sub iter " + std::to_string(sqp_sub_max_iter_);
          return opt;
        }

        double cur_fvalue = solver.getObjValue();
        double ftol = std::abs((last_fvalue - cur_fvalue) / last_fvalue);

        if (ftol < sqp_ftol_) {
          std::cout << "merit function value converges at sub itr num " << sub_itr
                    << "merit function value converges to " << cur_fvalue << ", with ftol " << ftol
                    << ", under max_ftol " << sqp_ftol_ << std::endl;
          fconverged = true;
          break;
        }

        last_fvalue = cur_fvalue;
        ++sub_itr;
      }

      if (!fconverged) {
        opt.feasible = false;
        opt.message = "Max number of iteration reached";
        return opt;
      }

      sol = solver.getSolution();
      for (int i = 0; i < N_; i++) {
        opt.x[i] = sol[2 * i];
        opt.y[i] = sol[2 * i + 1];
      }
      for (int i = 0; i < N_ - 2; i++) {
        slack_[i] = sol[2 * N_ + i];
      }

      ctol = CalculateConstraintViolation(opt.x, opt.y);

      std::cout << "ctol is " << ctol << ", at pen itr " << pen_itr << std::endl;

      if (ctol < sqp_ctol_) {
        std::cout << "constraint satisfied at pen itr num " << pen_itr
                  << "constraint voilation value drops to " << ctol << ", under max_ctol "
                  << sqp_ctol_ << std::endl;
        opt.feasible = true;
        opt.message = "constraint satisfied at pen itr num " + std::to_string(pen_itr) +
                      "constraint voilation value drops to " + std::to_string(ctol) +
                      ", under max_ctol " + std::to_string(sqp_ctol_);
        return opt;
      }

      wk_ *= 10;
      ++pen_itr;
    }

    std::cout << "constraint not satisfied with total itr num " << pen_itr
              << "constraint voilation value drops to " << ctol << ", higher than max_ctol "
              << sqp_ctol_ << std::endl;
    return opt;
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
    sqp_pen_max_iter_ = opt.sqp_pen_max_iter;
    sqp_sub_max_iter_ = opt.sqp_sub_max_iter;
    sqp_ctol_ = opt.sqp_ctol;
    sqp_ftol_ = opt.sqp_ftol;
  }

 private:
  void SetPrimalWarmStart(const std::vector<double>& x0, const std::vector<double>& y0,
                          Eigen::VectorXd& primal_warm_start) {
    for (int i = 0; i < N_; i++) {
      primal_warm_start[2 * i] = x0[i];
      primal_warm_start[2 * i + 1] = y0[i];
    }
    slack_.resize(N_ - 2);
    for (int i = 0; i < N_ - 2; i++) {
      primal_warm_start[2 * N_ + i] = slack_[i];
    }
  }

  void CalculateKernel(Eigen::SparseMatrix<double>& P) {
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
  }

  void CalculateOffset(const std::vector<double>& x0, const std::vector<double>& y0,
                       Eigen::VectorXd& q) {
    q.setZero();
    for (int i = 0; i < N_; i++) {
      q(2 * i) = -wr_ * x0[i];
      q(2 * i + 1) = -wr_ * y0[i];
    }
    for (int i = 0; i < N_ - 2; i++) {
      q(2 * N_ + i) = 0.5 * wk_;
    }
  }

  void CalculateAffineConstraint(const std::vector<double>& x0, const std::vector<double>& y0,
                                 Eigen::SparseMatrix<double>& A, Eigen::VectorXd& l,
                                 Eigen::VectorXd& u) {
    A.setZero();
    l.setZero();
    u.setZero();

    int total_constraints = 4 * N_ - 4;
    A.resize(total_constraints, 2 * N_ + (N_ - 2));
    l.resize(total_constraints);
    u.resize(total_constraints);
    // initialize
    l.setConstant(-1e20);
    u.setConstant(1e20);

    int constraint_idx = 0;

    // 1) box constraints per point: x_i in [x0_i - bounds_, x0_i + bounds_], same for y
    for (int i = 0; i < N_; ++i) {
      // x bound
      A.coeffRef(constraint_idx, 2 * i) = 1.0;
      l(constraint_idx) = x0[i] - bounds_;
      u(constraint_idx) = x0[i] + bounds_;
      constraint_idx++;

      // y bound
      A.coeffRef(constraint_idx, 2 * i + 1) = 1.0;
      l(constraint_idx) = y0[i] - bounds_;
      u(constraint_idx) = y0[i] + bounds_;
      constraint_idx++;
    }

    // 2) slack constraints: slack_i <= 0  (so l = -inf, u = 0)
    for (int i = 0; i < N_ - 2; ++i) {
      A.coeffRef(constraint_idx, 2 * N_ + i) = 1.0;  // coefficient for slack var
      l(constraint_idx) = 0.0;                     // -inf
      u(constraint_idx) = 1e20;                       // slack >= 0
      constraint_idx++;
    }

    // 3) curvature linearized constraints: a^T x + slack_i <= curvature_limit
    //    Here we follow your linearization pattern but fix indexing and bounds.
    double curvature_constraint_sqr = std::pow(ds_ * ds_ * max_cur_, 2);
    for (int i = 0; i < N_ - 2; ++i) {
      double temp_sum_x0 = x0[i] + x0[i + 2] - 2.0 * x0[i + 1];
      double temp_sum_y0 = y0[i] + y0[i + 2] - 2.0 * y0[i + 1];

      // coefficients for x_f, x_m, x_l
      A.coeffRef(constraint_idx, 2 * i) = 2.0 * temp_sum_x0;
      A.coeffRef(constraint_idx, 2 * i + 1) = 2.0 * temp_sum_y0;

      A.coeffRef(constraint_idx, 2 * (i + 1)) = -4.0 * temp_sum_x0;
      A.coeffRef(constraint_idx, 2 * (i + 1) + 1) = -4.0 * temp_sum_y0;

      A.coeffRef(constraint_idx, 2 * (i + 2)) = 2.0 * temp_sum_x0;
      A.coeffRef(constraint_idx, 2 * (i + 2) + 1) = 2.0 * temp_sum_y0;

      // slack variable for this curvature constraint
      A.coeffRef(constraint_idx, 2 * N_ + i) = 1.0;

      // RHS bounds: a^T x + slack <= curvature_constraint_sqr + something_linear_from_ref
      // Keep your original RHS if intended, but ensure l <= u.
      // I keep your original expression but ensure correct sign:
      double rhs = curvature_constraint_sqr + temp_sum_x0 * temp_sum_x0 + temp_sum_y0 * temp_sum_y0;
      l(constraint_idx) = -1e20;
      u(constraint_idx) = rhs;
      constraint_idx++;
    }

    // final check
    if (constraint_idx != total_constraints) {
      // optional: throw or LOG error
    }
  }

  double CalculateConstraintViolation(const std::vector<double>& x0,
                                      const std::vector<double>& y0) {
    // compute average interval length
    double total_length = 0.0;
    for (int i = 1; i < N_; ++i) {
      double dx = x0[i] - x0[i - 1];
      double dy = y0[i] - y0[i - 1];
      total_length += std::sqrt(dx * dx + dy * dy);
    }
    double avg_interval = total_length / std::max(1, N_ - 1);
    double interval_sqr = avg_interval * avg_interval;
    double curvature_constraint_sqr = (interval_sqr * max_cur_) * (interval_sqr * max_cur_);

    double max_cviolation = -std::numeric_limits<double>::infinity();

    for (int i = 0; i < N_ - 2; ++i) {
      double dx = -2.0 * x0[i + 1] + x0[i] + x0[i + 2];
      double dy = -2.0 * y0[i + 1] + y0[i] + y0[i + 2];
      double sqr = dx * dx + dy * dy;
      double violation = sqr - curvature_constraint_sqr;  // >0 means violated
      if (violation > max_cviolation) max_cviolation = violation;
    }
    return max_cviolation;
  }

 private:
  int N_;
  double ds_;
  std::vector<double> slack_;

  // 权重
  double wr_ = 100.0;
  double wl_ = 10.0;
  double ws_ = 10.0;
  double wk_ = 1000.0;

  // 约束边界
  double bounds_ = 0.5;
  double max_cur_ = 0.2;

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
  int sqp_pen_max_iter_ = 10;
  int sqp_sub_max_iter_ = 100;
  double sqp_ctol_ = 1e-3;
  double sqp_ftol_ = 1e-4;
};
}  // namespace fem