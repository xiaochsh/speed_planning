#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

namespace toppra {

struct Options {
  double eps = 1e-9;             // numeric epsilon
  bool enforce_terminal = true;  // clamp last speed to vend
  bool throw_on_fail = false;    // throw if infeasible
};

class ToppRA {
 public:
  using FuncAV = std::function<double(double /*s*/, double /*v*/)>;  // a_min/a_max(s, v)
  using FuncV = std::function<double(double /*s*/)>;                 // v_max(s)

  ToppRA(FuncAV a_min, FuncAV a_max, FuncV v_max, Options opt = {})
      : a_min_(std::move(a_min)), a_max_(std::move(a_max)), v_max_(std::move(v_max)), opt_(opt) {}

  struct Result {
    std::vector<double> s;   // size N
    std::vector<double> v;   // size N
    std::vector<double> a;   // size N-1 (piecewise-constant over segments)
    std::vector<double> dt;  // size N-1 (segment times)
    std::vector<double> t;   // size N   (cumulative times)
    bool feasible = true;
    std::string message;
  };

  struct Bounds {
    double s_min = -std::numeric_limits<double>::infinity();
    double s_max = std::numeric_limits<double>::infinity();
    double v_min = 0.0;
    double v_max = std::numeric_limits<double>::infinity();
    double a_min = -std::numeric_limits<double>::infinity();
    double a_max = std::numeric_limits<double>::infinity();
    double j_min = -std::numeric_limits<double>::infinity();
    double j_max = std::numeric_limits<double>::infinity();
  };

  Result solve(const std::vector<double>& s, double v0, double vend) const {
    const std::size_t N = s.size();
    if (N < 2) return makeFail("s must contain at least 2 samples");

    for (std::size_t k = 0; k + 1 < N; ++k) {
      if (!(s[k + 1] > s[k])) return makeFail("s must be strictly increasing");
    }

    Result out;
    out.v.assign(N, 0.0);
    out.s.assign(s.begin(), s.end());

    const double v0_cap = (!std::isfinite(v0)) ? v_max_(s[0]) : v0;
    out.v[0] = std::max(0.0, std::min(v_max_(s[0]), v0_cap));

    for (std::size_t k = 0; k + 1 < N; ++k) {
      const double ds = s[k + 1] - s[k];
      const double amax = a_max_(s[k], out.v[k]);
      double v_next_sq = out.v[k] * out.v[k] + 2.0 * amax * ds;
      if (v_next_sq < 0.0) v_next_sq = 0.0;
      double v_next = std::sqrt(v_next_sq);
      const double vmax_next = v_max_(s[k + 1]);
      out.v[k + 1] = std::min(v_next, std::max(0.0, vmax_next));
    }

    if (opt_.enforce_terminal && std::isfinite(vend)) {
      out.v[N - 1] = std::min(out.v[N - 1], std::max(0.0, vend));
    } else {
      out.v[N - 1] = std::min(out.v[N - 1], std::max(0.0, v_max_(s[N - 1])));
    }

    for (std::size_t idx = N - 1; idx-- > 0;) {
      const std::size_t k = idx;
      const double ds = s[k + 1] - s[k];
      const double amin = a_min_(s[k], out.v[k + 1]);
      double v_cap_sq = out.v[k + 1] * out.v[k + 1] - 2.0 * amin * ds;
      if (v_cap_sq < 0.0) v_cap_sq = 0.0;
      double v_cap = std::sqrt(v_cap_sq);
      out.v[k] = std::min({out.v[k], v_cap, std::max(0.0, v_max_(s[k]))});
    }

    for (std::size_t k = 0; k < N; ++k) {
      if (!std::isfinite(out.v[k])) return makeFail("non-finite speed encountered");
      if (out.v[k] < -opt_.eps) return makeFail("negative speed encountered");
      if (out.v[k] > v_max_(s[k]) + 1e-6) return makeFail("speed exceeds v_max after projection");
    }

    out.a.assign(N > 1 ? N - 1 : 0, 0.0);
    out.dt.assign(N > 1 ? N - 1 : 0, 0.0);
    out.t.assign(N, 0.0);

    for (std::size_t k = 0; k + 1 < N; ++k) {
      const double ds = s[k + 1] - s[k];
      const double v0k = out.v[k];
      const double v1k = out.v[k + 1];
      const double denom = 2.0 * ds;
      out.a[k] = (v1k * v1k - v0k * v0k) / (denom > opt_.eps ? denom : opt_.eps);
      const double vsum = v0k + v1k;
      if (std::fabs(vsum) > opt_.eps) {
        out.dt[k] = 2.0 * ds / vsum;
      } else {
        if (std::fabs(out.a[k]) > opt_.eps)
          out.dt[k] = std::fabs((v1k - v0k) / out.a[k]);
        else
          out.dt[k] = ds / std::max(opt_.eps, v0k);
      }
      out.t[k + 1] = out.t[k] + out.dt[k];
    }

    return out;
  }

  // Resample result to uniform time grid
  Result resample(const Result& in, double dt_target) const {
    Result out;
    if (!in.feasible) return in;
    double T = in.t.back();
    std::size_t M = static_cast<std::size_t>(std::floor(T / dt_target)) + 1;
    out.t.resize(M);
    out.s.resize(M);
    out.v.resize(M);
    out.a.resize(M);

    for (std::size_t i = 0; i < M; ++i) {
      double ti = i * dt_target;
      if (ti > T) ti = T;
      out.t[i] = ti;
      // Find segment k s.t. t[k] <= ti <= t[k+1]
      auto it = std::upper_bound(in.t.begin(), in.t.end(), ti);
      std::size_t k = (it == in.t.begin()) ? 0 : (it - in.t.begin() - 1);
      if (k >= in.a.size()) k = in.a.size() - 1;
      double t0 = in.t[k], t1 = in.t[k + 1];
      double s0 = in.s[k], s1 = in.s[k + 1];
      double v0 = in.v[k], v1 = in.v[k + 1];
      double a = in.a[k];
      double tau = (t1 > t0) ? (ti - t0) / (t1 - t0) : 0.0;
      out.s[i] = s0 + (s1 - s0) * tau;
      out.v[i] = v0 + (v1 - v0) * tau;
      out.a[i] = a;
    }
    out.dt.assign(M - 1, dt_target);
    out.feasible = true;
    return out;
  }

 private:
  FuncAV a_min_, a_max_;
  FuncV v_max_;
  Options opt_;

  Result makeFail(std::string msg) const {
    Result r;
    r.feasible = false;
    r.message = std::move(msg);
    if (opt_.throw_on_fail) throw std::runtime_error(r.message);
    return r;
  }
};

inline std::vector<double> linspace(double s0, double s1, std::size_t N) {
  if (N < 2) return {s0, s1};
  std::vector<double> s(N);
  const double ds = (s1 - s0) / static_cast<double>(N - 1);
  for (std::size_t i = 0; i < N; ++i) s[i] = s0 + ds * static_cast<double>(i);
  return s;
}

}  // namespace toppra

std::pair<toppra::ToppRA::Result, toppra::ToppRA::Bounds> test_topp_ra() {
  using namespace toppra;

  auto s = linspace(0.0, 100.0, 501);
  auto kappa_of_s = [&s](double si) {
    auto gauss = [](double x, double mu, double sigma) {
      const double d = (x - mu) / sigma;
      return std::exp(-0.5 * d * d);
    };
    const double k1 = 0.02 * gauss(si, 30.0, 8.0);
    const double k2 = -0.015 * gauss(si, 70.0, 10.0);
    return k1 + k2;
  };

  const double v_limit = 30.0;
  const double a_lat_max = 3.0;
  const double a_acc_max = 2.0;
  const double a_brake_max = 5.0;

  ToppRA::FuncV v_max = [&](double si) {
    const double k = std::abs(kappa_of_s(si));
    const double k_eps = 1e-6;
    const double v_curve = std::sqrt(a_lat_max / std::max(k, k_eps));
    return std::min(v_limit, v_curve);
  };
  ToppRA::FuncAV a_min = [&](double, double) { return -a_brake_max; };
  ToppRA::FuncAV a_max = [&](double, double) { return a_acc_max; };

  ToppRA solver(a_min, a_max, v_max, Options{1e-9, true, false});
  auto res = solver.solve(s, 0.0, 0.0);

  ToppRA::Result uniform = solver.resample(res, 0.1);

  std::cout << "ToppRA Planning Result:\n";
  for (std::size_t i = 0; i < uniform.v.size(); i += 10) {
    std::cout << uniform.t[i] << "\t" << uniform.s[i] << "\t" << uniform.v[i] << "\t"
              << uniform.a[i] << "\n";
  }
  ToppRA::Bounds bounds;
  bounds.s_min = 0.0;
  bounds.s_max = 1000.0;
  bounds.v_min = 0.0;
  bounds.v_max = 30.0;
  bounds.a_min = -5.0;
  bounds.a_max = 2.5;
  bounds.j_min = -5.0;
  bounds.j_max = 5.0;

  return {uniform, bounds};
}
