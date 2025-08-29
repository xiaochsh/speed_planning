#include "topp_ra.h"
#include "speed_optimizer.h"
#include "matplotlibcpp.h"
#include <iostream>

using namespace pjso;
using namespace toppra;
namespace plt = matplotlibcpp;

int main() {
  toppra::ToppRA::Result res0 = test_topp_ra();

  if (!res0.feasible) {
    std::cout << "ToppRA failed: " << res0.message << "\n";
    return -1;
  } else {
    res0.a[0] = 0.0;      // 确保初始加速度为0
    res0.a.back() = 0.0;  // 确保终端加速度为0
    res0.v[0] = 0.0;      // 确保初始速度为0
    res0.v.back() = 0.0;  // 确保终端速度为0
  }

  TrajectoryOptimizer solver(res0.s.size(), res0.dt.front());

  // 设置权重
  solver.setWeights(1.0, 1e5, 1.0, 1e5, 0.1, 1e5, 1e-4);

  // 设置约束边界
  solver.setBounds(0, 1000, -30, 20, -5, 2.5, -5, 5);

  // 设置求解器参数
  TrajectoryOptimizer::Options options;
  options.eps_abs = 1e-4;
  options.eps_rel = 1e-4;
  options.max_iter = 10000;
  options.verbose = true;
  options.warm_start = true;
  options.polish = false;
  solver.setOptions(options);

  // 输入初始状态
  pjso::TrajectoryOptimizer::Result res = solver.solve(res0.s, res0.v, res0.a);

  if (res.feasible) {
    std::cout << "\nOptimization Result:\n";
    for (int i = 0; i < res0.s.size(); i++)
      std::cout << "t=" << res.t[i] << "\ts=" << res.s[i] << "\tv=" << res.v[i]
                << "\ta=" << res.a[i] << "\tj=" << res.jerk[i] << "\n";
    plt::figure(1);
    plt::plot(res0.t, res0.s, {{"ls", "-"}, {"c", "red"}, {"label", "ToppRA Position"}});
    plt::plot(res.t, res.s, {{"ls", "-"}, {"c", "blue"}, {"label", "Optimized Position"}});
    plt::title("Position Profile");
    plt::ylabel("Position (m)");
    plt::xlabel("Time (s)");
    plt::legend();
    plt::grid(true);

    plt::figure(2);
    plt::plot(res0.t, res0.v, {{"ls", "-"}, {"c", "red"}, {"label", "ToppRA Velocity"}});
    plt::plot(res.t, res.v, {{"ls", "-"}, {"c", "blue"}, {"label", "Optimized Velocity"}});
    plt::title("Velocity Profile");
    plt::ylabel("Velocity (m/s)");
    plt::xlabel("Time (s)");
    plt::legend();
    plt::grid(true);

    plt::figure(3);
    plt::plot(res0.t, res0.a, {{"ls", "--"}, {"c", "red"}, {"label", "ToppRA Acceleration"}});
    plt::plot(res.t, res.a, {{"ls", "-"}, {"c", "blue"}, {"label", "Optimized Acceleration"}});
    plt::title("Acceleration Profile");
    plt::ylabel("Acceleration (m/s²)");
    plt::xlabel("Time (s)");
    plt::legend();
    plt::grid(true);

    plt::figure(4);
    plt::plot(res.t, res.jerk, {{"ls", "-"}, {"c", "blue"}, {"label", "Optimized Jerk"}});
    plt::title("Jerk Profile");
    plt::ylabel("Jerk (m/s³)");
    plt::xlabel("Time (s)");
    plt::legend();
    plt::grid(true);

    plt::show();
  } else {
    std::cout << "Optimization failed: " << res.message << "\n";
  }
}