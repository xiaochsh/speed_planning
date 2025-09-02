#include <iostream>
#include "path_optimizer.h"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;
using namespace pjso;

int main() {
  int N = 100;      // 采样点数量
  double ds = 0.2;  // 采样间距

  PathOptimizer opt(N, ds);

  // 设置权重（位置跟踪较强，速度/加速度平滑较弱）
  opt.setWeights(1.0,      // wx
                 10.0,     // wxe
                 1000.0,   // wdx
                 1.0,      // wdxe
                 10000,    // wddx
                 1.0,      // wddxe
                 1e-3,     // wdddx (jerk)
                 100000.0  // wobs (参考点跟踪)
  );

  // 初始条件
  std::vector<double> x0(N, 0.0);
  std::vector<double> dx0(N, 0.0);
  std::vector<double> ddx0(N, 0.0);

  std::vector<double> refx(N, 0.0);
  std::vector<double> kappa(N, 0.0);

  // 参考轨迹：车道保持
  for (int i = 0; i < N; i++) {
    double s = i * ds;
    refx[i] = 0.2 * std::sin(0.1 * s);
  }

  // 参考轨迹：换道
  // for (int i = 0; i < N; i++) {
  //   double s = i * ds;
  //   if (s < 5.0)
  //     refx[i] = 0.0;
  //   else if (s < 15.0) {
  //     refx[i] = (s - 5.0) * 3.75 / (15.0 - 5.0);
  //   } else if (s < 20.0) {
  //     refx[i] = 3.75;
  //   }
  // }

  // 参考轨迹：避障
  // for (int i = 0; i < N; i++) {
  //   double s = i * ds;
  //   if (s < 7.5)
  //     refx[i] = 0.0;
  //   else if (s < 10.0) {
  //     refx[i] = (s - 7.5) * 0.5 / (10.0 - 7.5);
  //   } else if (s < 15.0) {
  //     refx[i] = 0.5;
  //   } else if (s < 17.5) {
  //     refx[i] = 0.5 - (s - 15.0) * 0.5 / (17.5 - 15.0);
  //   } else if (s < 20.0)
  //     refx[i] = 0.0;
  // }

  // 参考轨迹：超车

  // 计算离散曲率
  for (int i = 1; i < N - 1; ++i) {
    double dx = (refx[i + 1] - refx[i - 1]) / (2 * ds);
    double ddx = (refx[i + 1] - 2 * refx[i] + refx[i - 1]) / (ds * ds);
    kappa[i] = std::abs(ddx) / std::pow(1 + dx * dx, 1.5);
  }
  // 边界点
  kappa[0] = std::abs((refx[2] - 2 * refx[1] + refx[0]) / (ds * ds)) /
             std::pow(1 + ((refx[1] - refx[0]) / ds) * ((refx[1] - refx[0]) / ds), 1.5);
  kappa[N - 1] =
      std::abs((refx[N - 1] - 2 * refx[N - 2] + refx[N - 3]) / (ds * ds)) /
      std::pow(1 + ((refx[N - 1] - refx[N - 2]) / ds) * ((refx[N - 1] - refx[N - 2]) / ds), 1.5);

  // 计算最大横向偏移约束 x_max_kappa
  double max_alpha = 35.0 * M_PI / 180.0;  // 最大前轮转角
  double tan_max_alpha = std::tan(max_alpha);
  double L = 2.7;                     // 车辆轴距
  std::vector<double> x_max_kappa(N, 10.0);  // 默认较大值
  for (int i = 0; i < N; ++i) {
    if (kappa[i] > 1e-6) {  // 避免除零
      x_max_kappa[i] = (tan_max_alpha + std::abs(kappa[i]) * L) / (tan_max_alpha * kappa[i]);  // 最大横向偏移不超过10米
    }
  }

  // 设置边界
  opt.setBounds(-10.0, 10.0,  // x 范围
                -1.0, 1.0,    // dx 范围
                -2.0, 2.0,    // ddx 范围
                -5.0, 5.0,    // jerk 范围
                x_max_kappa   // 曲率约束
  );

  // 求解
  auto res = opt.solve(x0, dx0, ddx0, refx);

  if (!res.feasible) {
    std::cout << "QP solve failed: " << res.message << std::endl;
    return -1;
  }

  // 打印结果
  std::cout << "=== Optimized path ===" << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "s=" << res.s[i] << "  x=" << res.s[i] << "  dx=" << res.dx[i]
              << "  ddx=" << res.ddx[i] << "  jerk=" << res.dddx[i] << std::endl;
  }

  if (res.feasible) {
    std::cout << "\nOptimization Result:\n";
    for (int i = 0; i < res.s.size(); i++)
      std::cout << "s=" << res.s[i] << "\tx=" << res.x[i] << "\tdx=" << res.dx[i]
                << "\tddx=" << res.ddx[i] << "\tdddx=" << res.dddx[i] << "\n";
    plt::figure(1);
    plt::plot(res.s, refx, {{"ls", "-"}, {"c", "red"}, {"label", "Reference Lateral Deviation"}});
    plt::plot(res.s, res.x, {{"ls", "-"}, {"c", "blue"}, {"label", "Optimized Lateral Deviation"}});
    plt::title("Lateral Deviation Profile");
    plt::ylabel("Lateral Deviation (m)");
    plt::xlabel("Station (m)");
    plt::legend();
    plt::grid(true);

    plt::figure(2);
    plt::plot(res.s, res.dx, {{"ls", "-"}, {"c", "blue"}, {"label", "Optimized Lateral Velocity"}});
    plt::title("Lateral Velocity Profile");
    plt::ylabel("Lateral Velocity (m/m)");
    plt::xlabel("Station (m)");
    plt::legend();
    plt::grid(true);

    plt::figure(3);
    plt::plot(res.s, res.ddx,
              {{"ls", "-"}, {"c", "blue"}, {"label", "Optimized Lateral Acceleration"}});
    plt::title("Lateral Acceleration Profile");
    plt::ylabel("Lateral Acceleration (m/s²)");
    plt::xlabel("Station (m)");
    plt::legend();
    plt::grid(true);

    plt::figure(4);
    plt::plot(res.s, res.dddx, {{"ls", "-"}, {"c", "blue"}, {"label", "Optimized Lateral Jerk"}});
    ;
    plt::title("Lateral Jerk Profile");
    plt::ylabel("Lateral Jerk (m/s³)");
    plt::xlabel("Station (m)");
    plt::legend();
    plt::grid(true);

    plt::show();
  } else {
    std::cout << "Optimization failed: " << res.message << "\n";
  }

  return 0;
}
