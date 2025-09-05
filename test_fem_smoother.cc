#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include "fem_smoother.h"

#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;
using namespace fem;

// 生成 U 型弯参考线：直线 + 半圆 + 直线
void generateUTurn(std::vector<double>& x, std::vector<double>& y, int N) {
  double R = 12.0;             // 半径 [m]
  double straight_len = 25.0;  // 直线长度 [m]
  double ds = 0.5;             // 采样步长 [m]

  // 随机噪声
  std::default_random_engine gen(42);
  std::normal_distribution<double> noise(0.0, 0.1);

  // 进入直线 (y=R, x从 -R-straight_len 到 -R)
  for (double xx = -R - straight_len; xx <= -R; xx += ds) {
    x.push_back(xx + noise(gen));
    y.push_back(R + noise(gen));
  }

  // 半圆 (中心在(-R,0), 从90°到-90°)
  int arc_pts = static_cast<int>(M_PI * R / ds);
  for (int i = 0; i <= arc_pts; i++) {
    double theta = M_PI / 2.0 - i * (M_PI / arc_pts);
    double xx = -R + R * cos(theta);
    double yy = R * sin(theta);
    x.push_back(xx + noise(gen));
    y.push_back(yy + noise(gen));
  }

  // 出去直线 (y=-R, x从 -R 到 -R-straight_len)
  for (double xx = -R; xx >= -R - straight_len; xx -= ds) {
    x.push_back(xx + noise(gen));
    y.push_back(-R + noise(gen));
  }
}

int main() {
  // 1. 生成 U 型弯参考线
  std::vector<double> refx, refy;
  generateUTurn(refx, refy, 0);
  int N = refx.size();
  std::cout << "Generated U-turn with " << N << " points.\n";

  // 2. 配置 FEM Smoother
  FemSmoother smoother(N);
  smoother.setWeights(50.0, 1.0, 10000.0);  // wr, wl, ws
  smoother.setBounds(0.5);           // corridor bounds

  // 3. 求解
  auto res = smoother.solve(refx, refy);

  if (res.feasible) {
    plt::figure(1);
    plt::plot(refx, refy, {{"ls", "-"}, {"c", "red"}, {"label", "Raw Reference Line"}});
    plt::plot(res.x, res.y, {{"ls", "-"}, {"c", "blue"}, {"label", "Optimized Reference Line"}});
    plt::title("Reference Line");
    plt::ylabel("y (m)");
    plt::xlabel("x (m)");
    plt::legend();
    plt::grid(true);

    plt::show();
  } else {
    std::cout << "Optimization failed: " << res.message << "\n";
  }

  return 0;
}
