#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include "kine_bicycle_mpc.h"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;
using namespace mpc;

int main() {
  // 参数配置
  int N = 30;          // 预测步长
  double Ts = 0.1;     // 采样时间
  double L = 2.7;      // 轴距
  double R = 20.0;     // 弯道半径
  double v_ref = 5.0;  // 期望速度

  KineBicycleMpc::Options opts;
  opts.sigma = 1e-6;

  KineBicycleMpc mpc(N, Ts);
  mpc.setOptions(opts);
  mpc.setWeights(10, 10, 5, 5, 10, 10, 10, 10, 1, 10, 1, 50);
  mpc.setBounds(-5.0, 2.5, -3.14 / 6.0, 2.14 / 6.0);

  // 初始状态 [x, y, psi, v]
  Eigen::VectorXd x0(4);
  x0 << 0.0, 0.0, 0.0, 0.0;

  // 参考轨迹：圆弧 (逆时针左转)
  std::vector<Eigen::VectorXd> x_ref(N + 1, Eigen::VectorXd(4));
  std::vector<Eigen::VectorXd> u_ref(N, Eigen::VectorXd(2));

  double omega = v_ref / R;             // 理想角速度
  double delta_ref = std::atan2(L, R);  // 理想转角

  for (int k = 0; k <= N; ++k) {
    double t = k * Ts;
    double theta = omega * t;  // 圆弧角度

    // 圆心在 (0, R)，车辆从 (0,0) 沿左转圆弧
    double x = R * std::sin(theta);
    double y = R * (1.0 - std::cos(theta));
    double psi = theta;  // 航向角随圆弧变化

    x_ref[k] << x, y, psi, v_ref;

    if (k == N) {
      x_ref[k][3] = 0.0;
    }

    if (k < N) {
      u_ref[k] << 0.5, delta_ref;  // 匀速、固定转角
    }
  }

  // 调用 MPC
  auto res = mpc.solve(x_ref, u_ref, x0);
  if (!res.feasible) {
    std::cout << res.message << std::endl;
    std::cerr << "MPC solve failed!" << std::endl;
    return -1;
  }

  // 打印结果
  std::cout << "First control action (a, delta): " << res.u_pred[0].transpose() << std::endl;

  std::cout << "Predicted sequence:" << std::endl;
  for (size_t k = 0; k < N; ++k) {
    std::cout << "k=" << k << " a=" << res.u_pred[k][0] << " delta=" << res.u_pred[k][1] << std::endl;
  }

  return 0;
}
