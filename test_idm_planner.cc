#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>

#include "idm_planner.h"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

int main() {
  std::cout << "IDM Planner 测试示例" << std::endl;
  std::cout << "====================" << std::endl << std::endl;

  // 创建 IDM 规划器实例（使用默认参数）
  IDMPlanner planner;

  // 测试场景 1: 自由流情况（前方无车或距离很远）
  std::cout << "测试场景 1: 自由流情况" << std::endl;
  std::cout << "当前速度: 20 m/s, 前车速度: 20 m/s, 距离: 200 m" << std::endl;
  double acc1 = planner.CalculateAcc(0.1, 200.0, 20.0, 20.0, 33.3);
  std::cout << "计算加速度: " << std::fixed << std::setprecision(3) << acc1 << " m/s²" << std::endl;
  std::cout << "预期: 轻微加速（接近期望速度）" << std::endl << std::endl;

  // 测试场景 2: 正常跟车情况
  std::cout << "测试场景 2: 正常跟车情况" << std::endl;
  std::cout << "当前速度: 20 m/s, 前车速度: 18 m/s, 距离: 40 m" << std::endl;
  double acc2 = planner.CalculateAcc(0.1, 40.0, 20.0, 18.0, 33.3);
  std::cout << "计算加速度: " << acc2 << " m/s²" << std::endl;
  std::cout << "预期: 轻微减速（保持安全距离）" << std::endl << std::endl;

  // 测试场景 3: 紧急制动情况
  std::cout << "测试场景 3: 紧急制动情况" << std::endl;
  std::cout << "当前速度: 20 m/s, 前车速度: 5 m/s, 距离: 20 m" << std::endl;
  double acc3 = planner.CalculateAcc(0.1, 20.0, 20.0, 5.0, 33.3);
  std::cout << "计算加速度: " << acc3 << " m/s²" << std::endl;
  std::cout << "预期: 明显减速（避免碰撞）" << std::endl << std::endl;

  // 测试场景 4: 前车更快，逐渐拉开距离
  std::cout << "测试场景 4: 前车更快" << std::endl;
  std::cout << "当前速度: 20 m/s, 前车速度: 25 m/s, 距离: 40 m" << std::endl;
  double acc4 = planner.CalculateAcc(0.1, 40.0, 20.0, 25.0, 33.3);
  std::cout << "计算加速度: " << acc4 << " m/s²" << std::endl;
  std::cout << "预期: 轻微加速（前车正在拉开距离）" << std::endl << std::endl;

  // 测试场景 5: 接近期望速度
  std::cout << "测试场景 5: 接近期望速度" << std::endl;
  std::cout << "当前速度: 32 m/s, 前车速度: 32 m/s, 距离: 100 m" << std::endl;
  double acc5 = planner.CalculateAcc(0.1, 100.0, 32.0, 32.0, 33.3);
  std::cout << "计算加速度: " << acc5 << " m/s²" << std::endl;
  std::cout << "预期: 非常小的加速度（几乎维持当前速度）" << std::endl << std::endl;

  // 测试场景 6: 自定义参数
  std::cout << "测试场景 6: 自定义参数（更保守的跟车行为）" << std::endl;
  IDMPlanner::IDMParams conservative_params;
  conservative_params.a_min = -5.0;
  conservative_params.a_max = 0.8;  // 更小的最大加速度
  conservative_params.b = 2.0;      // 更大的舒适减速度
  conservative_params.s_0 = 3.0;    // 更大的最小安全间距
  conservative_params.T = 1.5;      // 更长的安全时距
  conservative_params.d = 4.0;

  IDMPlanner conservative_planner(conservative_params);
  std::cout << "当前速度: 20 m/s, 前车速度: 18 m/s, 距离: 40 m" << std::endl;
  double acc6 = conservative_planner.CalculateAcc(0.1, 40.0, 20.0, 18.0, 30.0);
  std::cout << "计算加速度: " << acc6 << " m/s²" << std::endl;
  std::cout << "预期: 比默认参数更大的减速度（更保守的跟车）" << std::endl << std::endl;

  // 输出总结
  std::cout << "测试总结:" << std::endl;
  std::cout << "场景1 (自由流): " << acc1 << " m/s²" << std::endl;
  std::cout << "场景2 (正常跟车): " << acc2 << " m/s²" << std::endl;
  std::cout << "场景3 (紧急制动): " << acc3 << " m/s²" << std::endl;
  std::cout << "场景4 (前车更快): " << acc4 << " m/s²" << std::endl;
  std::cout << "场景5 (接近期望速度): " << acc5 << " m/s²" << std::endl;
  std::cout << "场景6 (保守参数): " << acc6 << " m/s²" << std::endl;

  // 测试场景7
  double delta_s = 100.0;
  double leader_v = 20.0;
  double leader_a = -0.5;
  double curr_v = 30.0;
  double desire_velocity = 33.3;
  double horizon = 50.0;
  double delta_t = 0.1;
  double t = 0.0;

  std::vector<double> trajectory[4];
  while (t <= horizon) {
    double est_acc = planner.CalculateAcc(delta_t, delta_s, curr_v, leader_v, desire_velocity);

    // update
    t += delta_t;
    delta_s += (leader_v * delta_t + 0.5 * leader_a * delta_t * delta_t) -
               (curr_v * delta_t + 0.5 * est_acc * delta_t * delta_t);
    leader_v += leader_a * delta_t;
    leader_v = std::max(leader_v, 0.0);
    curr_v += est_acc * delta_t;
    trajectory[0].push_back(t);
    trajectory[1].push_back(delta_s);
    trajectory[2].push_back(curr_v);
    trajectory[3].push_back(est_acc);
  }

  plt::figure(1);
  plt::plot(trajectory[0], trajectory[1]);
  plt::title("IDMPlanner Trajectory Profile");
  plt::ylabel("Position (m)");
  plt::xlabel("Time (s)");
  plt::grid(true);

  plt::figure(2);
  plt::plot(trajectory[0], trajectory[2]);
  plt::title("IDMPlanner Trajectory Profile");
  plt::ylabel("Velocity (m/s)");
  plt::xlabel("Time (s)");
  plt::grid(true);

  plt::figure(3);
  plt::plot(trajectory[0], trajectory[3]);
  plt::title("IDMPlanner Trajectory Profile");
  plt::ylabel("Acceleration (m/s²)");
  plt::xlabel("Time (s)");
  plt::grid(true);

  plt::show();

  return 0;
}