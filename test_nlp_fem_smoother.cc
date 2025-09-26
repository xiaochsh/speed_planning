#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <gtest/gtest.h>
#include "nlp_fem_smoother.h"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

namespace fem {

// 测试夹具类
class PathSmootherTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 公共设置
    ds = 0.25;
    options.sqp_ctol = 1e-3;
    options.sqp_ftol = 1e-4;
    options.sqp_pen_max_iter = 10;
    options.sqp_sub_max_iter = 100;
  }

  // 生成直线参考线
  void generateStraightLine(std::vector<double>& x, std::vector<double>& y, double length = 50.0,
                            double noise_std = 0.01) {
    std::default_random_engine gen(42);
    std::normal_distribution<double> noise(0.0, noise_std);

    for (double s = 0; s <= length; s += ds) {
      x.push_back(s + noise(gen));
      y.push_back(noise(gen));
    }
  }

  // 生成圆弧参考线
  void generateArc(std::vector<double>& x, std::vector<double>& y, double R = 10.0,
                   double angle = M_PI, double noise_std = 0.01) {
    std::default_random_engine gen(42);
    std::normal_distribution<double> noise(0.0, noise_std);

    int num_points = static_cast<int>(R * angle / ds);
    for (int i = 0; i <= num_points; i++) {
      double theta = i * angle / num_points;
      x.push_back(R * cos(theta) + noise(gen));
      y.push_back(R * sin(theta) + noise(gen));
    }
  }

  // 生成S形曲线参考线
  void generateScurve(std::vector<double>& x, std::vector<double>& y, double length = 50.0,
                      double amplitude = 5.0, double noise_std = 0.01) {
    std::default_random_engine gen(42);
    std::normal_distribution<double> noise(0.0, noise_std);

    for (double s = 0; s <= length; s += ds) {
      x.push_back(s + noise(gen));
      y.push_back(amplitude * sin(2 * M_PI * s / length) + noise(gen));
    }
  }

  // 计算路径长度
  double computePathLength(const std::vector<double>& x, const std::vector<double>& y) {
    double length = 0.0;
    for (size_t i = 1; i < x.size(); i++) {
      double dx = x[i] - x[i - 1];
      double dy = y[i] - y[i - 1];
      length += std::sqrt(dx * dx + dy * dy);
    }
    return length;
  }

  // 计算最大曲率
  double computeMaxCurvature(const std::vector<double>& x, const std::vector<double>& y) {
    std::vector<double> kappa = computeCurvature(x, y, ds);
    return *std::max_element(kappa.begin(), kappa.end(),
                             [](double a, double b) { return std::abs(a) < std::abs(b); });
  }

  // 计算曲率变化率
  double computeCurvatureChangeRate(const std::vector<double>& x, const std::vector<double>& y) {
    std::vector<double> kappa = computeCurvature(x, y, ds);
    double max_change = 0.0;
    for (size_t i = 1; i < kappa.size(); i++) {
      double change = std::abs(kappa[i] - kappa[i - 1]) / ds;
      if (change > max_change) max_change = change;
    }
    return max_change;
  }

  // 计算路径与参考线的最大偏差
  double computeMaxDeviation(const std::vector<double>& x1, const std::vector<double>& y1,
                             const std::vector<double>& x2, const std::vector<double>& y2) {
    double max_dev = 0.0;
    for (size_t i = 0; i < x1.size(); i++) {
      double dx = x1[i] - x2[i];
      double dy = y1[i] - y2[i];
      double dev = std::sqrt(dx * dx + dy * dy);
      if (dev > max_dev) max_dev = dev;
    }
    return max_dev;
  }

  // 计算曲率 (与主代码中的相同)
  std::vector<double> computeCurvature(const std::vector<double>& x, const std::vector<double>& y,
                                       double ds) {
    int N = x.size();
    std::vector<double> kappa(N, 0.0);

    for (int i = 1; i < N - 1; i++) {
      double dx = (x[i + 1] - x[i - 1]) / (2 * ds);
      double dy = (y[i + 1] - y[i - 1]) / (2 * ds);
      double ddx = (x[i + 1] - 2 * x[i] + x[i - 1]) / (ds * ds);
      double ddy = (y[i + 1] - 2 * y[i] + y[i - 1]) / (ds * ds);

      double denom = std::pow(dx * dx + dy * dy, 1.5);
      if (denom > 1e-6) {
        kappa[i] = std::fabs(dx * ddy - dy * ddx) / denom;
      }
    }

    return kappa;
  }

  void plot(const std::vector<double>& rawx, const std::vector<double>& rawy,
            const std::vector<double>& optx, const std::vector<double>& opty) {
    plt::figure(1);
    plt::plot(rawx, rawy, {{"ls", "-"}, {"c", "red"}, {"label", "Raw Reference Line"}});
    plt::plot(optx, opty, {{"ls", "-"}, {"c", "blue"}, {"label", "Optimized Reference Line"}});
    plt::title("Reference Line");
    plt::ylabel("y (m)");
    plt::xlabel("x (m)");
    plt::legend();
    plt::grid(true);

    // ---- 曲率绘图 ----
    auto kappa_ref = computeCurvature(rawx, rawy, ds);
    auto kappa_opt = computeCurvature(optx, opty, ds);
    std::vector<double> s(kappa_ref.size());
    for (int i = 0; i < s.size(); i++) s[i] = i * ds;

    plt::figure(2);
    plt::plot(s, kappa_ref, {{"c", "red"}, {"label", "Raw Curvature"}});
    plt::plot(s, kappa_opt, {{"c", "blue"}, {"label", "Optimized Curvature"}});
    plt::title("Curvature Comparison");
    plt::ylabel("Curvature (1/m)");
    plt::xlabel("s (m)");
    plt::legend();
    plt::grid(true);

    plt::show();
  }

  // 公共变量
  double ds;
  NlpFemSmoother::Options options;
};

// // 测试用例1: 直线路径平滑
// TEST_F(PathSmootherTest, StraightLineSmoothing) {
//   std::vector<double> refx, refy;
//   generateStraightLine(refx, refy, 50.0, 0.05);

//   NlpFemSmoother smoother(refx.size(), ds);
//   smoother.setWeights(1.0, 1.0, 1000.0, 1000000.0);
//   smoother.setBounds(0.5, 0.02);
//   smoother.setOptions(options);

//   auto res = smoother.solve(refx, refy);

//   ASSERT_TRUE(res.feasible) << "Optimization failed: " << res.message;

//   // 验证路径长度变化不大
//   double orig_length = computePathLength(refx, refy);
//   double smooth_length = computePathLength(res.x, res.y);
//   EXPECT_NEAR(orig_length, smooth_length, 1.0);

//   // 验证曲率接近零（直线）
//   double max_kappa = computeMaxCurvature(res.x, res.y);
//   EXPECT_LT(max_kappa, 0.01);

//   // 验证与原始路径的偏差在允许范围内
//   double max_dev = computeMaxDeviation(refx, refy, res.x, res.y);
//   EXPECT_LT(max_dev, 0.5);

//   // 绘图
//   plot(refx, refy, res.x, res.y);
// }

// // 测试用例2: 圆弧路径平滑
// TEST_F(PathSmootherTest, ArcSmoothing) {
//   std::vector<double> refx, refy;
//   generateArc(refx, refy, 15.0, M_PI, 0.03);

//   NlpFemSmoother smoother(refx.size(), ds);
//   smoother.setWeights(1.0, 1.0, 1000.0, 10000.0);
//   smoother.setBounds(0.5, 0.02);
//   smoother.setOptions(options);

//   auto res = smoother.solve(refx, refy);

//   ASSERT_TRUE(res.feasible) << "Optimization failed: " << res.message;

//   // 验证曲率连续性改善
//   double orig_change_rate = computeCurvatureChangeRate(refx, refy);
//   double smooth_change_rate = computeCurvatureChangeRate(res.x, res.y);
//   EXPECT_LT(smooth_change_rate, orig_change_rate);

//   // 验证与原始路径的偏差在允许范围内
//   double max_dev = computeMaxDeviation(refx, refy, res.x, res.y);
//   EXPECT_LT(max_dev, 0.5);

//   // 绘图
//   plot(refx, refy, res.x, res.y);
// }

// 测试用例3: S形曲线路径平滑
// TEST_F(PathSmootherTest, ScurveSmoothing) {
//   std::vector<double> refx, refy;
//   generateScurve(refx, refy, 60.0, 8.0, 0.05);

//   NlpFemSmoother smoother(refx.size(), ds);
//   smoother.setWeights(1.0, 1.0, 10000.0, 1000000.0);
//   smoother.setBounds(1.0, 0.05);
//   smoother.setOptions(options);

//   auto res = smoother.solve(refx, refy);

//   ASSERT_TRUE(res.feasible) << "Optimization failed: " << res.message;

//   // 验证曲率连续性改善
//   double orig_change_rate = computeCurvatureChangeRate(refx, refy);
//   double smooth_change_rate = computeCurvatureChangeRate(res.x, res.y);
//   EXPECT_LT(smooth_change_rate, orig_change_rate);

//   // 验证与原始路径的偏差在允许范围内
//   double max_dev = computeMaxDeviation(refx, refy, res.x, res.y);
//   EXPECT_LT(max_dev, 0.8);

//   // 绘图
//   plot(refx, refy, res.x, res.y);
// }

// // 测试用例4: 高噪声情况下的平滑
// TEST_F(PathSmootherTest, HighNoiseSmoothing) {
//   std::vector<double> refx, refy;
//   generateStraightLine(refx, refy, 40.0, 0.2);  // 高噪声

//   NlpFemSmoother smoother(refx.size(), ds);
//   smoother.setWeights(1.0, 1.0, 5000.0, 50000.0);  // 增加平滑权重
//   smoother.setBounds(1.0, 0.05);                   // 放宽边界
//   smoother.setOptions(options);

//   auto res = smoother.solve(refx, refy);

//   ASSERT_TRUE(res.feasible) << "Optimization failed: " << res.message;

//   // 验证曲率接近零（直线）
//   double max_kappa = computeMaxCurvature(res.x, res.y);
//   EXPECT_LT(max_kappa, 0.02);

//   // 验证与原始路径的偏差在允许范围内
//   double max_dev = computeMaxDeviation(refx, refy, res.x, res.y);
//   EXPECT_LT(max_dev, 1.0);

//   // 绘图
//   plot(refx, refy, res.x, res.y);
// }

// // 测试用例5: 权重参数敏感性测试
// TEST_F(PathSmootherTest, WeightSensitivity) {
//   std::vector<double> refx, refy;
//   generateScurve(refx, refy, 50.0, 6.0, 0.03);

//   // 测试不同的权重组合
//   std::vector<std::tuple<double, double, double, double>> weight_combinations = {
//       {1.0, 1.0, 1000.0, 10000.0},   // 默认
//       {0.1, 1.0, 1000.0, 10000.0},   // 低参考权重
//       {10.0, 1.0, 1000.0, 10000.0},  // 高参考权重
//       {1.0, 1.0, 100.0, 10000.0},    // 低平滑权重
//       {1.0, 1.0, 10000.0, 10000.0},  // 高平滑权重
//       {1.0, 1.0, 1000.0, 1000.0},    // 低曲率权重
//       {1.0, 1.0, 1000.0, 100000.0}   // 高曲率权重
//   };

//   for (const auto& weights : weight_combinations) {
//     NlpFemSmoother smoother(refx.size(), ds);
//     smoother.setWeights(std::get<0>(weights), std::get<1>(weights), std::get<2>(weights),
//                         std::get<3>(weights));
//     smoother.setBounds(0.5, 0.02);
//     smoother.setOptions(options);

//     auto res = smoother.solve(refx, refy);

//     EXPECT_TRUE(res.feasible) << "Optimization failed with weights: " << std::get<0>(weights)
//                               << ", " << std::get<1>(weights) << ", " << std::get<2>(weights)
//                               << ", " << std::get<3>(weights) << ": " << res.message;
    
//                               // 绘图
//     plot(refx, refy, res.x, res.y);
//   }
// }

// 测试用例6: U型弯平滑 (使用原始代码中的函数)
TEST_F(PathSmootherTest, UTurnSmoothing) {
  std::vector<double> refx, refy;

  // 复制原始代码中的generateUTurn函数逻辑
  double R = 12.0;
  double straight_len = 25.0;
  std::default_random_engine gen(42);
  std::normal_distribution<double> noise(0.0, 0.001);

  // 进入直线
  for (double xx = -R - straight_len; xx <= -R; xx += ds) {
    refx.push_back(xx + noise(gen));
    refy.push_back(R + noise(gen));
  }

  // 半圆
  int arc_pts = static_cast<int>(M_PI * R / ds);
  for (int i = 0; i <= arc_pts; i++) {
    double theta = M_PI / 2.0 - i * (M_PI / arc_pts);
    double xx = -R + R * cos(theta);
    double yy = R * sin(theta);
    refx.push_back(xx + noise(gen));
    refy.push_back(yy + noise(gen));
  }

  // 出去直线
  for (double xx = -R; xx >= -R - straight_len; xx -= ds) {
    refx.push_back(xx + noise(gen));
    refy.push_back(-R + noise(gen));
  }

  NlpFemSmoother smoother(refx.size(), ds);
  smoother.setWeights(1.0, 1.0, 10000.0, 1000000.0);
  smoother.setBounds(0.5, 0.02);
  smoother.setOptions(options);

  auto res = smoother.solve(refx, refy);

  ASSERT_TRUE(res.feasible) << "Optimization failed: " << res.message;

  // 验证曲率连续性改善
  double orig_change_rate = computeCurvatureChangeRate(refx, refy);
  double smooth_change_rate = computeCurvatureChangeRate(res.x, res.y);
  EXPECT_LT(smooth_change_rate, orig_change_rate);

  // 验证与原始路径的偏差在允许范围内
  double max_dev = computeMaxDeviation(refx, refy, res.x, res.y);
  EXPECT_LT(max_dev, 0.5);

  // 绘图
  plot(refx, refy, res.x, res.y);
}

}  // namespace fem

// 主函数 - 运行所有测试
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}