#include <vector>
#include <cmath>
#include <algorithm>

class IDMPlanner {
 public:
  struct IDMParams {
    double a_min = -5.0;  // 最大减速度 -5.0 - -3.0 m/s²
    double a_max = 2.0;   // 最大加速度 1.0 - 1.5 m/s²
    double b = 2.5;       // 舒适减速度 1.5 - 2.0 m/s²
    double s_0 = 2.5;     // 最小安全间距 2 - 3 m
    double T = 1.25;      // 安全车头时距 1.0 - 1.5 s
    double d = 4.0;       // 加速到期望速度的曲线形状
  };

  IDMPlanner() {}
  IDMPlanner(const IDMParams& idm_params) : idm_params_(idm_params) {}

  double CalculateAcc(double delta_t, double delta_s, double curr_v, double leader_v,
                      double desire_velocity) {
    double est_acc = 0.0;

    auto safe_s = [&](double delta_v) -> double {
      return idm_params_.s_0 + std::max(0.0, curr_v * idm_params_.T +
                                                 curr_v * delta_v / 2.0 /
                                                     std::sqrt(idm_params_.a_max * idm_params_.b));
    };

    est_acc =
        idm_params_.a_max * (1.0 - std::pow(curr_v / desire_velocity, idm_params_.d) -
                             std::pow(safe_s(curr_v - leader_v) / std::max(0.001, delta_s), 2.0));

    return std::clamp(est_acc, idm_params_.a_min, idm_params_.a_max);
  }

  void set_params(const IDMParams& idm_params) { idm_params_ = idm_params; }

 private:
  IDMParams idm_params_;
};