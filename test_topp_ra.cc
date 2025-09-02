#include <iostream>
#include <vector>
#include "matplotlibcpp.h"
#include "topp_ra.h"
namespace plt = matplotlibcpp;

int main() {
  auto [res, bounds] = test_topp_ra();

  if (res.feasible) {
    plt::figure(1);
    plt::plot(res.t, res.s);
    plt::title("ToppRA Position Profile");
    plt::ylabel("Position (m)");
    plt::xlabel("Time (s)");
    plt::grid(true);

    plt::figure(2);
    plt::plot(res.t, res.v);
    plt::title("ToppRA Velocity Profile");
    plt::ylabel("Velocity (m/s)");
    plt::xlabel("Time (s)");
    plt::grid(true);

    plt::figure(3);
    plt::plot(res.t, res.a);
    plt::title("ToppRA Acceleration Profile");
    plt::ylabel("Acceleration (m/s²)");
    plt::xlabel("Time (s)");
    plt::grid(true);

    plt::show();
  } else {
    std::cerr << "ToppRA failed: " << res.message << "\n";
  }
  return 0;
}