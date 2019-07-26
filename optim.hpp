// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//

#include <flashlight/flashlight.h>

#include <vector>

class AdagradOptimizer : public fl::FirstOrderOptimizer {
public:
  AdagradOptimizer(const std::vector<fl::Variable>& parameters,
                   double learning_rate,
                   double clip = 0.1,
                   double epsilon = 1e-8)
    : FirstOrderOptimizer(parameters, learning_rate)
    , epsilon_(epsilon)
    , clip_(clip)
  {
    variance_.reserve(parameters.size());
    for (const auto& param : parameters_) {
      variance_.push_back(af::constant(epsilon, param.dims(), param.type()));
      variance_.back().eval();
    }
  }

  void step()
  {
    for (size_t i = 0; i < parameters_.size(); i++) {
      if (!parameters_[i].isGradAvailable()) {
        continue;
      }

      af::array& grad = parameters_[i].grad().array();
      af::array& data = parameters_[i].array();
      af::array& variance = variance_[i];

      float grad_norm = af::norm(grad, AF_NORM_EUCLID);
      if (grad_norm > clip_) {
        grad = grad * (clip_ / grad_norm);
      }
      af::eval(grad);

      variance = variance + grad * grad;
      af::eval(variance);
      data = data - lr_ * grad / af::sqrt(variance);
      af::eval(data);
    }
  }

  std::string prettyString() const
  {
    return "AdagradOptimizer";
  }

private:
  std::vector<af::array> variance_;
  double epsilon_, clip_, weight_decay_;
};
