// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//

#include "data.hpp"
#include "optim.hpp"
#include "transformer.hpp"

#include <flashlight/flashlight.h>
#include <cereal/archives/binary.hpp>

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <unordered_map>

class Args {

public:
  int bsz;
  int bptt;
  int d_model;
  int d_ff;
  int n_blocks;
  int n_heads;
  int warmup;
  int warmup_loss;
  int ngram;
  int threshold;
  int nepoch;
  float lr;
  float dropout;
  float clip;
  float epsilon;
  bool use_cache;
  std::string data;

  Args()
    : bsz(32)
    , bptt(64)
    , d_model(512)
    , d_ff(2048)
    , n_blocks(12)
    , n_heads(4)
    , warmup(8000)
    , warmup_loss(-1)
    , ngram(1)
    , threshold(1)
    , nepoch(30)
    , lr(0.025)
    , dropout(0.3)
    , clip(0.1)
    , epsilon(1e-6)
    , use_cache(false)
  {
  }

  void print(std::ostream& out)
  {
    out << "bsz:              " << bsz << std::endl;
    out << "bptt:             " << bptt << std::endl;
    out << "d_model:          " << d_model << std::endl;
    out << "d_ff:             " << d_ff << std::endl;
    out << "n_blocks:         " << n_blocks << std::endl;
    out << "n_heads:          " << n_heads << std::endl;
    out << "warmup:           " << warmup << std::endl;
    out << "warmup_loss       " << warmup_loss << std::endl;
    out << "ngram:            " << ngram << std::endl;
    out << "threshold         " << threshold << std::endl;
    out << "lr:               " << lr << std::endl;
    out << "dropout:          " << dropout << std::endl;
    out << "clip:             " << clip << std::endl;
    out << "epsilon:          " << epsilon << std::endl;
    out << "data:             " << data << std::endl;
    out << std::endl;
  }

  void print()
  {
    print(std::cout);
  }

  void parse(const std::vector<std::string>& args)
  {
    int i = 1;
    while (i < args.size()) {
      try {
        if (args[i] == "--bptt") {
          bptt = std::stoi(args.at(i + 1));
        } else if (args[i] == "--bsz") {
          bsz = std::stoi(args.at(i + 1));
        } else if (args[i] == "--d_model") {
          d_model = std::stoi(args.at(i + 1));
        } else if (args[i] == "--d_ff") {
          d_ff = std::stoi(args.at(i + 1));
        } else if (args[i] == "--n_blocks") {
          n_blocks = std::stoi(args.at(i + 1));
        } else if (args[i] == "--n_heads") {
          n_heads = std::stoi(args.at(i + 1));
        } else if (args[i] == "--warmup") {
          warmup = std::stoi(args.at(i + 1));
        } else if (args[i] == "--warmup_loss") {
          warmup_loss = std::stoi(args.at(i + 1));
        } else if (args[i] == "--ngram") {
          ngram = std::stoi(args.at(i + 1));
        } else if (args[i] == "--threshold") {
          threshold = std::stoi(args.at(i + 1));
        } else if (args[i] == "--nepoch") {
          nepoch = std::stoi(args.at(i + 1));
        } else if (args[i] == "--lr") {
          lr = std::stof(args.at(i + 1));
        } else if (args[i] == "--dropout") {
          dropout = std::stof(args.at(i + 1));
        } else if (args[i] == "--clip") {
          clip = std::stof(args.at(i + 1));
        } else if (args[i] == "--epsilon") {
          epsilon = std::stof(args.at(i + 1));
        } else if (args[i] == "--use_cache") {
          use_cache = true;
          i -= 1;
        } else if (args[i] == "--data") {
          data = args.at(i + 1);
        } else {
          std::cerr << "Unrecognized option: " << args[i] << std::endl;
          std::exit(EXIT_FAILURE);
        }
        i += 2;
      } catch (std::out_of_range) {
        std::cerr << args[i] << " is missing an argument!" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }

  void parse(int argc, char** argv)
  {
    parse(std::vector<std::string>(argv, argv + argc));
  }
};

fl::Variable logsumexp(const fl::Variable& lhs, const fl::Variable& rhs)
{
  auto m = max(max(lhs, rhs), -1.0e10);
  return m + log(exp(lhs - m) + exp(rhs - m));
}

fl::Variable myloss(const fl::Variable& input,
                    const fl::Variable& targets,
                    bool flat_loss=false,
                    bool eval=true)
{
  int32_t d0 = input.dims(0);
  int32_t d1 = input.dims(1);
  int32_t d2 = input.dims(2);
  int32_t n = targets.elements();
  int32_t m = targets.dims(2);
  auto _idx = af::moddims(targets.array(), af::dim4(n)) +
    d0 * af::range(af::dim4(n), -1, u32);
  auto scores = tile(logSoftmax(input, 0), af::dim4(1, 1, 1, m));

  scores = moddims(scores, af::dim4(scores.elements()));
  scores = scores(_idx);
  scores = moddims(scores, af::dim4(d1, d2, m));

  if (flat_loss) {
    af::array mask = targets.array() != 1;
    scores = scores * fl::Variable(mask.as(f32), false);
    return -1.0 * mean(flat(scores), {0});
  }

  af::array mask = targets.array() != 1;
  if (eval) {
    scores = scores + fl::Variable(af::log(mask.as(f32)), false);
  } else {
    scores = scores + fl::Variable(af::log(1.0e-30 + mask.as(f32)), false);
  }
  scores = reorder(scores, 2, 1, 0);

  //fl::Variable buffer(af::constant(-1.0e20, m, d2), false);
  //fl::Variable a(af::constant(0.0, m, d2), false);
  auto buffer = scores(af::span, af::span, 0);
  for (int t = 1; t < d1; t++) {
    //buffer = concatenate(std::vector<fl::Variable>({
    //      buffer(af::seq(1, m-1), af::span),
    //        fl::Variable(af::constant(-1.0e20, 1, d2), false)}), 0);
    //buffer = logsumexp(buffer, a + reorder(scores(t, af::span, af::span), 2, 1, 0));
    //buffer = logsumexp(buffer, a + scores(af::span, af::span, t));
    //a = tile(buffer(0, af::span), af::dim4(m, 1, 1, 1));

    auto tmp = scores.slice(t) + tile(buffer.row(0), af::dim4(m));
    buffer = logsumexp(buffer.rows(1, m-1), tmp.rows(0, m-2));
    buffer = fl::concatenate({buffer, tmp.row(m-1)}, 0);
  }
  return -1.0 * mean(flat(buffer(0, af::span)), {0}) / d1;
}

int main(int argc, char** argv)
{
  Args args;
  args.parse(argc, argv);

  int bsz = args.bsz;
  int bptt = args.bptt;
  int warmup = args.warmup;
  int32_t ngram = args.ngram;
  float lr = args.lr;
  float dropout = args.dropout;

  af::setSeed(439);

  std::cout << std::endl;
  std::cout << "*************************************" << std::endl;
  std::cout << "***        TRANSFORMER LM        ****" << std::endl;
  std::cout << "*************************************" << std::endl;
  std::cout << std::endl;
  std::cout << std::setprecision(4);
  args.print();

  //// LOAD DATA ////
  Dict dict;
  dict.build_dict(args.data + "/train.txt", ngram);
  dict.build_dict(args.data + "/valid.txt", 1);
  dict.build_dict(args.data + "/test.txt", 1);
  dict = dict.threshold(args.threshold);
  CharLMDataset traindata(args.data + "/train.txt", dict, bsz, bptt, ngram);
  CharLMDataset validdata(args.data + "/valid.txt", dict, bsz, bptt, ngram);
  CharLMDataset testdata(args.data + "/test.txt", dict, bsz, bptt, ngram);

  std::cout << "Dict size:         " << dict.size() << std::endl;
  std::cout << "Training set size: " << traindata.size() << std::endl;
  std::cout << std::endl;

  //// SETUP MODEL ////
  fl::CategoricalCrossEntropy criterion;
  Transformer model(args.d_model, args.d_ff, args.n_heads, bptt,
                    args.n_blocks, dict.size(), args.dropout);
  AdagradOptimizer opt(model.params(), 0.0, args.clip, args.epsilon);

  //// VALIDATION/TEST LOOP ////
  auto valid_loop = [&](CharLMDataset& data) {
    double meter = 0.0;
    model.eval();

    std::vector<fl::Variable> cache;
    for (size_t i = 0; i < data.size(); i++) {
      auto example = data.get(i);
      auto input = fl::noGrad(example[0]);
      auto target = fl::noGrad(example[1]);

      if (!args.use_cache) cache.clear();
      cache.push_back(input);
      cache = model.forward(cache);

      auto loss = ngram > 1 ? myloss(cache.back(), target)
        : criterion(logSoftmax(cache.back(), 0), target.slice(0));

      meter += loss.array().scalar<float>();

      cache.pop_back();
      for (auto& h : cache) h.setCalcGrad(false);
    }

    return meter / data.size() / std::log(2.0);
  };

  //// TRAIN LOOP ////
  int64_t iter = 1;
  double best_valid_loss = 1e8, test_loss = 1e8;
  for (int epoch = 0; epoch < args.nepoch; epoch++) {
    float meter = 0.0;

    model.train();
    traindata.set_offset(std::rand() % (bptt-1));
    if (epoch >= args.nepoch - 10 || epoch == args.warmup_loss) {
      lr /= 2;
    }

    std::vector<fl::Variable> cache;
    for (size_t i = 0; i < traindata.size(); i++, iter++) {
      opt.setLr(lr * std::min(iter / double(warmup), 1.0));

      auto example = traindata.get(i % traindata.size());
      auto input = fl::noGrad(example[0]);
      auto target = fl::noGrad(example[1]);

      if (!args.use_cache || iter < warmup) cache.clear();
      cache.push_back(input);
      cache = model.forward(cache);

      auto loss = ngram > 1 ? myloss(cache.back(), target,
                                     epoch < args.warmup_loss, false)
        : criterion(logSoftmax(cache.back(), 0), target.slice(0));

      meter += loss.array().scalar<float>() / std::log(2.0);
      opt.zeroGrad();
      loss.backward();
      opt.step();

      cache.pop_back();
      for (auto& h : cache) h.setCalcGrad(false);
    }
    //// PRINT INFORMATION ////
    double valid_loss = valid_loop(validdata);
    if (valid_loss < best_valid_loss) {
      best_valid_loss = valid_loss;
      test_loss = valid_loop(testdata);
    }
    std::cout << "    epoch: " << epoch
              << "    iter: " << iter
              << "    train: " << meter / traindata.size()
              << "    valid: " << valid_loss
              << "    best valid: " << best_valid_loss
              << "    test: " << test_loss
              << "    learning rate: " << lr
              << std::endl;
  }
  return 0;
}
