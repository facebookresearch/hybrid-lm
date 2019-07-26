// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//

#include <flashlight/flashlight.h>

fl::Variable init_linear(int32_t d_in, int32_t d_out)
{
  float std = std::sqrt(1.0 / float(d_in));
  return fl::uniform(d_out, d_in, -std, std);
}

fl::Variable rotate(const fl::Variable& input)
{
  auto data = input.array();
  int d0 = data.dims(0);
  int d1 = data.dims(1);
  int d2 = data.dims(2);
  int d3 = data.dims(3);
  data = af::join(0, data, af::constant(0.0, d1 - d0 + 1, d1, d2, d3));
  data = af::moddims(data, af::dim4(d1, d1 + 1, d2, d3));
  data = af::lower(data, false);
  data = data(af::seq(d1-d0, d1-1), af::seq(0, d1-1), af::span, af::span);
  auto grad_func = [d0, d1, d2, d3](std::vector<fl::Variable> &inputs,
                                    const fl::Variable &grad_output) {
    auto grad_data = grad_output.array();
    auto grad = af::constant(0.0, d1, d1 + 1, d2, d3);
    grad(af::seq(d1-d0, d1-1), af::seq(0, d1-1), af::span, af::span) = grad_data;
    grad = af::lower(grad, false);
    grad = af::moddims(grad, af::dim4(d1 + 1, d1, d2, d3));
    grad = grad(af::seq(0, d0-1), af::seq(0, d1-1), af::span, af::span);
    inputs[0].addGrad(fl::Variable(grad, false));
  };
  return fl::Variable(data, {input}, grad_func);
}

class LayerNorm : public fl::UnaryModule {
public:
  LayerNorm(int32_t d_model, bool linear = true)
    : linear_(linear)
  {
    if (linear) {
      params_.push_back(fl::Variable(af::constant(1.0, d_model), true));
      params_.push_back(fl::Variable(af::constant(0.0, d_model), true));
      //params_.push_back(fl::Variable(af::constant(1.0, 1), true));
      //params_.push_back(fl::Variable(af::constant(0.0, 1), true));
    }
  }

  fl::Variable forward(const fl::Variable& input)
  {
    auto m = mean(input, {0});
    auto v = sqrt(var(input, {0})) + 1e-5;
    if (linear_) {
      return tileAs(params_[0], input) * (input - tileAs(m, input)) \
        / tileAs(v, input) + tileAs(params_[1], input);
    } else {
      return (input - tileAs(m, input)) / (tileAs(v, input));
    }
  }

  fl::Variable operator()(const fl::Variable& input)
  {
    return forward(input);
  }

  std::string prettyString() const override
  {
    return "LayerNorm";
  }

private:
  bool linear_;

  FL_SAVE_LOAD_WITH_BASE(fl::UnaryModule, linear_)
  friend class cereal::access;
  LayerNorm() {}
};

CEREAL_REGISTER_TYPE(LayerNorm);

class MultiAttention : public fl::Container {
public:

  MultiAttention(int32_t d_model, int32_t d_head, int32_t n_heads,
                 float p_dropout, int32_t bptt)
    : d_model(d_model)
    , d_head(d_head)
    , n_heads(n_heads)
    , bptt(bptt)
    , wq(std::make_shared<fl::Linear>(init_linear(d_model, d_head * n_heads)))
    , wk(std::make_shared<fl::Linear>(init_linear(d_model, d_head * n_heads)))
    , wv(std::make_shared<fl::Linear>(init_linear(d_model, d_head * n_heads)))
    , wf(std::make_shared<fl::Linear>(init_linear(d_head * n_heads, d_model)))
    , dropout(std::make_shared<fl::Dropout>(p_dropout))
  {
    params_.push_back(fl::uniform(bptt, d_head, -0.1, 0.1));

    add(wq);
    add(wk);
    add(wv);
    add(wf);
    add(dropout);
  }

  fl::Variable get_mask(int32_t n, int32_t d, bool cache = false)
  {
    auto mask = af::lower(af::constant(1.0, n, n), true);
    if (cache) {
      auto mask_cache = af::upper(af::constant(1.0, n, n));
      mask = af::join(1, mask_cache, mask);
    }
    return fl::Variable(af::log(mask), false);
  }

  std::vector<fl::Variable> forward(const std::vector<fl::Variable>& input)
  {
    int n = input[0].dims(1), bsz = input[0].dims(2);
    auto q = transpose((*wq)(input.back()));
    auto k = transpose((*wk)(concatenate(input, 1)));
    auto v = transpose((*wv)(concatenate(input, 1)));

    q = moddims(q, af::dim4(-1, d_head, n_heads * bsz));
    k = moddims(k, af::dim4(-1, d_head, n_heads * bsz));
    v = moddims(v, af::dim4(-1, d_head, n_heads * bsz));

    auto pos_emb = tile(params_[0], af::dim4(1, 1, n_heads * bsz));
    auto pos_scores = rotate(matmulNT(pos_emb, k));
    auto scores = pos_scores + matmulNT(q, k);
    scores = scores / std::sqrt(float(d_head));

    auto mask = get_mask(n, 4*d_model, input.size() == 2);
    scores = scores + tileAs(mask, scores);
    auto attn = (*dropout)(softmax(scores, 1));
    auto result = matmul(attn, v);

    result = moddims(result, af::dim4(-1, d_head * n_heads, bsz));
    result = (*wf)(transpose(result));

    return {result};
  }

  std::vector<fl::Variable> operator()(const std::vector<fl::Variable>& input)
  {
    return forward(input);
  }

  std::string prettyString() const override
  {
    return "Multi Head Attention";
  }

private:
  int32_t d_model, d_head, n_heads, bptt;
  std::shared_ptr<fl::Linear> wq, wk, wv, wf;
  std::shared_ptr<fl::Dropout> dropout;

  FL_SAVE_LOAD_WITH_BASE(fl::Container, d_model, d_head, n_heads,
                         bptt, wq, wk, wv, wf, dropout)

  friend class cereal::access;
  MultiAttention() {}
};

CEREAL_REGISTER_TYPE(MultiAttention);

class TransformerLayer : public fl::Container {

public:

  TransformerLayer(int32_t d_model, int32_t d_head, int32_t d_mlp,
                   int32_t n_heads, int32_t bptt, float p_dropout)
    : w1(std::make_shared<fl::Linear>(init_linear(d_model, d_mlp)))
    , w2(std::make_shared<fl::Linear>(init_linear(d_mlp, d_model)))
    , dropout(std::make_shared<fl::Dropout>(p_dropout))
    , norm1(std::make_shared<LayerNorm>(d_model))
    , norm2(std::make_shared<LayerNorm>(d_model))
    , attention(std::make_shared<MultiAttention>(d_model, d_head, n_heads,
                                                 p_dropout, bptt))
  {
    add(w1);
    add(w2);
    add(dropout);
    add(norm1);
    add(norm2);
    add(attention);
  }

  fl::Variable relu(const fl::Variable& input)
  {
    return max(input, 0.0);
  }

  fl::Variable mlp(const fl::Variable& input)
  {
    return (*w2)(dropout->forward(relu((*w1)(input))));
  }

  std::vector<fl::Variable> forward(const std::vector<fl::Variable>& input)
  {
    auto x = input.back();
    auto h = (*norm1)(attention->forward(input)[0] + x);
    return { (*norm2)(mlp(h) + h) };
  }

  std::vector<fl::Variable> operator()(const std::vector<fl::Variable>& input)
  {
    return forward(input);
  }

  std::string prettyString() const override
  {
    return "Transformer Layer";
  }

private:
  std::shared_ptr<fl::Linear> w1, w2;
  std::shared_ptr<fl::Dropout> dropout;
  std::shared_ptr<LayerNorm> norm1, norm2;
  std::shared_ptr<MultiAttention> attention;

  FL_SAVE_LOAD_WITH_BASE(fl::Container, w1, w2, dropout, norm1, norm2, attention)
  friend class cereal::access;
  TransformerLayer() {}
};

CEREAL_REGISTER_TYPE(TransformerLayer);

class Transformer : public fl::Container {

public:
  Transformer(int32_t d_model, int32_t d_ff, int32_t n_heads, int32_t bptt,
              int32_t n_layer, int32_t dsz, float p_dropout)
  {
    embedding = std::make_shared<fl::Embedding>(d_model, dsz);
    add(embedding);
    for (size_t i = 0; i < n_layer; i++) {
      std::shared_ptr<TransformerLayer> layer;
      layer = std::make_shared<TransformerLayer>(d_model,
                                                 d_model / n_heads,
                                                 d_ff,
                                                 n_heads,
                                                 bptt,
                                                 p_dropout);
      layers.push_back(layer);
      add(layer);
    }
    classifier = std::make_shared<fl::Linear>(init_linear(d_model, dsz));
    add(classifier);
  }

  std::vector<fl::Variable> forward(const std::vector<fl::Variable>& input)
  {
    std::vector<fl::Variable> outputs;

    // embedding layer
    auto h = embedding->forward(input.back());
    outputs.push_back(h);

    // self attention layers
    for(int i = 0; i < layers.size(); i++) {
      if (input.size() == 1) {
        h = layers[i]->forward(std::vector<fl::Variable>({ h }))[0];
      } else {
        h = layers[i]->forward({ input[i], h })[0];
      }
      outputs.push_back(h);
    }

    // output layer
    h = classifier->forward(h);
    outputs.push_back(h);

    return outputs;
  }

  std::vector<fl::Variable> operator()(const std::vector<fl::Variable>& input)
  {
    return forward(input);
  }

  std::string prettyString() const override
  {
    return "Transformer";
  }

private:
  std::shared_ptr<fl::Embedding> embedding;
  std::vector<std::shared_ptr<TransformerLayer>> layers;
  std::shared_ptr<fl::Linear> classifier;

  FL_SAVE_LOAD_WITH_BASE(fl::Container, embedding, layers, classifier)

  Transformer() {}
  friend class cereal::access;
};

CEREAL_REGISTER_TYPE(Transformer);
