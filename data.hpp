// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//

#include <flashlight/flashlight.h>

#include <codecvt>
#include <exception>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>

class Dict {

public:
  Dict()
  {
    to_int(eos, true);
    to_int(unk, true);
  }

  void build_dict(const std::string& filename, int32_t ngram)
  {
    std::wifstream fin(filename);
    fin.imbue(std::locale("en_US.UTF-8"));
    if (!fin.is_open()) {
      throw af::exception("Cannot open dataset file.");
    }

    for (std::wstring line; std::getline(fin, line); ) {
      for (size_t i = 0; i < line.size(); i++) {
        for (size_t j = 1; j <= ngram; j++) {
          if (i + j <= line.size()) {
            to_int(line.substr(i, j), true);
          } else {
            int32_t offset = line.size() - i;
            to_int(line.substr(i, offset) + unk, true);
          }
        }
      }
    }
  }

  int32_t to_int(std::wstring word, bool append = false, int c = 1)
  {
    if (word2int.find(word) == word2int.end()) {
      if (append) {
        word2int[word] = int2word.size();
        int2word.push_back(word);
        counts.push_back(0);
      } else {
        return word2int[unk];
      }
    }
    int32_t idx = word2int[word];
    if (append) {
      counts[idx] += c;
    }
    return idx;
  }

  Dict threshold(int64_t t)
  {
    Dict dict;
    for (int i = 2; i < size(); i++) {
      if (counts[i] >= t || int2word[i].length() == 1) {
        dict.to_int(int2word[i], true, counts[i]);
      }
    }
    return dict;
  }

  std::wstring to_string(int32_t idx)
  {
    return int2word[idx];
  }

  size_t size() const
  {
    return int2word.size();
  }

  static const std::wstring eos, unk;

private:
  std::unordered_map<std::wstring, int32_t> word2int;
  std::vector<std::wstring> int2word;
  std::vector<int64_t> counts;
};

const std::wstring Dict::eos = L"</s>";
const std::wstring Dict::unk = L"<unk>";

class CharLMDataset : public fl::Dataset {

public:
  CharLMDataset(const std::string& filename, Dict& dict,
                int32_t bsz, int32_t bptt, int32_t ngram)
    : bsz(bsz)
    , bptt(bptt)
    , offset(0)
  {
    std::wifstream fin(filename);
    fin.imbue(std::locale("en_US.UTF-8"));
    if (!fin.is_open()) {
      throw af::exception("Cannot open dataset file.");
    }

    std::vector<int32_t> chars;
    for (std::wstring line; std::getline(fin, line); ) {
      for (size_t i = 0; i < line.size(); i++) {
        for (size_t j = 1; j <= ngram; j++) {
          if (i + j <= line.size()) {
            chars.push_back(dict.to_int(line.substr(i, j)));
          } else {
            int32_t offset = line.size() - i;
            chars.push_back(dict.to_int(line.substr(i, offset) + dict.unk));
          }
        }
      }
    }

    dim_t n_batch = chars.size() / bsz / ngram;
    chars.resize(n_batch * bsz * ngram);
    data = af::reorder(af::array(ngram, n_batch, bsz, chars.data()), 1, 2, 0);
  }

  int64_t size() const override
  {
    return (data.dims(0) - offset - 1) / bptt;
  }

  std::vector<af::array> get(const int64_t idx) const
  {
    int s = idx * bptt + offset, e = (idx + 1) * bptt + offset;
    return { data(af::seq(s, e-1), af::span, 0),
             data(af::seq(s+1, e), af::span, af::span)};
  }

  void set_offset(int32_t i)
  {
    offset = i;
  }

private:
  int32_t bsz, bptt, offset;
  af::array data;
};
