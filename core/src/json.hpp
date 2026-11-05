// Minimal JSON reader (objects, arrays, strings, numbers, bool, null) for the asset index.
#pragma once
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace stego {
struct JVal {
  enum T { Null, Bool, Num, Str, Arr, Obj } t = Null;
  bool b = false;
  double n = 0;
  std::string s;
  std::vector<JVal> a;
  std::map<std::string, JVal> o;
  const JVal& at(const std::string& k) const {
    auto it = o.find(k);
    if (it == o.end()) throw std::runtime_error("json: missing key " + k);
    return it->second;
  }
  bool has(const std::string& k) const { return o.count(k) > 0; }
};

class JParser {
 public:
  explicit JParser(const std::string& s) : s_(s) {}
  JVal parse() { JVal v = value(); return v; }

 private:
  const std::string& s_;
  size_t i_ = 0;
  void ws() { while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\n' || s_[i_] == '\t' || s_[i_] == '\r')) i_++; }
  void expect(char c) { ws(); if (i_ >= s_.size() || s_[i_] != c) throw std::runtime_error("json: expected char"); i_++; }
  std::string str() {
    expect('"');
    std::string r;
    while (i_ < s_.size() && s_[i_] != '"') {
      if (s_[i_] == '\\') {
        i_++;
        char c = s_[i_];
        if (c == 'n') r += '\n'; else if (c == 't') r += '\t'; else if (c == 'u') { r += '?'; i_ += 4; } else r += c;
        i_++;
      } else r += s_[i_++];
    }
    i_++;
    return r;
  }
  JVal value() {
    ws();
    JVal v;
    char c = s_[i_];
    if (c == '{') {
      v.t = JVal::Obj; i_++; ws();
      if (s_[i_] == '}') { i_++; return v; }
      for (;;) { ws(); std::string k = str(); expect(':'); v.o[k] = value(); ws(); if (s_[i_] == ',') { i_++; continue; } expect('}'); break; }
    } else if (c == '[') {
      v.t = JVal::Arr; i_++; ws();
      if (s_[i_] == ']') { i_++; return v; }
      for (;;) { v.a.push_back(value()); ws(); if (s_[i_] == ',') { i_++; continue; } expect(']'); break; }
    } else if (c == '"') { v.t = JVal::Str; v.s = str(); }
    else if (s_.compare(i_, 4, "true") == 0) { v.t = JVal::Bool; v.b = true; i_ += 4; }
    else if (s_.compare(i_, 5, "false") == 0) { v.t = JVal::Bool; v.b = false; i_ += 5; }
    else if (s_.compare(i_, 4, "null") == 0) { i_ += 4; }
    else { size_t j; v.t = JVal::Num; v.n = std::stod(s_.substr(i_), &j); i_ += j; }
    return v;
  }
};
}  // namespace stego
