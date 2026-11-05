// Demo token payload ("Nothing To See Here"): 7 base-62 chars (42 bits) + tag 0b10 (2 bits) + CRC-16 = 60 bits = 5 Golay(24,12) words = 120 symbols,
// the same 120 symbols in every region (own pilot / PN / interleaver). Port of geostego/demo.py combine().
#pragma once
#include <array>
#include <string>
#include <vector>
#include "stego.hpp"

namespace stego {

// keep in sync with geostego/demo.py (TOKEN_LEN, TOKEN_BITS, TAG, TAG_BITS, DEMO_Z_ACCEPT)
constexpr int kTokenLen = 7, kTokenBits = 42, kTagBits = 2, kTag = 2;
constexpr double kDemoZAccept = 8.0;

struct DemoRegion { const Asset* asset = nullptr; const Result* r = nullptr; };   // r == nullptr: region not found (z 0, no LLR)

struct DemoResult {
  bool accept = false, crc_ok = false, tag_ok = false;
  std::string token;          // 7 base-62 chars when crc_ok && tag_ok, else empty
  double z = 0;               // Stouffer sum(z_j) / sqrt(n) over ALL regions (raw pilot z)
  std::vector<double> zs, gains;
  std::vector<uint8_t> bits;  // 60 decoded info bits
};

// Sum the regions' LLRs (2 g c / s2) of those with z >= zmin_use, one soft-ML Golay decode, CRC-16 + tag check.
// Region results must come from decode() (chip_y / chip_w populated).
DemoResult combine_demo(const std::vector<DemoRegion>& regions, double z_accept = kDemoZAccept, double zmin_use = 2.0);

// Whole-design path: `comp` = registration asset of the full design (geometry of all regions in one frame), `regions` = the decode assets,
// shifts[k] = region page frame -> comp page frame translation (mm). H_seed = comp page mm -> `full` px (ORB + RANSAC/MAGSAC against the card render).
// Crops around the design (view, no copy), registers with the edge fit from the seed (wide windows if the first pass is poor), decodes every
// region under that one H and combines.
struct DemoDecodeResult {
  DemoResult comb;
  std::vector<Result> regions;
  double H[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};   // comp page mm -> full px after registration
  double reg_score = 0;
  bool wide = false, ok_crop = false;
  int crop[4] = {0, 0, 0, 0};
  double ms_reg = 0, ms_dec = 0, ms_total = 0;
};
DemoDecodeResult demo_decode(const Asset& comp, const std::vector<const Asset*>& regions, const std::vector<std::array<double, 2>>& shifts,
                             const ImageView& full, const double* H_seed, bool fast = false, double crop_margin_mm = 3.0);

}  // namespace stego
