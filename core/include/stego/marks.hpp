// Mark detector: finds the pose (centre, scale, rotation, affine foreshortening) of a known compact mark (rings, guilloche, op-art, swirl...)
// in a grayscale frame or crop. Port of geostego/marks_detect.py without OpenCV. The result H (asset page mm -> image px, pixel i covers
// [i, i+1)) goes to stego::decode(asset, image, params, H) (then optionally refine_init). See core/README.md.
#pragma once
#include <array>
#include <memory>
#include <vector>
#include "stego.hpp"

namespace stego {

struct MarkParams {
  double ppm_lo = 8, ppm_hi = 24, step = 1.18;  // scale candidates (frame px per mm of the mark)
  int sym = 1;            // n-fold rotational symmetry of the geometry (about sym_centre): returns `sym` rotation hypotheses (hyps)
  bool affine = true;     // affine refinement (tilt / anisotropic foreshortening) after the similarity fit
  int n_hyp = 12;         // distinct rotation peaks of the polar stage that are screened by the oriented-template NCC
  int n_gn = 3;           // the best few of those get Gauss-Newton similarity iterations (the best by NCC wins)
  bool homography = true; // projective refinement (8 params) after the affine fit (strong tilt)
  int pad = 4;            // half window (px at the 2.5 px/mm working scale) of the centre search around the stage-A/polar estimate
  double sym_cx = -1, sym_cy = -1;  // symmetry centre (asset page mm); <0 = page centre
  std::vector<std::array<double, 3>> exclude;  // already-handled marks: (cx, cy, radius) in view px; stage-A peaks whose centre falls inside are ignored
};

struct MarkTiming { double prep = 0, stage_a = 0, polar = 0, fine = 0, ecc = 0, total = 0; int n_scales = 0, n_fft = 0; };

struct MarkResult {
  bool found = false;
  double score = 0;      // zero-mean NCC of the template and the warped frame at the final pose (the detection statistic)
  double iso = 0;        // stage-A rotation-averaged NCC peak
  double sim_score = 0, aff_score = 0, hom_score = 0;
  double H[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  double centre[2] = {0, 0};      // image px of the page centre
  double ppm = 0, rot_deg = 0;    // sqrt|det| of the linear part (px/mm), atan2(H10,H00)
  double scale_ppm = 0;           // stage-A scale candidate
  std::vector<std::array<double, 9>> hyps;   // sym rotation hypotheses (hyps[0] = H)
  MarkTiming t;
};

class MarkDetector {
 public:
  explicit MarkDetector(const Asset&, double blur_mm = 0.45);
  ~MarkDetector();
  MarkDetector(const MarkDetector&) = delete;
  MarkResult detect(const ImageView&, const MarkParams& = MarkParams()) const;
  struct Impl;
 private:
  std::unique_ptr<Impl> p_;
};

// Detect + decode in one call (the live path): detect the mark pose in `full`, crop around it (no copy: the crop is a view into `full`),
// decode against each candidate asset (same geometry, different regions/keys) from every symmetry hypothesis; best = accepted, then highest z_cal.
struct MarkDecodeOpts {
  MarkParams det;
  Params dec;
  double crop_margin_mm = 3.0;   // around the page rectangle's image (clipped to the frame)
  bool refine = true;            // retry the best candidate with refine_init when it is not accepted
};
struct MarkDecodeResult {
  MarkResult det;
  Result dec;                    // of the best (asset, hypothesis)
  int asset = -1, hyp = -1;      // indices; -1 = no decode was run
  int crop[4] = {0, 0, 0, 0};    // x0, y0, x1, y1 in `full`
  bool refined = false;
  double ms_decode = 0, ms_total = 0;
  int n_decodes = 0;
};
MarkDecodeResult detect_and_decode(const MarkDetector&, const std::vector<const Asset*>& assets, const ImageView& full, const MarkDecodeOpts& = MarkDecodeOpts());
// Decode only (pose given): H = page mm -> `full` px.
MarkDecodeResult decode_at(const std::vector<const Asset*>& assets, const ImageView& full, const MarkResult& det, const MarkDecodeOpts& = MarkDecodeOpts());

// FFT self-test helper (max abs error of fft2 vs a naive DFT for an nx x ny grid); used by the unit test.
double mark_fft_selftest(int nx, int ny);

}  // namespace stego
