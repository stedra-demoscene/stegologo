// stego core: shared native decoder (registration + nuisance fit + despread) for Python (pybind11) and Android (JNI).
// No OpenCV. Deterministic, single-threaded. See core/README.md.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace stego {

struct Asset;  // reference geometry + chip layout + code tables (opaque; see asset.cpp)

// Load the reference asset written by tools/export_android_ref.py (<name>.json + <name>.bin).
std::shared_ptr<const Asset> load_asset(const std::string& json_path, const std::string& bin_path = "");
std::shared_ptr<const Asset> load_asset_mem(const std::string& json_text, const uint8_t* bin, size_t bin_len);
double asset_page_w(const Asset&);
double asset_page_h(const Asset&);

// Row-major single-channel image view (uint8 or float32, values 0..255).
struct ImageView {
  const void* data = nullptr;
  int w = 0, h = 0;
  int stride = 0;  // elements per row (0 = w)
  bool is_float = false;
};

struct Params {
  double W = 0.0, step = 0.01;  // final-fit edge half-window (mm; 0 = the asset's own W, 0.2 for legacy assets) / tap spacing (mm)
  double iter_step = 0.02;      // tap spacing in the non-final iterations
  int n_iter = 3;
  int fit_stride = 2;     // final nuisance fit: every k-th sample enters the normal equations (residuals use all)
  int warp_deg = 4;       // fixed by the asset (poly total degree 4)
  double hp_chips = 4.0;
  double clip_mad = 3.0;
  int iter_stride = 4;    // use every k-th sample in the non-final nuisance iterations
  // registration
  std::vector<double> reg_W = {1.2, 0.6, 0.3};   // edge-fit windows, coarse to fine (mm)
  std::vector<int> reg_iters = {3, 3, 2};
  std::vector<double> reg_step = {0.1, 0.05, 0.03};
  int coarse_rounds = 0;  // greedy NCC refinement rounds of the coarse pose (0 = parabolic angle interpolation only)
  int reg_stride = 8;     // sample stride of the registration stages
  int rot_step_deg = 10;  // rotation sweep step for square-ish blocks
  int n_cands = 3;        // best coarse candidates that are refined (square blocks)
  bool local_norm = false;
  bool refine_init = false;  // with H_init: still run the edge-fit stages from it (capture-range tests, multi-frame)
};

struct Timing {  // milliseconds
  double darkness = 0, coarse = 0, edge = 0, measure = 0, fit = 0, tail = 0, golay = 0, total = 0;
  double register_total() const { return coarse + edge; }
};

struct Result {
  std::string hex;
  bool crc_ok = false, accept = false;
  double z_pilot = 0, amp_um = 0, amp_raw_um = 0, hp_retention = 0;
  int n_valid = 0, n_points = 0;
  std::vector<float> soft;   // per coded symbol
  std::vector<uint8_t> bits; // payload bits
  double H[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // final ref mm -> crop px (pixel i covers [i, i+1))
  double reg_score = 0;      // registration quality (edge-fit inlier fraction)
  int reg_cand = 0;
  // matched-filter chip statistics after calibration (for accumulating several decodes of one block) and the pilot residual variance
  std::vector<double> chip_y, chip_w;
  double pilot_s2 = 0;
  // decoy-pilot null calibration: z_cal = (z_pilot - null_mean) / null_sd over 32 fixed random +-1 sequences on the pilot chips; accept uses z_cal
  double z_cal = 0, null_mean = 0, null_sd = 1;
  Timing t;
};

// Full decode of one block crop. H_init (3x3 row-major, ref mm -> crop px) skips the coarse search when given.
Result decode(const Asset&, const ImageView&, const Params& = Params(), const double* H_init = nullptr);

// Registration only (coarse + edge fit); returns the quality score, H in out.
struct RegResult { double H[9]; double score = 0; int cand = 0; Timing t; };
RegResult register_block(const Asset&, const ImageView&, const Params& = Params(), const double* H_init = nullptr);

// Multi-frame: K crops of (roughly) the same block, registered individually to the reference with the edge fit,
// resampled into crop 0's frame, averaged, then decoded. Crops need not be the same size.
Result decode_multi(const Asset&, const ImageView* crops, int k, const Params& = Params());

// Accumulate: sum a_k*chip_y / a_k*chip_w over decodes (a_k = 1/pilot_s2 of each) and call this: pilot z + despread + Golay + CRC + accept.
// Returns hex/crc/accept/z/amp/soft/bits (H and timings are not set).
Result finalize_chips(const Asset&, const double* y, const double* w);

// Helpers exposed for tests.
std::string bits_to_hex(const std::vector<uint8_t>& bits);

}  // namespace stego
