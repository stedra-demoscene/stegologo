#pragma once
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <string>
#include <vector>
#include "stego/stego.hpp"

namespace stego {

constexpr int NT = 15;         // poly-4 total-degree Legendre terms
constexpr int NCOL = 2 * NT + 1;  // 31 regressors (N0*phi, N1*phi, dot gain)
constexpr int NPAD = 32;

struct Timer {
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  double ms() const { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
  void reset() { t0 = std::chrono::steady_clock::now(); }
};

struct Asset {
  // meta
  int version = 1;
  std::string region;                 // per-region key id (v2 assets), empty for legacy shared-pilot assets
  double page[2] = {0, 0};
  int n_chips = 0, n_sym = 0, n_bits = 0;
  double ref_ppm = 20;
  double W = 0.2, step = 0.01, hp_chips = 4, clip_mad = 3;
  int n_iter = 4;
  bool has_zacc = false;
  double z_accept = 0;
  std::string code = "golay_crc16";   // or "rep_crc8"
  double rho = 1.0;                   // pilot ratio (rep_crc8)
  // samples (every 2nd contour point of chip-carrying contours), SoA float
  int ns = 0;
  std::vector<float> Px, Py, Nx, Ny, w0, esyn, wa, wb;
  std::vector<int32_t> cid, ca, cb;
  std::vector<uint8_t> nc, valid0;
  std::vector<double> tk;
  std::vector<int32_t> contour_n;
  std::vector<double> contour_chip_len;
  std::vector<int32_t> contour_start;   // sample range start per contour (size nc+1)
  std::vector<int32_t> hp_win;           // high-pass window (samples) per contour
  // chips
  std::vector<float> pilot, pn;
  std::vector<uint8_t> pilot_mask;
  std::vector<int32_t> bit_of;
  std::vector<uint8_t> golay;  // 4096*24 (0/1)
  // reference raster: coverage*4 at ref_ppm
  int rw = 0, rh = 0;
  std::vector<uint8_t> dref;
  // derived
  std::vector<float> reg;      // ns x NPAD  [N0*phi(15), N1*phi(15), 1, 0]
  std::vector<float> phi;      // ns x 16 Legendre poly basis (15 + pad)
  std::vector<float> golay_pm; // 4096*24 codebook as +-1
  double cen[2] = {0, 0};
  std::vector<int32_t> fit_idx;   // samples with valid0 && !nc && w0>0 (the only ones that can ever carry weight)
  // reference moments and coarse templates
  double ref_area_mm2 = 0, ref_c[2] = {0, 0}, ref_phi = 0;
  bool square_block = true;
  int cw_ = 0, ch_ = 0;            // coarse template dims (2.5 ppm blurred ref, band points)
  std::vector<float> ctx, cty, ctv;  // template points (mm) and blurred ref values (centred, unit norm)
  std::vector<float> ctx2, cty2, ctv2;  // every 2nd point (rotation sweep)
  // edge-registration validity lists per stage window (index into samples, subsampled)
  struct RegSet { double W; int stride; std::vector<int32_t> idx; std::vector<float> feat; /* n x 12 */ };
  mutable std::deque<RegSet> regsets;
  const RegSet& regset(double W, int stride) const;
};

struct Mat3 { double m[9]; };

inline void h_identity(double* H) { for (int i = 0; i < 9; i++) H[i] = (i % 4 == 0); }
inline void h_mul(const double* A, const double* B, double* C) {
  double T[9];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) T[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
  std::memcpy(C, T, sizeof T);
}
inline void h_norm(double* H) { double s = H[8]; if (std::fabs(s) > 1e-12) for (int i = 0; i < 9; i++) H[i] /= s; }
bool h_inv(const double* H, double* out);

// image.cpp
struct Dark { std::vector<float> D; int w = 0, h = 0; double paper = 0, ink = 0; };
void darkness(const ImageView& im, Dark& out);

// linalg
bool chol_solve(double* A, double* b, int n, double ridge, double* x);  // A is n x n (row-major, symmetric), destroyed

// fit.cpp: shared samples->edge offset measurement
struct MeasureCfg { double W, step; };
// For samples idx[0..n): edge offset e[i] (mm) of the darkness image along the reference normal (area method);
// ok[i]=0 if the tap line leaves the image.
void measure_edges(const Asset& A, const float* D, int w, int h, const double* H, double W, double step, const int32_t* idx,
                   int n, float* e, uint8_t* ok);

// decode.cpp
Result decode_dark(const Asset&, const Dark&, const Params&, const double* H, double* reg_score, Timer& total);

// register.cpp
RegResult register_dark(const Asset&, const Dark&, const Params&, const double* H_init);

}  // namespace stego
