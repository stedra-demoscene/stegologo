#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "internal.hpp"
#include "json.hpp"

namespace stego {

namespace {
std::string read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

struct Arr {
  const uint8_t* p;
  std::string dtype;
  size_t n;
  double get(size_t i) const {
    if (dtype == "f64") { double v; std::memcpy(&v, p + 8 * i, 8); return v; }
    if (dtype == "i32") { int32_t v; std::memcpy(&v, p + 4 * i, 4); return v; }
    return p[i];
  }
};

template <class T>
void to(std::vector<T>& dst, const Arr& a) {
  dst.resize(a.n);
  for (size_t i = 0; i < a.n; i++) dst[i] = (T)a.get(i);
}

// separable Gaussian (reflect-less: clamp) on a float image, in place via tmp
void gauss(std::vector<float>& img, int w, int h, double sigma) {
  int r = std::max(1, (int)std::ceil(3 * sigma));
  std::vector<float> k(2 * r + 1);
  double s = 0;
  for (int i = -r; i <= r; i++) { k[i + r] = (float)std::exp(-0.5 * i * i / (sigma * sigma)); s += k[i + r]; }
  for (auto& v : k) v = (float)(v / s);
  std::vector<float> tmp((size_t)w * h);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      float a = 0;
      for (int i = -r; i <= r; i++) a += k[i + r] * img[(size_t)y * w + std::clamp(x + i, 0, w - 1)];
      tmp[(size_t)y * w + x] = a;
    }
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      float a = 0;
      for (int i = -r; i <= r; i++) a += k[i + r] * tmp[(size_t)std::clamp(y + i, 0, h - 1) * w + x];
      img[(size_t)y * w + x] = a;
    }
}
}  // namespace

std::shared_ptr<const Asset> load_asset(const std::string& json_path, const std::string& bin_path) {
  std::string js = read_file(json_path);
  std::string bp = bin_path;
  if (bp.empty()) { bp = json_path; bp.replace(bp.size() - 4, 4, "bin"); }
  std::string bin = read_file(bp);
  return load_asset_mem(js, (const uint8_t*)bin.data(), bin.size());
}

std::shared_ptr<const Asset> load_asset_mem(const std::string& json_text, const uint8_t* bin, size_t bin_len) {
  JParser jp(json_text);
  JVal m = jp.parse();
  auto A = std::make_shared<Asset>();
  const JVal& key = m.at("key");
  A->version = m.has("version") ? (int)m.at("version").n : 1;
  if (A->version > 2) throw std::runtime_error("asset version " + std::to_string(A->version) + " is newer than this core supports (2)");
  // v2: per-region keys; the pilot / pn / bit_of / pilot_mask arrays are already this region's (derived by the exporter from SHA-256(seed, region))
  if (key.has("region") && key.at("region").t == JVal::Str) A->region = key.at("region").s;
  A->code = key.has("code") ? key.at("code").s : "rep_crc8";
  if (key.has("pilot_ratio")) A->rho = key.at("pilot_ratio").n;
  A->page[0] = m.at("page").a[0].n; A->page[1] = m.at("page").a[1].n;
  A->n_chips = (int)m.at("n_chips").n; A->n_sym = (int)m.at("n_sym").n; A->n_bits = (int)m.at("n_bits").n;
  A->ref_ppm = m.at("ref_ppm").n;
  const JVal& pr = m.at("params");
  A->W = pr.at("W").n; A->step = pr.at("step").n; A->hp_chips = pr.at("hp_chips").n; A->clip_mad = pr.at("clip_mad").n;
  A->n_iter = (int)pr.at("n_iter").n;
  if (pr.has("z_accept") && pr.at("z_accept").t == JVal::Num) { A->has_zacc = true; A->z_accept = pr.at("z_accept").n; }
  A->ns = (int)m.at("n_samples").n;
  const JVal& ix = m.at("arrays");
  auto get = [&](const char* name) {
    const JVal& e = ix.at(name);
    size_t off = (size_t)e.at("offset").n, nb = (size_t)e.at("nbytes").n;
    if (off + nb > bin_len) throw std::runtime_error("asset bin truncated");
    size_t n = 1;
    for (auto& d : e.at("shape").a) n *= (size_t)d.n;
    return Arr{bin + off, e.at("dtype").s, n};
  };
  const int ns = A->ns;
  {
    Arr P = get("P"), N = get("N");
    A->Px.resize(ns); A->Py.resize(ns); A->Nx.resize(ns); A->Ny.resize(ns);
    for (int i = 0; i < ns; i++) {
      A->Px[i] = (float)P.get(2 * i); A->Py[i] = (float)P.get(2 * i + 1);
      A->Nx[i] = (float)N.get(2 * i); A->Ny[i] = (float)N.get(2 * i + 1);
    }
  }
  to(A->w0, get("w0")); to(A->esyn, get("esyn")); to(A->wa, get("wa")); to(A->wb, get("wb"));
  to(A->cid, get("cid")); to(A->ca, get("ca")); to(A->cb, get("cb"));
  to(A->nc, get("nc")); to(A->valid0, get("valid0")); to(A->tk, get("tk"));
  to(A->contour_n, get("contour_n")); to(A->contour_chip_len, get("contour_chip_len"));
  to(A->pilot, get("pilot")); to(A->pn, get("pn")); to(A->pilot_mask, get("pilot_mask")); to(A->bit_of, get("bit_of"));
  to(A->golay, get("golay"));
  {
    const JVal& e = ix.at("dref");
    A->rh = (int)e.at("shape").a[0].n; A->rw = (int)e.at("shape").a[1].n;
    to(A->dref, get("dref"));
  }
  A->golay_pm.resize(A->golay.size());
  for (size_t i = 0; i < A->golay.size(); i++) A->golay_pm[i] = A->golay[i] ? 1.f : -1.f;
  // contour ranges + high-pass windows
  int nco = (int)A->contour_n.size();
  A->contour_start.assign(nco + 1, 0);
  for (int k = 0; k < nco; k++) A->contour_start[k + 1] = A->contour_start[k] + A->contour_n[k];
  A->hp_win.resize(nco);
  for (int k = 0; k < nco; k++) {
    long r = (long)std::nearbyint(A->hp_chips * A->contour_chip_len[k] / (2 * 0.01));
    long win = (std::max(3L, r)) | 1L;
    A->hp_win[k] = (int)std::min<long>(A->contour_n[k], win);
  }
  // poly-4 Legendre regressors
  A->cen[0] = A->page[0] / 2; A->cen[1] = A->page[1] / 2;
  A->reg.assign((size_t)ns * NPAD, 0.f);
  A->phi.assign((size_t)ns * 16, 0.f);
  std::vector<std::pair<int, int>> terms;
  for (int i = 0; i <= 4; i++) for (int j = 0; j <= 4 - i; j++) terms.push_back({i, j});
  for (int s = 0; s < ns; s++) {
    double x = (A->Px[s] - A->cen[0]) / A->cen[0], y = (A->Py[s] - A->cen[1]) / A->cen[1];
    double Lx[5], Ly[5];
    Lx[0] = Ly[0] = 1; Lx[1] = x; Ly[1] = y;
    for (int k = 2; k <= 4; k++) {
      Lx[k] = ((2 * k - 1) * x * Lx[k - 1] - (k - 1) * Lx[k - 2]) / k;
      Ly[k] = ((2 * k - 1) * y * Ly[k - 1] - (k - 1) * Ly[k - 2]) / k;
    }
    float* r = &A->reg[(size_t)s * NPAD];
    for (int t = 0; t < NT; t++) {
      double phi = Lx[terms[t].first] * Ly[terms[t].second];
      A->phi[(size_t)s * 16 + t] = (float)phi;
      r[t] = (float)(A->Nx[s] * phi);
      r[NT + t] = (float)(A->Ny[s] * phi);
    }
    r[2 * NT] = 1.f;
  }
  for (int s = 0; s < ns; s++)
    if (A->valid0[s] && !A->nc[s] && A->w0[s] > 0) A->fit_idx.push_back(s);
  // reference moments
  {
    double m0 = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (int y = 0; y < A->rh; y++)
      for (int x = 0; x < A->rw; x++) {
        double d = A->dref[(size_t)y * A->rw + x] * 0.25;
        if (d == 0) continue;
        double px = x + 0.5, py = y + 0.5;
        m0 += d; sx += d * px; sy += d * py; sxx += d * px * px; syy += d * py * py; sxy += d * px * py;
      }
    double mx = sx / m0, my = sy / m0;
    double vxx = sxx / m0 - mx * mx, vyy = syy / m0 - my * my, vxy = sxy / m0 - mx * my;
    A->ref_area_mm2 = m0 / (A->ref_ppm * A->ref_ppm);
    A->ref_c[0] = mx / A->ref_ppm; A->ref_c[1] = my / A->ref_ppm;
    A->ref_phi = 0.5 * std::atan2(2 * vxy, vxx - vyy);
    A->square_block = std::max(A->page[0], A->page[1]) / std::min(A->page[0], A->page[1]) < 2.0;
  }
  // coarse NCC template: decimate dref by 4 (5 ppm), blur sigma 0.5 mm, keep every 2nd cell in the band
  {
    const int f = 4;
    int cw = A->rw / f, ch = A->rh / f;
    std::vector<float> c5((size_t)cw * ch, 0.f);
    for (int y = 0; y < ch; y++)
      for (int x = 0; x < cw; x++) {
        int s = 0;
        for (int dy = 0; dy < f; dy++) for (int dx = 0; dx < f; dx++) s += A->dref[(size_t)(y * f + dy) * A->rw + x * f + dx];
        c5[(size_t)y * cw + x] = s / (4.0f * f * f);
      }
    double cell_mm = f / A->ref_ppm;
    std::vector<float> bl = c5, band = c5;
    gauss(bl, cw, ch, 0.5 / cell_mm);
    gauss(band, cw, ch, (2 * 0.5 + 0.3) / cell_mm);
    double sum = 0;
    for (int y = 0; y < ch; y += 2)
      for (int x = 0; x < cw; x += 2)
        if (band[(size_t)y * cw + x] > 0.002f) {
          A->ctx.push_back((float)((x + 0.5) * cell_mm)); A->cty.push_back((float)((y + 0.5) * cell_mm));
          float v = bl[(size_t)y * cw + x]; A->ctv.push_back(v); sum += v;
        }
    double mean = sum / A->ctv.size(), nrm = 0;
    for (auto& v : A->ctv) { v -= (float)mean; nrm += (double)v * v; }
    nrm = std::sqrt(nrm);
    for (auto& v : A->ctv) v = (float)(v / nrm);
    {   // sub-template for the rotation sweep: every 2nd point, centred and normalised on its own
      double s2 = 0;
      size_t n2 = 0;
      for (size_t j = 0; j < A->ctv.size(); j += 2) { s2 += A->ctv[j]; n2++; }
      double m2 = s2 / n2, q2 = 0;
      for (size_t j = 0; j < A->ctv.size(); j += 2) q2 += (A->ctv[j] - m2) * (A->ctv[j] - m2);
      q2 = std::sqrt(q2);
      for (size_t j = 0; j < A->ctv.size(); j += 2) {
        A->ctx2.push_back(A->ctx[j]); A->cty2.push_back(A->cty[j]); A->ctv2.push_back((float)((A->ctv[j] - m2) / q2));
      }
    }
  }
  return A;
}

double asset_page_w(const Asset& a) { return a.page[0]; }
double asset_page_h(const Asset& a) { return a.page[1]; }

// Validity of samples for a wide edge window: the reference must be one clean edge (ink on the inner side, paper outside)
// out to W + 0.1 mm, looked up in the 20 ppm reference raster (nearest pixel).
const Asset::RegSet& Asset::regset(double Wm, int stride) const {
  for (auto& r : regsets) if (r.W == Wm && r.stride == stride) return r;
  RegSet rs;
  rs.W = Wm; rs.stride = stride;
  auto ink = [&](double x, double y) {
    int ix = (int)std::floor(x * ref_ppm), iy = (int)std::floor(y * ref_ppm);
    if (ix < 0 || iy < 0 || ix >= rw || iy >= rh) return false;
    return dref[(size_t)iy * rw + ix] >= 2;
  };
  for (int s = 0; s < ns; s += stride) {
    if (nc[s] || w0[s] <= 0) continue;
    bool ok = true;
    for (double t = 0.03; t <= Wm + 0.1 + 1e-9 && ok; t += 0.02) {
      if (!ink(Px[s] - t * Nx[s], Py[s] - t * Ny[s])) ok = false;
      else if (ink(Px[s] + t * Nx[s], Py[s] + t * Ny[s])) ok = false;
    }
    if (ok) rs.idx.push_back(s);
  }
  rs.feat.assign(rs.idx.size() * 12, 0.f);
  for (size_t i = 0; i < rs.idx.size(); i++) {
    const int s = rs.idx[i];
    const double X = Px[s] - cen[0], Y = Py[s] - cen[1], nx = Nx[s], ny = Ny[s], nX = nx * X + ny * Y;
    float* f = &rs.feat[i * 12];
    f[0] = (float)nx; f[1] = (float)ny; f[2] = (float)(nx * X); f[3] = (float)(nx * Y); f[4] = (float)(ny * X); f[5] = (float)(ny * Y);
    f[6] = (float)(-nX * X); f[7] = (float)(-nX * Y); f[8] = 1.f;
  }
  regsets.push_back(std::move(rs));
  return regsets.back();
}

std::string bits_to_hex(const std::vector<uint8_t>& bits) {
  std::string s;
  size_t nh = (bits.size() + 3) / 4;
  // value is MSB-first; left-pad to nh nibbles
  size_t pad = nh * 4 - bits.size();
  for (size_t h = 0; h < nh; h++) {
    int v = 0;
    for (int b = 0; b < 4; b++) {
      long i = (long)(h * 4 + b) - (long)pad;
      v = (v << 1) | (i >= 0 ? bits[i] : 0);
    }
    s += "0123456789abcdef"[v];
  }
  return s;
}

}  // namespace stego
