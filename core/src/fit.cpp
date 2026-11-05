// Edge-offset measurement and normal-equation machinery (hot loops; plain C++ written for auto-vectorisation).
#include <algorithm>
#include "internal.hpp"

namespace stego {

bool h_inv(const double* H, double* o) {
  double a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7], i = H[8];
  double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (std::fabs(det) < 1e-300) return false;
  double id = 1.0 / det;
  o[0] = (e * i - f * h) * id; o[1] = (c * h - b * i) * id; o[2] = (b * f - c * e) * id;
  o[3] = (f * g - d * i) * id; o[4] = (a * i - c * g) * id; o[5] = (c * d - a * f) * id;
  o[6] = (d * h - e * g) * id; o[7] = (b * g - a * h) * id; o[8] = (a * e - b * d) * id;
  return true;
}

bool chol_solve(double* A, double* b, int n, double ridge, double* x) {
  for (int i = 0; i < n; i++) A[i * n + i] += ridge;
  for (int j = 0; j < n; j++) {
    double d = A[j * n + j];
    for (int k = 0; k < j; k++) d -= A[j * n + k] * A[j * n + k];
    if (d <= 0) return false;
    d = std::sqrt(d);
    A[j * n + j] = d;
    for (int i = j + 1; i < n; i++) {
      double s = A[i * n + j];
      for (int k = 0; k < j; k++) s -= A[i * n + k] * A[j * n + k];
      A[i * n + j] = s / d;
    }
  }
  for (int i = 0; i < n; i++) {
    double s = b[i];
    for (int k = 0; k < i; k++) s -= A[i * n + k] * x[k];
    x[i] = s / A[i * n + i];
  }
  for (int i = n - 1; i >= 0; i--) {
    double s = x[i];
    for (int k = i + 1; k < n; k++) s -= A[k * n + i] * x[k];
    x[i] = s / A[i * n + i];
  }
  return true;
}

void measure_edges(const Asset& A, const float* D, int w, int h, const double* H, double W, double step, const int32_t* idx,
                   int n, float* e, uint8_t* ok) {
  const int m = (int)std::lround(2 * W / step) + 1;
  const float fw1 = (float)(w - 1), fh1 = (float)(h - 1);
  const float stepf = (float)step;
  for (int i = 0; i < n; i++) {
    const int s = idx[i];
    const double px = A.Px[s], py = A.Py[s], nx = A.Nx[s], ny = A.Ny[s];
    const double ax = H[0] * px + H[1] * py + H[2], bx = H[0] * nx + H[1] * ny;
    const double ay = H[3] * px + H[4] * py + H[5], by = H[3] * nx + H[4] * ny;
    const double ad = H[6] * px + H[7] * py + H[8], bd = H[6] * nx + H[7] * ny;
    const double t0 = -W, t1 = -W + (m - 1) * step;
    const double d0 = ad + t0 * bd, d1 = ad + t1 * bd;
    bool good = d0 > 1e-9 && d1 > 1e-9;
    if (good) {
      double x0 = (ax + t0 * bx) / d0, y0 = (ay + t0 * by) / d0, x1 = (ax + t1 * bx) / d1, y1 = (ay + t1 * by) / d1;
      good = x0 > 1 && x0 < fw1 && x1 > 1 && x1 < fw1 && y0 > 1 && y0 < fh1 && y1 > 1 && y1 < fh1;
    }
    if (!good) { ok[i] = 0; e[i] = 0; continue; }
    ok[i] = 1;
    const float fax = (float)ax, fbx = (float)bx, fay = (float)ay, fby = (float)by, fad = (float)ad, fbd = (float)bd;
    float sum = 0;
    float t = (float)t0;
    for (int k = 0; k < m; k++, t += stepf) {
      const float inv = 1.0f / (fad + t * fbd);
      const float x = (fax + t * fbx) * inv - 0.5f, y = (fay + t * fby) * inv - 0.5f;
      const int ix = (int)x, iy = (int)y;  // x,y >= 0.5 here
      const float fx = x - ix, fy = y - iy;
      const float* r = D + (size_t)iy * w + ix;
      const float a = r[0] + fx * (r[1] - r[0]);
      const float b = r[w] + fx * (r[w + 1] - r[w]);
      sum += a + fy * (b - a);
    }
    e[i] = (float)(sum * step - (W + step / 2));
  }
}

}  // namespace stego
